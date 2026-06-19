"""
Knowledge Graph Academic Assistant — Streamlit Edition
Pipeline: PDF parsing -> KeyBERT concepts -> NetworkX graph -> FAISS -> GraphRAG -> Gemini
"""

import os
import gc
import json
import time
from collections import deque

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import fitz
import numpy as np
import networkx as nx
import faiss
import requests
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

# ── Constants ──────────────────────────────────────────────────────────────────

PAPERS_DIR = os.path.join(os.path.dirname(__file__), "papers")
SEED_DATA_PATH = os.path.join(os.path.dirname(__file__), "seed_corpus.json")

QUERY_PROMPTS = {
    "qa": """You are an academic research assistant. Using the research context below, give a clear and direct answer to the question. Synthesize information across the papers — never say "the context does not state" or hedge. Always name the paper(s) you draw from. Keep the answer to 2-3 focused paragraphs.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""",

    "summarize": """You are an academic research assistant. Produce a structured summary of the relevant paper(s) from the context below. Use exactly this structure:

Main Contribution: [one clear sentence]
Methodology: [key methods and techniques used]
Key Findings: [main results or contributions]
Significance: [why this paper matters to the field]

Always reference the paper title(s) explicitly.

CONTEXT:
{context}

PAPER / TOPIC TO SUMMARIZE: {question}

STRUCTURED SUMMARY:""",

    "compare": """You are an academic research assistant. Compare and contrast the papers or approaches from the context that are relevant to the query. Structure your response as:

Similarities: [shared ideas, goals, or methods across the papers]
Key Differences: [what makes each approach distinct]
Synthesis: [what we learn by comparing them — which is better suited for what]

Reference specific paper titles throughout your response.

CONTEXT:
{context}

COMPARISON QUERY: {question}

COMPARATIVE ANALYSIS:""",

    "path": """You are an academic research assistant and learning guide. Based on the papers and concepts in the context below, generate a structured learning path. Format exactly as:

Prerequisites: [foundational knowledge needed before starting]
Step 1 — Foundation: [first paper or concept to study, with a one-line reason why]
Step 2 — Core Methods: [key methodology paper, with a one-line reason why]
Step 3 — Advanced Topics: [deeper or more advanced work, with a one-line reason why]
Step 4 — Applications: [applied or practical use cases from the corpus]

Reference actual paper titles from the context at each step.

CONTEXT:
{context}

LEARNING TOPIC: {question}

LEARNING PATH:""",
}

FALLBACK_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b"]

MODE_LABELS = {
    "qa":        "Factual Q&A",
    "summarize": "Summarization",
    "compare":   "Comparative Analysis",
    "path":      "Learning Path",
}

MODE_PLACEHOLDERS = {
    "qa":        "Ask anything about the papers in the corpus…",
    "summarize": "Enter a paper name or topic to summarize…",
    "compare":   "Enter two concepts or papers to compare…",
    "path":      "Enter a topic to generate a learning path…",
}

SAMPLE_QUESTIONS = {
    "qa": [
        "What are Naive RAG, Advanced RAG, and Modular RAG?",
        "How does multi-head attention work in the Transformer?",
        "What is masked language modeling in BERT?",
        "What is Corrective Retrieval Augmented Generation (CRAG)?",
    ],
    "summarize": [
        "Summarize the three RAG paradigms from the RAG survey paper",
        "Summarize the key contributions of the BERT paper",
        "Summarize the best practices for building a RAG pipeline",
        "Summarize how corrective RAG improves retrieval quality",
    ],
    "compare": [
        "Compare Naive RAG and Advanced RAG approaches",
        "Compare BERT and the Transformer from Attention Is All You Need",
        "Compare parametric memory and non-parametric memory in RAG",
        "Compare different retrieval methods used in RAG systems",
    ],
    "path": [
        "Give me a learning path to understand RAG from basics to advanced",
        "What papers should I read to understand Transformer-based RAG?",
        "Learning path for knowledge-intensive NLP tasks using RAG",
        "What should I study first to understand retrieval-augmented generation?",
    ],
}

# ── Pipeline functions ─────────────────────────────────────────────────────────

def _extract_display_title(meta_title, lines, filename):
    import re
    arxiv_pat = re.compile(r"^\d{4}\.\d{4,5}", re.IGNORECASE)
    if meta_title and len(meta_title) > 10 and not arxiv_pat.match(meta_title):
        return meta_title.strip()
    for line in lines[:40]:
        l = line.strip()
        if (
            20 < len(l) < 180
            and not arxiv_pat.match(l)
            and not l.lower().startswith("abstract")
            and not l.lower().startswith("introduction")
            and not l.startswith("http")
            and not l.startswith("©")
            and not l[0].isdigit()
            and l[0].isupper()
        ):
            return l
    return filename.replace("_", " ").replace("-", " ").title()


def parse_pdf(filepath):
    doc = fitz.open(filepath)
    meta_title = (doc.metadata.get("title") or "").strip()
    full_text = "".join(page.get_text() for page in doc)
    doc.close()
    lines = [l.strip() for l in full_text.split("\n") if l.strip()]
    full_text_clean = " ".join(lines)
    text_lower = full_text_clean.lower()
    abs_start = text_lower.find("abstract")
    intro_start = text_lower.find("introduction")
    abstract = ""
    if abs_start != -1 and intro_start != -1 and intro_start > abs_start:
        abstract = full_text_clean[abs_start + 8 : intro_start].strip()
    filename = os.path.basename(filepath).replace(".pdf", "")
    display = _extract_display_title(meta_title, lines, filename)
    return {
        "title": filename,
        "display": display,
        "abstract": abstract[:1200] if abstract else full_text_clean[:400],
        "fulltext": full_text_clean[:6000],
    }


def load_corpus_from_pdfs():
    if not os.path.isdir(PAPERS_DIR):
        return []
    corpus = []
    for fname in [f for f in os.listdir(PAPERS_DIR) if f.lower().endswith(".pdf")]:
        try:
            corpus.append(parse_pdf(os.path.join(PAPERS_DIR, fname)))
        except Exception as e:
            print(f"[kg] Failed to parse {fname}: {e}", flush=True)
    return corpus


def load_seed_corpus():
    if os.path.exists(SEED_DATA_PATH):
        with open(SEED_DATA_PATH) as f:
            return json.load(f)
    return []


def build_knowledge_graph(corpus, kw_model):
    G = nx.DiGraph()
    for paper in corpus:
        G.add_node(paper["title"], type="paper",
                   display=paper["display"], abstract=paper["abstract"])
    for paper in corpus:
        text = paper["abstract"] + " " + paper["fulltext"][:2500]
        try:
            keywords = kw_model.extract_keywords(
                text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=10)
        except Exception:
            keywords = []
        for kw, score in keywords:
            if not G.has_node(kw):
                G.add_node(kw, type="concept")
            G.add_edge(paper["title"], kw, relation="DISCUSSES", weight=float(score))
    concept_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "concept"]
    for concept in concept_nodes:
        papers = [n for n in G.predecessors(concept)
                  if G.nodes[n].get("type") == "paper"]
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                if not G.has_edge(papers[i], papers[j]):
                    G.add_edge(papers[i], papers[j], relation="RELATED_TO")
                if not G.has_edge(papers[j], papers[i]):
                    G.add_edge(papers[j], papers[i], relation="RELATED_TO")
    return G


def build_faiss_index(corpus, embedder):
    chunks = []
    for paper in corpus:
        words = (paper["abstract"] + " " + paper["fulltext"]).split()
        for i in range(0, len(words), 60):
            chunk_text = " ".join(words[i : i + 60])
            if chunk_text.strip():
                chunks.append({"text": chunk_text, "paper_title": paper["title"]})
    if not chunks:
        return None, []
    embeddings = embedder.encode(
        [c["text"] for c in chunks], show_progress_bar=False, batch_size=8)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    return index, chunks


def graphrag_retrieve(query, corpus, graph, faiss_index, chunks, embedder,
                      top_k_per_paper=2, candidate_pool=15):
    if faiss_index is None or not chunks:
        return "", [], [], []
    query_vec = embedder.encode([query])
    k = min(candidate_pool, len(chunks))
    _, indices = faiss_index.search(np.array(query_vec).astype("float32"), k)
    candidates = [chunks[i] for i in indices[0]]
    grouped = {}
    for c in candidates:
        grouped.setdefault(c["paper_title"], []).append(c)
    vector_chunks = []
    for title, group in grouped.items():
        vector_chunks.extend(group[:top_k_per_paper])
    paper_titles = list({c["paper_title"] for c in vector_chunks})
    graph_context, all_concepts = [], []
    for title in paper_titles:
        if graph.has_node(title):
            neighbors = list(graph.successors(title))
            concepts = [n for n in neighbors if graph.nodes[n].get("type") == "concept"]
            all_concepts.extend(concepts)
            related_papers = []
            for concept in concepts[:5]:
                for predecessor in graph.predecessors(concept):
                    if predecessor != title and graph.nodes[predecessor].get("type") == "paper":
                        related_papers.append(predecessor)
            graph_context.append({
                "paper": title,
                "concepts": concepts[:8],
                "related_papers": list(set(related_papers)),
            })
    context_parts = [f"[From paper: {vc['paper_title']}]\n{vc['text']}"
                     for vc in vector_chunks]
    for gc in graph_context:
        context_parts.append(
            f"[Graph context for: {gc['paper']}]\n"
            f"Key concepts: {', '.join(gc['concepts'][:6])}\n"
            f"Related papers: {', '.join(gc['related_papers'])}"
        )
    context = "\n\n---\n\n".join(context_parts)
    display_map = {p["title"]: p["display"] for p in corpus}
    return (context, paper_titles,
            [display_map.get(t, t) for t in paper_titles],
            list(set(all_concepts)))


def bfs_learning_path(seed_concepts, G, corpus):
    visited_papers, visited_concepts, path_papers = set(), set(), []
    queue = deque(seed_concepts[:8])
    while queue and len(path_papers) < 6:
        concept = queue.popleft()
        if concept in visited_concepts:
            continue
        visited_concepts.add(concept)
        for paper in G.predecessors(concept):
            if G.nodes[paper].get("type") == "paper" and paper not in visited_papers:
                visited_papers.add(paper)
                path_papers.append(paper)
                for next_concept in G.successors(paper):
                    if (next_concept not in visited_concepts
                            and G.nodes[next_concept].get("type") == "concept"):
                        queue.append(next_concept)
    display_map = {p["title"]: p["display"] for p in corpus}
    return [display_map.get(p, p) for p in path_papers]


def _call_gemini_once(prompt, api_key, model_name):
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model_name}:generateContent?key={api_key}")
    try:
        resp = requests.post(
            url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=20)
    except requests.RequestException as e:
        return None, str(e)
    if resp.status_code == 429:
        return None, f"429 from {model_name}"
    if not resp.ok:
        return None, f"{model_name} error {resp.status_code}"
    try:
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"], None
    except (KeyError, IndexError):
        return None, f"Unexpected response from {model_name}"


def call_gemini(question, context, api_key, answer_cache, sources_display=None, mode="qa"):
    cache_key = (question.strip().lower(), mode)
    if cache_key in answer_cache:
        return answer_cache[cache_key]
    prompt = QUERY_PROMPTS.get(mode, QUERY_PROMPTS["qa"]).format(
        context=context, question=question)
    for i, model_name in enumerate(FALLBACK_MODELS):
        answer, error = _call_gemini_once(prompt, api_key, model_name)
        if answer:
            answer_cache[cache_key] = answer
            return answer
        if i < len(FALLBACK_MODELS) - 1:
            time.sleep(4)
    sources_note = f" from {', '.join(sources_display)}" if sources_display else ""
    snippet = context[:600].strip()
    return (
        f"The language model is temporarily unavailable (all configured models are "
        f"rate-limited or unreachable), but the retrieval pipeline successfully found "
        f"relevant passages{sources_note}. Here is the raw retrieved context:\n\n"
        f"{snippet}...\n\n(Fallback response — try again shortly.)"
    )

# ── Cached model loading ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI models (one-time, ~30s)…")
def _load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    kw_model = KeyBERT(model=embedder)
    gc.collect()
    return embedder, kw_model


def _build_pipeline(embedder, kw_model):
    corpus = load_corpus_from_pdfs()
    if not corpus:
        corpus = load_seed_corpus()
    graph = build_knowledge_graph(corpus, kw_model)
    gc.collect()
    faiss_index, chunks = build_faiss_index(corpus, embedder)
    gc.collect()
    return corpus, graph, faiss_index, chunks

# ── Graph rendering ────────────────────────────────────────────────────────────

def render_graph(G, corpus, highlighted_papers=None, highlighted_concepts=None):
    net = Network(height="480px", width="100%", bgcolor="#fffdf8",
                  font_color="#1b1a17", directed=True)
    net.set_options("""{
      "physics": {
        "barnesHut": { "gravitationalConstant": -2200, "springLength": 90 }
      },
      "interaction": { "hover": true, "tooltipDelay": 150 },
      "edges": {
        "smooth": { "type": "continuous" },
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.4 } }
      }
    }""")
    display_map = {p["title"]: p["display"] for p in corpus}
    hl_papers   = set(highlighted_papers or [])
    hl_concepts = set(highlighted_concepts or [])
    for node, data in G.nodes(data=True):
        is_paper = data.get("type") == "paper"
        if is_paper:
            label = display_map.get(node, node)
            label = (label[:28] + "…") if len(label) > 28 else label
            color = "#e0a83a" if node in hl_papers else "#9a2b1f"
            size  = 28 if node in hl_papers else 22
            net.add_node(f"paper:{node}", label=label, color=color,
                         size=size, title=label)
        else:
            color = "#e0a83a" if node in hl_concepts else "#3d6b54"
            size  = 14 if node in hl_concepts else 10
            net.add_node(f"concept:{node}", label=node, color=color,
                         size=size, title=node)
    for src, dst, edge_data in G.edges(data=True):
        src_id = ("paper:" if G.nodes[src].get("type") == "paper" else "concept:") + src
        dst_id = ("paper:" if G.nodes[dst].get("type") == "paper" else "concept:") + dst
        relation = edge_data.get("relation", "DISCUSSES")
        net.add_edge(
            src_id, dst_id,
            color="#c0b8a8" if relation == "RELATED_TO" else "#c9c2ab",
            title=relation,
            dashes=(relation == "RELATED_TO"),
            width=1 if relation == "RELATED_TO" else 1.5,
        )
    components.html(net.generate_html(), height=500, scrolling=False)

# ── Page config (must be first Streamlit call) ─────────────────────────────────

st.set_page_config(
    page_title="Knowledge Graph Academic Assistant",
    page_icon="◈",
    layout="wide",
)

# ── Bootstrap models + pipeline ───────────────────────────────────────────────

embedder, kw_model = _load_models()

if "pipeline_ready" not in st.session_state:
    with st.spinner("Building knowledge graph and FAISS index…"):
        corpus, graph, faiss_index, chunks = _build_pipeline(embedder, kw_model)
    st.session_state.corpus       = corpus
    st.session_state.graph        = graph
    st.session_state.faiss_index  = faiss_index
    st.session_state.chunks       = chunks
    st.session_state.pipeline_ready = True
    st.session_state.answers      = []
    st.session_state.answer_cache = {}
    st.session_state.hl_papers    = []
    st.session_state.hl_concepts  = []
    st.session_state.last_uploaded = None

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ◈ Academic Assistant")
    st.caption("GraphRAG · FAISS · NetworkX · KeyBERT · Gemini")
    st.divider()

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="Paste your Gemini API key…",
        help="Stored in this session only — never saved to any server",
    )

    st.divider()

    G      = st.session_state.graph
    corpus = st.session_state.corpus
    paper_count    = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "paper")
    concept_count  = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "concept")
    discusses_cnt  = sum(1 for _, _, d in G.edges(data=True) if d.get("relation") == "DISCUSSES")
    related_cnt    = sum(1 for _, _, d in G.edges(data=True) if d.get("relation") == "RELATED_TO")

    st.markdown("**Corpus Statistics**")
    c1, c2 = st.columns(2)
    c1.metric("Papers",     paper_count)
    c2.metric("Concepts",   concept_count)
    c1.metric("DISCUSSES",  discusses_cnt)
    c2.metric("RELATED_TO", related_cnt)
    st.metric("Graph edges", G.number_of_edges())
    st.caption("Embedding: all-MiniLM-L6-v2  ·  Retrieval: GraphRAG")

    st.divider()

    st.markdown("**Papers in Corpus**")
    for paper in corpus:
        short = paper["display"][:55] + ("…" if len(paper["display"]) > 55 else "")
        with st.expander(short):
            st.caption(paper["abstract"][:500])

    st.divider()

    st.markdown("**Add a Paper**")
    uploaded_file = st.file_uploader(
        "Upload PDF", type=["pdf"], label_visibility="collapsed")
    if (uploaded_file is not None
            and st.session_state.last_uploaded != uploaded_file.name):
        os.makedirs(PAPERS_DIR, exist_ok=True)
        with open(os.path.join(PAPERS_DIR, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner(f"Processing {uploaded_file.name}…"):
            new_corpus, new_graph, new_index, new_chunks = _build_pipeline(embedder, kw_model)
        st.session_state.corpus       = new_corpus
        st.session_state.graph        = new_graph
        st.session_state.faiss_index  = new_index
        st.session_state.chunks       = new_chunks
        st.session_state.last_uploaded = uploaded_file.name
        st.success(f"Added! Corpus now has {len(new_corpus)} papers.")
        st.rerun()

# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='margin-bottom:0'>Knowledge Graph <em>Academic Assistant</em></h1>",
    unsafe_allow_html=True,
)
st.caption(
    "GraphRAG — FAISS · NetworkX · KeyBERT · Gemini · Python 3.11  ·  "
    "Factual Q&A · Summarization · Comparative Analysis · Learning Path"
)
st.divider()

col_qa, col_graph = st.columns([3, 2], gap="large")

# ── Q&A column ─────────────────────────────────────────────────────────────────

with col_qa:
    mode = st.radio(
        "Query mode",
        options=list(MODE_LABELS.keys()),
        format_func=lambda x: MODE_LABELS[x],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.caption("Sample questions:")
    sq_cols = st.columns(2)
    for i, q in enumerate(SAMPLE_QUESTIONS[mode]):
        if sq_cols[i % 2].button(q, key=f"sq_{mode}_{i}", use_container_width=True):
            st.session_state["qa_input"] = q

    # Clear the box from a previous submit BEFORE the widget is instantiated —
    # Streamlit forbids mutating a widget's key after the widget is created.
    if st.session_state.get("_clear_qa_input"):
        st.session_state["qa_input"] = ""
        st.session_state["_clear_qa_input"] = False

    if "qa_input" not in st.session_state:
        st.session_state["qa_input"] = ""

    question = st.text_area(
        "Your question",
        key="qa_input",
        placeholder=MODE_PLACEHOLDERS[mode],
        height=80,
        label_visibility="collapsed",
    )

    ask_disabled = not api_key
    if st.button("Ask →", disabled=ask_disabled, type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner(f"Retrieving via FAISS + graph traversal ({MODE_LABELS[mode]})…"):
                context, source_keys, sources_display, concepts = graphrag_retrieve(
                    question,
                    st.session_state.corpus,
                    st.session_state.graph,
                    st.session_state.faiss_index,
                    st.session_state.chunks,
                    embedder,
                )
            if not context:
                st.error("Retrieval returned no context — corpus may be empty.")
            else:
                with st.spinner("Generating answer with Gemini…"):
                    answer = call_gemini(
                        question, context, api_key,
                        st.session_state.answer_cache,
                        sources_display=sources_display,
                        mode=mode,
                    )
                learning_path = []
                if mode == "path" and concepts:
                    learning_path = bfs_learning_path(
                        concepts, st.session_state.graph, st.session_state.corpus)
                st.session_state.answers.append({
                    "question":        question,
                    "answer":          answer,
                    "mode":            mode,
                    "sources_display": sources_display,
                    "source_keys":     source_keys,
                    "concepts":        concepts,
                    "learning_path":   learning_path,
                    "degraded":        "fallback response" in answer.lower(),
                })
                st.session_state.hl_papers   = source_keys
                st.session_state.hl_concepts = concepts
                st.session_state["_clear_qa_input"] = True
                st.rerun()

    if not api_key:
        st.info("Enter your Gemini API key in the sidebar to start querying.")

    # Answer history (newest first)
    for ans in reversed(st.session_state.get("answers", [])):
        st.divider()
        st.markdown(f"**{MODE_LABELS[ans['mode']]}** — *{ans['question']}*")
        if ans.get("degraded"):
            st.warning("All Gemini models were rate-limited — showing retrieved context.")
        st.markdown(ans["answer"])
        if ans.get("learning_path"):
            step_labels = ["Foundation", "Core Methods", "Advanced Topics",
                           "Applications", "Further Reading", "Beyond"]
            with st.expander("Graph-Generated Reading Order (BFS over knowledge graph)"):
                for i, title in enumerate(ans["learning_path"]):
                    label = step_labels[i] if i < len(step_labels) else "Reading"
                    st.markdown(f"**Step {i + 1} — {label}:** {title}")
        if ans.get("sources_display"):
            st.caption("Sources: " + "  ·  ".join(f"📄 {s}" for s in ans["sources_display"]))

# ── Graph column ───────────────────────────────────────────────────────────────

with col_graph:
    st.markdown("**Knowledge Graph**")
    render_graph(
        st.session_state.graph,
        st.session_state.corpus,
        highlighted_papers=st.session_state.get("hl_papers", []),
        highlighted_concepts=st.session_state.get("hl_concepts", []),
    )
    if st.session_state.get("hl_papers"):
        display_map = {p["title"]: p["display"] for p in st.session_state.corpus}
        retrieved = ", ".join(display_map.get(t, t) for t in st.session_state.hl_papers)
        st.caption(f"Last query retrieved: {retrieved}. Amber nodes were used to answer.")
    else:
        st.caption(
            "Ask a question — nodes retrieved by FAISS + graph traversal highlight in amber.")

    st.markdown("""
**Node types** &nbsp; 🔴 Paper &nbsp;·&nbsp; 🟢 Concept &nbsp;·&nbsp; 🟡 Retrieved for query

**Edge types** &nbsp; — DISCUSSES &nbsp;·&nbsp; ╌ RELATED_TO
""")

# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
G  = st.session_state.graph
pn = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "paper")
cn = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "concept")
st.caption(
    f"Knowledge Graph-Based Academic Assistant — Manal K · 25ETCS126008 · "
    f"Dept. of CS&E · 2026–2027 · {pn} papers · {cn} concepts · "
    f"Streamlit · FAISS · NetworkX · KeyBERT · Sentence-Transformers · Gemini · pyvis"
)
