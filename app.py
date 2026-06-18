"""
Knowledge Graph-Based Academic Assistant — Backend
Implements the real pipeline: PDF parsing -> KeyBERT concepts ->
NetworkX knowledge graph -> Sentence-Transformer embeddings ->
FAISS vector search -> GraphRAG context merge -> Gemini answer.

Supports 4 query modes: factual Q&A, paper summarization,
comparative analysis, and learning path generation (BFS over graph).

Run locally:
    pip install -r requirements.txt
    python app.py

Deploy on Render:
    Build command: pip install -r requirements.txt
    Start command: gunicorn app:app
"""

import os
import gc
import json
import traceback
from collections import deque
import time

# Reduce PyTorch/tokenizer thread overhead on small instances
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import fitz  # PyMuPDF
import numpy as np
import networkx as nx
import faiss
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

app = Flask(__name__)
CORS(app)

# Simple in-memory cache — keyed by (question_lower, mode)
_answer_cache: dict = {}

PAPERS_DIR = os.path.join(os.path.dirname(__file__), "papers")
SEED_DATA_PATH = os.path.join(os.path.dirname(__file__), "seed_corpus.json")

# ---------------------------------------------------------------------------
# Global state — built once at startup, rebuildable via /api/rebuild
# ---------------------------------------------------------------------------
STATE = {
    "corpus": [],
    "chunks": [],
    "embedder": None,
    "kw_model": None,
    "faiss_index": None,
    "graph": None,
    "ready": False,
}

# ---------------------------------------------------------------------------
# Query prompt templates — one per mode
# ---------------------------------------------------------------------------
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

LEARNING PATH:"""
}


def log(msg):
    print(f"[kg-backend] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Step 1 — PDF parsing
# ---------------------------------------------------------------------------
def _extract_display_title(meta_title, lines, filename):
    """Return the best human-readable title for a paper."""
    import re
    arxiv_pat = re.compile(r'^\d{4}\.\d{4,5}', re.IGNORECASE)

    # 1. PDF metadata title if it looks real
    if meta_title and len(meta_title) > 10 and not arxiv_pat.match(meta_title):
        return meta_title.strip()

    # 2. First long line from the text that looks like a title
    for line in lines[:40]:
        l = line.strip()
        if (20 < len(l) < 180
                and not arxiv_pat.match(l)
                and not l.lower().startswith("abstract")
                and not l.lower().startswith("introduction")
                and not l.startswith("http")
                and not l.startswith("©")
                and not l[0].isdigit()
                and l[0].isupper()):
            return l

    # 3. Fall back to filename
    return filename.replace("_", " ").replace("-", " ").title()


def parse_pdf(filepath):
    doc = fitz.open(filepath)
    meta_title = (doc.metadata.get("title") or "").strip()

    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    lines = [l.strip() for l in full_text.split("\n") if l.strip()]
    full_text_clean = " ".join(lines)

    text_lower = full_text_clean.lower()
    abs_start = text_lower.find("abstract")
    intro_start = text_lower.find("introduction")

    abstract = ""
    if abs_start != -1 and intro_start != -1 and intro_start > abs_start:
        abstract = full_text_clean[abs_start + 8: intro_start].strip()

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
    pdfs = [f for f in os.listdir(PAPERS_DIR) if f.lower().endswith(".pdf")]
    corpus = []
    for fname in pdfs:
        try:
            parsed = parse_pdf(os.path.join(PAPERS_DIR, fname))
            corpus.append(parsed)
            log(f"Parsed PDF: {fname}")
        except Exception as e:
            log(f"Failed to parse {fname}: {e}")
    return corpus


def load_seed_corpus():
    if os.path.exists(SEED_DATA_PATH):
        with open(SEED_DATA_PATH) as f:
            return json.load(f)
    return []


# ---------------------------------------------------------------------------
# Step 2 — Knowledge graph construction (NetworkX)
# Nodes: Paper, Concept
# Edges: DISCUSSES (paper→concept), RELATED_TO (paper↔paper via shared concepts)
# ---------------------------------------------------------------------------
def build_knowledge_graph(corpus, kw_model):
    G = nx.DiGraph()

    for paper in corpus:
        G.add_node(paper["title"], type="paper",
                   display=paper["display"], abstract=paper["abstract"])

    for paper in corpus:
        text = paper["abstract"] + " " + paper["fulltext"][:2500]
        try:
            keywords = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                top_n=10,
            )
        except Exception as e:
            log(f"KeyBERT failed for {paper['title']}: {e}")
            keywords = []

        for kw, score in keywords:
            if not G.has_node(kw):
                G.add_node(kw, type="concept")
            G.add_edge(paper["title"], kw, relation="DISCUSSES", weight=float(score))

    # Add RELATED_TO edges between papers that share concepts
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

    log(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ---------------------------------------------------------------------------
# Step 3 — FAISS vector index over chunks
# ---------------------------------------------------------------------------
def build_faiss_index(corpus, embedder):
    chunks = []
    for paper in corpus:
        text = paper["abstract"] + " " + paper["fulltext"]
        words = text.split()
        for i in range(0, len(words), 60):
            chunk_text = " ".join(words[i:i + 60])
            if chunk_text.strip():
                chunks.append({"text": chunk_text, "paper_title": paper["title"]})

    if not chunks:
        return None, []

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False, batch_size=8)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    log(f"FAISS index built: {index.ntotal} vectors")
    return index, chunks


# ---------------------------------------------------------------------------
# Step 4 — GraphRAG retrieval
# ---------------------------------------------------------------------------
def graphrag_retrieve(query, top_k_per_paper=2, candidate_pool=15):
    embedder = STATE["embedder"]
    index = STATE["faiss_index"]
    chunks = STATE["chunks"]
    G = STATE["graph"]
    corpus = STATE["corpus"]

    if index is None or not chunks:
        return "", [], [], []

    query_vec = embedder.encode([query])
    k = min(candidate_pool, len(chunks))
    distances, indices = index.search(np.array(query_vec).astype("float32"), k)
    candidates = [chunks[i] for i in indices[0]]

    grouped = {}
    for c in candidates:
        grouped.setdefault(c["paper_title"], []).append(c)

    vector_chunks = []
    for title, group in grouped.items():
        vector_chunks.extend(group[:top_k_per_paper])

    paper_titles = list({c["paper_title"] for c in vector_chunks})

    graph_context = []
    all_concepts = []
    for title in paper_titles:
        if G.has_node(title):
            neighbors = list(G.successors(title))
            concepts = [n for n in neighbors if G.nodes[n].get("type") == "concept"]
            all_concepts.extend(concepts)

            related_papers = []
            for concept in concepts[:5]:
                for predecessor in G.predecessors(concept):
                    if predecessor != title and G.nodes[predecessor].get("type") == "paper":
                        related_papers.append(predecessor)

            graph_context.append({
                "paper": title,
                "concepts": concepts[:8],
                "related_papers": list(set(related_papers)),
            })

    context_parts = []
    for vc in vector_chunks:
        context_parts.append(f"[From paper: {vc['paper_title']}]\n{vc['text']}")
    for gc in graph_context:
        context_parts.append(
            f"[Graph context for: {gc['paper']}]\n"
            f"Key concepts: {', '.join(gc['concepts'][:6])}\n"
            f"Related papers: {', '.join(gc['related_papers'])}"
        )

    context = "\n\n---\n\n".join(context_parts)
    display_map = {p["title"]: p["display"] for p in corpus}
    sources_display = [display_map.get(t, t) for t in paper_titles]

    return context, paper_titles, sources_display, list(set(all_concepts))


# ---------------------------------------------------------------------------
# Step 4b — BFS learning path generation over the knowledge graph
# ---------------------------------------------------------------------------
def bfs_learning_path(seed_concepts, G, corpus):
    """BFS from seed concepts through the graph to suggest a reading order."""
    visited_papers = set()
    visited_concepts = set()
    path_papers = []
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
                    if next_concept not in visited_concepts and G.nodes[next_concept].get("type") == "concept":
                        queue.append(next_concept)

    display_map = {p["title"]: p["display"] for p in corpus}
    return [display_map.get(p, p) for p in path_papers]


# ---------------------------------------------------------------------------
# Step 5 — Gemini call with mode-aware prompts and multi-model fallback
# ---------------------------------------------------------------------------
# gemini-1.5-flash-8b has 1000 RPM on the free tier — almost never rate-limited
FALLBACK_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",   # 1000 RPM free tier — last resort
]


def _call_gemini_once(prompt, api_key, model_name):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        resp = requests.post(url, json=payload, timeout=20)
    except requests.RequestException as e:
        return None, str(e)

    if resp.status_code == 429:
        log(f"{model_name} rate limited — skipping to next model")
        return None, f"429 from {model_name}"

    if not resp.ok:
        log(f"Gemini error {resp.status_code}: {resp.text[:200]}")
        return None, f"{model_name} error {resp.status_code}"

    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"], None
    except (KeyError, IndexError):
        return None, f"Unexpected response from {model_name}"


def call_gemini(question, context, api_key, sources_display=None, mode="qa"):
    cache_key = (question.strip().lower(), mode)
    if cache_key in _answer_cache:
        log(f"Cache hit for '{question[:60]}' (mode:{mode})")
        return _answer_cache[cache_key]

    prompt_template = QUERY_PROMPTS.get(mode, QUERY_PROMPTS["qa"])
    prompt = prompt_template.format(context=context, question=question)

    errors = []
    for i, model_name in enumerate(FALLBACK_MODELS):
        answer, error = _call_gemini_once(prompt, api_key, model_name)
        if answer:
            log(f"Answered using {model_name} (mode: {mode})")
            _answer_cache[cache_key] = answer
            return answer
        errors.append(error)
        log(f"Model {model_name} failed: {error}")
        if i < len(FALLBACK_MODELS) - 1:
            time.sleep(4)  # brief pause so rate-limit window can partially reset

    log("All Gemini models failed — returning context-based fallback answer.")
    sources_note = f" from {', '.join(sources_display)}" if sources_display else ""
    snippet = context[:600].strip()
    return (
        f"The language model is temporarily unavailable (all configured models are "
        f"rate-limited or unreachable), but the retrieval pipeline successfully found "
        f"relevant passages{sources_note}. Here is the raw retrieved context that would "
        f"normally be summarized:\n\n{snippet}...\n\n"
        f"(This is a fallback response — the GraphRAG retrieval step worked correctly; "
        f"only the final LLM call failed. Try again shortly.)"
    )


# ---------------------------------------------------------------------------
# Startup — build everything once
# ---------------------------------------------------------------------------
def initialize_pipeline():
    log("Loading models (this can take 30-60s on first boot)...")
    STATE["embedder"] = SentenceTransformer("all-MiniLM-L6-v2")
    STATE["kw_model"] = KeyBERT(model=STATE["embedder"])
    gc.collect()

    corpus = load_corpus_from_pdfs()
    if not corpus:
        log("No PDFs found in /papers — loading seed corpus instead.")
        corpus = load_seed_corpus()

    STATE["corpus"] = corpus
    gc.collect()

    STATE["graph"] = build_knowledge_graph(corpus, STATE["kw_model"])
    gc.collect()

    STATE["faiss_index"], STATE["chunks"] = build_faiss_index(corpus, STATE["embedder"])
    gc.collect()

    STATE["ready"] = True
    log("Pipeline ready.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/status")
def status():
    if not STATE["ready"]:
        return jsonify({"ready": False})
    G = STATE["graph"]
    papers = [
        {"title": p["title"], "display": p["display"], "abstract": p["abstract"]}
        for p in STATE["corpus"]
    ]
    concept_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "concept")
    discusses_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("relation") == "DISCUSSES")
    related_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("relation") == "RELATED_TO")

    return jsonify({
        "ready": True,
        "papers": papers,
        "stats": {
            "paper_nodes": len(papers),
            "concept_nodes": concept_nodes,
            "edges": G.number_of_edges(),
            "discusses_edges": discusses_edges,
            "related_edges": related_edges,
        },
    })


@app.route("/api/graph")
def graph_data():
    G = STATE["graph"]
    nodes = []
    edges = []
    for node, data in G.nodes(data=True):
        if data.get("type") == "paper":
            display_map = {p["title"]: p["display"] for p in STATE["corpus"]}
            nodes.append({
                "id": "paper:" + node,
                "label": display_map.get(node, node),
                "group": "paper",
                "key": node,
            })
        else:
            nodes.append({"id": "concept:" + node, "label": node, "group": "concept"})

    for src, dst, data in G.edges(data=True):
        src_id = "paper:" + src if G.nodes[src].get("type") == "paper" else "concept:" + src
        dst_id = "paper:" + dst if G.nodes[dst].get("type") == "paper" else "concept:" + dst
        relation = data.get("relation", "DISCUSSES")
        edges.append({"from": src_id, "to": dst_id, "relation": relation})

    return jsonify({"nodes": nodes, "edges": edges})


@app.route("/api/ask", methods=["POST"])
def ask():
    if not STATE["ready"]:
        return jsonify({"error": "Pipeline still initializing — try again in a few seconds."}), 503

    body = request.get_json(force=True)
    question = (body or {}).get("question", "").strip()
    api_key = (body or {}).get("api_key", "").strip()
    mode = (body or {}).get("mode", "qa").strip()

    if mode not in QUERY_PROMPTS:
        mode = "qa"
    if not question:
        return jsonify({"error": "No question provided."}), 400
    if not api_key:
        return jsonify({"error": "No Gemini API key provided."}), 400

    try:
        context, source_keys, source_display, concepts = graphrag_retrieve(question)
        if not context:
            return jsonify({"error": "Retrieval returned no context — corpus may be empty."}), 500

        answer = call_gemini(question, context, api_key,
                             sources_display=source_display, mode=mode)
        degraded = "fallback response" in answer.lower()

        learning_path = []
        if mode == "path" and concepts:
            learning_path = bfs_learning_path(concepts, STATE["graph"], STATE["corpus"])

        return jsonify({
            "answer": answer,
            "sources": source_keys,
            "sources_display": source_display,
            "concepts_used": concepts,
            "degraded": degraded,
            "learning_path": learning_path,
            "mode": mode,
        })
    except Exception as e:
        log(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/rebuild", methods=["POST"])
def rebuild():
    try:
        initialize_pipeline()
        return jsonify({"status": "rebuilt", "paper_count": len(STATE["corpus"])})
    except Exception as e:
        log(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400

    save_path = os.path.join(PAPERS_DIR, f.filename)
    os.makedirs(PAPERS_DIR, exist_ok=True)
    f.save(save_path)
    log(f"Saved upload: {f.filename}")

    initialize_pipeline()
    return jsonify({"status": "added", "paper_count": len(STATE["corpus"])})


initialize_pipeline()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
