---
title: KG Academic Assistant
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Knowledge Graph-Based Academic Assistant

A real GraphRAG pipeline: PDF parsing → KeyBERT concept extraction → NetworkX
knowledge graph → Sentence-Transformer embeddings → FAISS vector search →
graph-augmented retrieval → Gemini-generated, cited answers.

The app is split into a **React frontend** and a **Flask backend**. The backend
exposes the pipeline over a small REST API; the frontend is a single-page React
app that consumes it.

## Project structure

```
kg_app/
├── backend/                  # Flask API + the real GraphRAG pipeline
│   ├── app.py                #   routes: /api/status, /api/graph, /api/ask, /api/upload, /api/rebuild
│   ├── requirements.txt      #   backend Python dependencies
│   ├── seed_corpus.json      #   pre-extracted text for the demo papers
│   └── papers/               #   drop PDFs here to extend the corpus (git-ignored)
│
├── frontend/                 # React (Vite) single-page app — the primary UI
│   ├── package.json
│   ├── vite.config.js        #   proxies /api → backend during dev
│   ├── index.html
│   └── src/
│       ├── App.jsx           #   top-level state + 3-column layout
│       ├── api.js            #   fetch wrappers for the backend
│       ├── constants.js      #   query modes, sample questions
│       ├── markdown.js       #   XSS-safe markdown renderer
│       ├── styles.css
│       └── components/       #   Masthead, KeyBar, CorpusPanel, QueryPanel,
│                             #   AnswerCard, GraphView, Footer
│
├── legacy/                   # Old self-contained Streamlit app (still runnable)
│   ├── streamlit_app.py
│   ├── requirements.txt
│   ├── templates/index.html  #   the original vanilla-JS frontend
│   └── static/
│
├── render.yaml               # Render.com deployment (builds frontend + runs backend)
├── requirements.txt          # re-exports legacy/ for Streamlit Cloud
└── README.md
```

## Run it locally

You need **Python 3.11+** and **Node 18+**.

**1. Start the backend** (terminal 1):

```bash
pip install -r backend/requirements.txt
python backend/app.py            # API on http://localhost:5000
```

The first request takes 20–40 seconds while the sentence-transformer model loads
and the graph + FAISS index build. After that, queries are fast.

**2. Start the frontend** (terminal 2):

```bash
cd frontend
npm install
npm run dev                      # UI on http://localhost:5173
```

The Vite dev server proxies `/api/*` to the backend, so open
**http://localhost:5173**.

### One-service mode (production-style)

Build the frontend once and let Flask serve it — then you only run the backend:

```bash
cd frontend && npm install && npm run build   # emits frontend/dist/
cd .. && python backend/app.py                # open http://localhost:5000
```

## Using it

Paste your Gemini API key into the bar at the top and click **Connect**. The key
stays in your browser tab only — it is sent straight to Google's API per-request
and is never written to disk or logged on the server.

Pick a query mode (Factual Q&A, Summarization, Comparative Analysis, or Learning
Path), ask a question, and the GraphRAG pipeline retrieves from FAISS + the
knowledge graph and answers with source attribution. Nodes used to answer light
up amber in the live graph.

## Add your own papers

Drop PDF files into `backend/papers/` before starting, or use the **Add a Paper**
upload control in the UI — either way the app re-parses and rebuilds the
knowledge graph and FAISS index automatically. If `backend/papers/` is empty, the
app falls back to `backend/seed_corpus.json` (Attention Is All You Need, BERT, and
a RAG survey) so the demo works with zero PDFs present.

## How a question is answered (the actual pipeline)

1. The query is embedded with `all-MiniLM-L6-v2` and matched against a FAISS
   index of all paper chunks.
2. Results are grouped per-paper so a single dominant paper can't crowd out
   others — this matters for comparison-style questions.
3. For every retrieved paper, the NetworkX graph is traversed to pull its
   KeyBERT-extracted concept nodes and any other papers sharing those concepts.
4. The merged vector + graph context is sent to Gemini with a mode-specific
   prompt instructing it to answer only from that context and cite papers by name.
5. The frontend highlights exactly which paper and concept nodes were used.

## Deploying to Render (free tier)

`render.yaml` builds the React frontend and serves it from the Flask backend as a
single web service:

- **Build:** `pip install -r backend/requirements.txt && cd frontend && npm install && npm run build`
- **Start:** `gunicorn --chdir backend app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1`

The free tier cold-starts after inactivity, so the first request after idling can
take 30–60 seconds while it reloads the models — this is normal.

## Legacy Streamlit app

The original single-file Streamlit version still lives in `legacy/` and shares
the same corpus (`backend/seed_corpus.json`, `backend/papers/`). Run it with:

```bash
pip install -r legacy/requirements.txt
streamlit run legacy/streamlit_app.py
```

It is kept as a fallback; the React + Flask stack above is the primary app.

## Notes for the report / viva

- The retrieval is genuine GraphRAG: FAISS handles the vector-similarity half,
  NetworkX handles the graph-traversal half, and both are merged into one context
  block before the LLM call.
- Relation extraction (e.g. a Rebel-style model) was intentionally scoped out;
  the graph encodes paper→concept (`DISCUSSES`) edges from KeyBERT plus
  paper↔paper (`RELATED_TO`) edges for shared concepts, which is sufficient to
  demonstrate multi-hop retrieval.
- API keys are deliberately never stored server-side — a conscious security
  decision after an earlier key was accidentally exposed during development.
