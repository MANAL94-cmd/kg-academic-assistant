# Knowledge Graph-Based Academic Assistant

A real GraphRAG pipeline: PDF parsing → KeyBERT concept extraction → NetworkX
knowledge graph → Sentence-Transformer embeddings → FAISS vector search →
graph-augmented retrieval → Gemini-generated, cited answers. Served through a
Flask backend with a connected frontend.

## Project structure

```
kg_app/
├── app.py                 # Flask backend — the real pipeline
├── requirements.txt
├── render.yaml            # Render.com deployment config
├── seed_corpus.json       # Pre-extracted text for the 3 demo papers
├── papers/                # Drop PDFs here to extend the corpus
└── templates/
    └── index.html          # Frontend — talks to the backend via /api/*
```

## Run it locally

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser.

The first request after starting will take 20–40 seconds while the
sentence-transformer model loads and the graph + FAISS index build. After
that, queries are fast.

On the page: paste your Gemini API key into the bar at the top and click
**Connect** (the key stays in your browser tab only — it is sent straight to
Google's API per-request, and is never written to disk or logged on the
server).

## Add your own papers

Drop PDF files into `papers/` before starting the app, or use the **Add a
Paper** upload control in the UI once it's running — either way the app
re-parses and rebuilds the knowledge graph and FAISS index automatically.

If `papers/` is empty, the app falls back to `seed_corpus.json`, which holds
pre-extracted text for the three demo papers (Attention Is All You Need,
BERT, and Retrieval-Augmented Generation) so the demo works even with zero
PDFs present.

## How a question is answered (the actual pipeline)

1. The query is embedded with `all-MiniLM-L6-v2` and matched against a FAISS
   index of all paper chunks.
2. Results are grouped per-paper so a single dominant paper can't crowd out
   others — this matters for comparison-style questions.
3. For every retrieved paper, the NetworkX graph is traversed to pull its
   KeyBERT-extracted concept nodes and any other papers sharing those
   concepts.
4. The merged vector + graph context is sent to Gemini
   (`gemini-flash-latest`) with a prompt instructing it to answer only from
   that context and cite the papers by name.
5. The frontend highlights exactly which paper and concept nodes were used,
   in amber, on the live graph.

## Deploying to Render (free tier)

1. Push this folder to a GitHub repository.
2. On Render: **New → Web Service** → connect the repo.
3. Render will detect `render.yaml` automatically, or set manually:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1`
4. Deploy. The free tier cold-starts after inactivity, so the first request
   after idling can take 30–60 seconds while it reloads the models — this is
   normal and worth mentioning if demoing live.

## Notes for your report / viva

- The retrieval is genuine GraphRAG: FAISS handles the vector-similarity
  half, NetworkX handles the graph-traversal half, and both are merged into
  one context block before the LLM call.
- Relation extraction (e.g. a Rebel-style model) was intentionally scoped
  out to keep the project deliverable within the available timeline; the
  graph currently encodes paper→concept (`DISCUSSES`) edges derived from
  KeyBERT, which is sufficient to demonstrate multi-hop retrieval across
  papers sharing a concept.
- API keys are deliberately never stored server-side — this was a
  conscious security decision after an earlier key was accidentally
  exposed during development and had to be rotated.
