# Frontend — React (Vite)

The React UI for the Knowledge Graph Academic Assistant. It talks to the Flask
backend in [`../backend`](../backend) over the `/api/*` endpoints.

## Develop

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

The Vite dev server proxies `/api/*` to the Flask backend on
`http://localhost:5000`, so start the backend too:

```bash
# from the repo root, in another terminal
pip install -r backend/requirements.txt
python backend/app.py
```

## Build for production

```bash
npm run build        # emits frontend/dist/
```

The Flask app serves `frontend/dist/` directly, so once built you can run just
the backend and open <http://localhost:5000>.

## Structure

```
src/
├── main.jsx            # React entry point
├── App.jsx             # top-level state + layout (3-column shell)
├── api.js              # fetch wrappers for the backend REST API
├── constants.js        # query modes, sample questions, UI copy
├── markdown.js         # tiny XSS-safe markdown renderer for answers
├── styles.css          # the newspaper-aesthetic stylesheet
└── components/
    ├── Masthead.jsx     # header
    ├── KeyBar.jsx       # Gemini API key input
    ├── CorpusPanel.jsx  # left column: papers, PDF upload, stats
    ├── QueryPanel.jsx   # center column: mode tabs, query box, answer feed
    ├── AnswerCard.jsx   # one answer (markdown + sources + learning path)
    ├── GraphView.jsx    # right column: live vis-network graph + modal
    └── Footer.jsx
```
