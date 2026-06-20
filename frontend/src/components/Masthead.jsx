// Top "newspaper masthead" header. `backendLabel` reflects the live status of
// the Flask backend (connecting / initializing / live / unreachable).
export default function Masthead({ backendLabel }) {
  return (
    <header className="masthead">
      <div className="masthead-top">
        <span>Vol. I — No. 1 · Dept. of Computer Science &amp; Engineering</span>
        <span>{backendLabel}</span>
      </div>
      <h1 className="masthead-title">
        Knowledge Graph <em>Academic Assistant</em>
      </h1>
      <div className="masthead-sub">
        GraphRAG — FAISS · NetworkX · KeyBERT · Gemini · Python 3.11
      </div>
      <div className="masthead-pills">
        <span className="masthead-pill">Factual Q&amp;A</span>
        <span className="masthead-pill">Summarization</span>
        <span className="masthead-pill">Comparative Analysis</span>
        <span className="masthead-pill">Learning Path</span>
      </div>
    </header>
  );
}
