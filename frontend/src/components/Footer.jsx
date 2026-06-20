export default function Footer({ stats }) {
  const papers = stats?.paper_nodes ?? "—";
  const concepts = stats?.concept_nodes ?? "—";
  return (
    <footer className="footer">
      <span>
        Knowledge Graph-Based Academic Assistant — Manal K · 25ETCS126008 ·
        Dept. of CS&amp;E · 2026–2027 ·{" "}
        <span className="footer-stat">{papers}</span> papers ·{" "}
        <span className="footer-stat">{concepts}</span> concepts
      </span>
      <span>
        Flask · React · FAISS · NetworkX · KeyBERT · Sentence-Transformers ·
        Gemini · vis-network
      </span>
    </footer>
  );
}
