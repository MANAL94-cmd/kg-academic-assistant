import { useState } from "react";

function PaperCard({ paper }) {
  const [open, setOpen] = useState(false);
  return (
    <div className={`paper-card${open ? " open" : ""}`}>
      <div className="ptitle">{paper.display}</div>
      <div className="pabs">{paper.abstract}</div>
      <span className="ptoggle" onClick={() => setOpen((o) => !o)}>
        {open ? "− hide abstract" : "+ read abstract"}
      </span>
    </div>
  );
}

const UPLOAD_STEPS = [
  "Uploading PDF",
  "Parsing text & extracting title",
  "Running KeyBERT concept extraction",
  "Rebuilding knowledge graph",
  "Rebuilding FAISS index",
];

export default function CorpusPanel({ status, papers, stats, onUpload }) {
  // uploadStep: -1 = idle, 0..N = step index in progress, N = done
  const [uploadName, setUploadName] = useState(null);
  const [uploadStep, setUploadStep] = useState(-1);

  const handleFile = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setUploadName(file.name);
    setUploadStep(0);

    // The backend does parsing + rebuild in one synchronous call, so we advance
    // the visible steps optimistically to give the user a sense of progress.
    const tick = (i) => setUploadStep(i);
    tick(0);
    await new Promise((r) => setTimeout(r, 300));
    tick(2);
    try {
      await onUpload(file);
      tick(UPLOAD_STEPS.length); // all done
    } catch (err) {
      alert("Upload failed: " + err.message);
    } finally {
      setUploadName(null);
      setUploadStep(-1);
      e.target.value = "";
    }
  };

  const uploading = uploadStep >= 0;

  return (
    <div className="col col-left">
      <div className="eyebrow">Corpus</div>

      {status !== "live" ? (
        <div className="loading-state">
          <span className="blink">▋</span> loading papers from backend…
        </div>
      ) : (
        papers.map((p, i) => <PaperCard key={i} paper={p} />)
      )}

      {uploading && (
        <div>
          <div className="loading-state" style={{ marginBottom: 8 }}>
            <span className="blink">▋</span> Processing <strong>{uploadName}</strong>
          </div>
          <div className="upload-steps">
            {UPLOAD_STEPS.map((label, i) => {
              const cls =
                i < uploadStep ? "done" : i === uploadStep ? "active" : "";
              return (
                <div key={i} className={`upload-step ${cls}`}>
                  <span className="step-dot" />
                  {label}
                </div>
              );
            })}
          </div>
        </div>
      )}

      <div className="eyebrow" style={{ marginTop: 24 }}>
        Add a Paper
      </div>
      <label className="upload-zone">
        <input type="file" accept=".pdf" onChange={handleFile} disabled={uploading} />
        Drop a PDF here, or click to choose
        <br />
        <span style={{ fontSize: 10, opacity: 0.7 }}>
          (parsed server-side · concepts extracted · added to graph)
        </span>
      </label>

      <div className="eyebrow" style={{ marginTop: 24 }}>
        Corpus Statistics
      </div>
      <StatRow label="Paper nodes" value={stats?.paper_nodes} />
      <StatRow label="Concept nodes" value={stats?.concept_nodes} cls="green" />
      <StatRow label="DISCUSSES edges" value={stats?.discusses_edges} />
      <StatRow label="RELATED_TO edges" value={stats?.related_edges} cls="blue" />
      <StatRow label="Total graph edges" value={stats?.edges} />
      <StatRow label="Retrieval mode" value="GraphRAG" />
      <StatRow label="Embedding model" value="all-MiniLM-L6-v2" />
    </div>
  );
}

function StatRow({ label, value, cls = "" }) {
  return (
    <div className="stat-row">
      <span>{label}</span>
      <span className={`v ${cls}`}>{value ?? "—"}</span>
    </div>
  );
}
