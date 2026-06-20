import { useState } from "react";
import AnswerCard from "./AnswerCard.jsx";
import {
  MODES,
  MODE_TAB_LABELS,
  MODE_PLACEHOLDERS,
  SAMPLE_QUESTIONS,
} from "../constants.js";

export default function QueryPanel({
  mode,
  onModeChange,
  answers,
  keyConnected,
  onAsk,
}) {
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (questionText) => {
    const question = (questionText ?? input).trim();
    if (!question || busy) return;
    if (!keyConnected) {
      alert("Paste your Gemini API key in the bar at the top and click Connect first.");
      return;
    }
    setBusy(true);
    try {
      await onAsk(question, mode);
    } catch {
      // error is rendered inside the answer card; nothing more to do here
    } finally {
      setBusy(false);
      setInput("");
    }
  };

  const askLabel = busy ? "Asking…" : "Ask →";
  const askDisabled = busy;

  return (
    <div className="col col-center">
      <div className="qa-shell">
        <div className="eyebrow">Query Mode</div>

        <div className="mode-tabs">
          {MODES.map((m) => (
            <button
              key={m}
              className={`mode-tab${m === mode ? " active" : ""}`}
              onClick={() => onModeChange(m)}
            >
              {MODE_TAB_LABELS[m]}
            </button>
          ))}
        </div>

        <div className="query-form">
          <textarea
            value={input}
            placeholder={MODE_PLACEHOLDERS[mode]}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
          />
          <button onClick={() => submit()} disabled={askDisabled}>
            {askLabel}
          </button>
        </div>

        <div className="sample-qs">
          {SAMPLE_QUESTIONS[mode].map((q) => (
            <button key={q} onClick={() => submit(q)}>
              {q}
            </button>
          ))}
        </div>

        <div>
          {answers.length === 0 ? (
            <div className="empty-state">
              <div className="glyph">◈</div>
              <div className="et">
                Connect your Gemini key above, then select a query mode and ask a
                question — the GraphRAG pipeline will retrieve from FAISS + the
                knowledge graph and answer with full source attribution.
              </div>
            </div>
          ) : (
            answers.map((a) => <AnswerCard key={a.id} answer={a} />)
          )}
        </div>
      </div>
    </div>
  );
}
