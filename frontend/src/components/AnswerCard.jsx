import { renderMarkdown } from "../markdown.js";
import { MODE_LABELS, LEARNING_PATH_LABELS } from "../constants.js";

function LearningPath({ path }) {
  if (!path || path.length === 0) return null;
  return (
    <div className="learning-path-card">
      <div className="lp-header">
        Graph-Generated Reading Order (BFS over knowledge graph)
      </div>
      {path.map((title, i) => (
        <div key={i} className="lp-step">
          <span className="lp-badge">
            Step {i + 1} — {LEARNING_PATH_LABELS[i] || "Reading"}
          </span>
          <span className="lp-text">{title}</span>
        </div>
      ))}
    </div>
  );
}

// One entry in the answer feed. `answer.status` is "loading" | "done" | "error".
export default function AnswerCard({ answer }) {
  const { question, status, mode } = answer;

  return (
    <div className="answer-card">
      <div className="answer-q">
        Question — <span>{question}</span>
      </div>

      {status === "loading" && (
        <div className="loading-state">
          <span className="blink">▋</span> retrieving via FAISS + graph traversal
          ({MODE_LABELS[mode]})…
        </div>
      )}

      {status === "error" && (
        <div className="answer-body error">{answer.errorMessage}</div>
      )}

      {status === "done" && (
        <>
          <div className="answer-mode-badge">
            {MODE_LABELS[answer.mode] || MODE_LABELS[mode]}
          </div>
          {answer.degraded && (
            <div className="degraded-note">
              ⚠ all models were rate-limited — showing retrieved context directly
            </div>
          )}
          <div
            className="answer-body"
            dangerouslySetInnerHTML={{ __html: renderMarkdown(answer.answer) }}
          />
          <LearningPath path={answer.learning_path} />
          <div className="source-tags">
            {(answer.sources_display || []).map((s, i) => (
              <span key={i} className="source-tag">
                {s}
              </span>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
