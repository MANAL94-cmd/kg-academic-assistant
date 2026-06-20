import { useCallback, useEffect, useRef, useState } from "react";
import Masthead from "./components/Masthead.jsx";
import KeyBar from "./components/KeyBar.jsx";
import CorpusPanel from "./components/CorpusPanel.jsx";
import QueryPanel from "./components/QueryPanel.jsx";
import GraphView from "./components/GraphView.jsx";
import Footer from "./components/Footer.jsx";
import { MODE_LABELS } from "./constants.js";
import { getStatus, getGraph, ask, uploadPaper } from "./api.js";

const BACKEND_LABELS = {
  connecting: "Backend: connecting…",
  initializing: "Backend: initializing…",
  live: "Backend: live",
  unreachable: "Backend: unreachable",
};

const DEFAULT_NOTE =
  "Live graph from the backend. Ask a question — nodes retrieved by FAISS + " +
  "graph traversal will highlight in amber.";

export default function App() {
  const [geminiKey, setGeminiKey] = useState(null);
  const [serverKey, setServerKey] = useState(false);
  const [backendState, setBackendState] = useState("connecting");
  const [papers, setPapers] = useState([]);
  const [stats, setStats] = useState(null);
  const [graphData, setGraphData] = useState({ nodes: [], edges: [] });
  const [mode, setMode] = useState("qa");
  const [answers, setAnswers] = useState([]);
  const [highlight, setHighlight] = useState({ paperKeys: [], conceptLabels: [] });
  const [graphNote, setGraphNote] = useState(DEFAULT_NOTE);

  const answerIdRef = useRef(0);
  const pollRef = useRef(null);

  // Load corpus status (polling until the pipeline finishes warming up), then
  // pull the full graph once it's ready.
  const refreshStatusAndGraph = useCallback(async () => {
    try {
      const data = await getStatus();
      if (!data.ready) {
        setBackendState("initializing");
        pollRef.current = setTimeout(refreshStatusAndGraph, 2000);
        return;
      }
      setBackendState("live");
      setServerKey(!!data.has_server_key);
      setPapers(data.papers || []);
      setStats(data.stats || null);
      const g = await getGraph();
      setGraphData({ nodes: g.nodes || [], edges: g.edges || [] });
    } catch {
      setBackendState("unreachable");
      pollRef.current = setTimeout(refreshStatusAndGraph, 3000);
    }
  }, []);

  useEffect(() => {
    refreshStatusAndGraph();
    return () => clearTimeout(pollRef.current);
  }, [refreshStatusAndGraph]);

  const runQuery = useCallback(
    async (question, queryMode) => {
      const id = ++answerIdRef.current;
      setAnswers((prev) => [
        { id, question, mode: queryMode, status: "loading" },
        ...prev,
      ]);
      try {
        const data = await ask({ question, apiKey: geminiKey, mode: queryMode });
        setHighlight({
          paperKeys: data.sources || [],
          conceptLabels: data.concepts_used || [],
        });
        const label = MODE_LABELS[data.mode || queryMode];
        setGraphNote(
          `Last query (${label}) retrieved: ${(data.sources_display || []).join(
            ", "
          )}. Amber nodes were used to answer.`
        );
        setAnswers((prev) =>
          prev.map((a) =>
            a.id === id ? { ...a, status: "done", ...data } : a
          )
        );
      } catch (err) {
        setAnswers((prev) =>
          prev.map((a) =>
            a.id === id
              ? { ...a, status: "error", errorMessage: err.message }
              : a
          )
        );
        throw err;
      }
    },
    [geminiKey]
  );

  const handleUpload = useCallback(
    async (file) => {
      await uploadPaper(file);
      // Re-pull corpus + graph so the new paper shows up everywhere.
      await refreshStatusAndGraph();
    },
    [refreshStatusAndGraph]
  );

  return (
    <>
      <Masthead backendLabel={BACKEND_LABELS[backendState]} />
      <KeyBar
        connected={!!geminiKey || serverKey}
        serverKey={serverKey && !geminiKey}
        onConnect={setGeminiKey}
      />

      <div className="shell">
        <CorpusPanel
          status={backendState}
          papers={papers}
          stats={stats}
          onUpload={handleUpload}
        />
        <QueryPanel
          mode={mode}
          onModeChange={setMode}
          answers={answers}
          keyConnected={!!geminiKey || serverKey}
          onAsk={runQuery}
        />
        <GraphView graphData={graphData} highlight={highlight} note={graphNote} />
      </div>

      <Footer stats={stats} />
    </>
  );
}
