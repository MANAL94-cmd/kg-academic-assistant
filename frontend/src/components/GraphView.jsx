import { useEffect, useRef, useState } from "react";
import { Network, DataSet } from "vis-network/standalone";

// ── Node / edge styling (ported from the original vis-network frontend) ──────

function paperNodeColor(highlighted) {
  return highlighted
    ? { background: "#e0a83a", border: "#8a661f" }
    : { background: "#9a2b1f", border: "#5e150d" };
}
function conceptNodeColor(highlighted) {
  return highlighted
    ? { background: "#e0a83a", border: "#8a661f" }
    : { background: "#3d6b54", border: "#264636" };
}

function truncate(label, max) {
  return label.length > max ? label.slice(0, max - 2) + "…" : label;
}

function buildNodes(rawNodes, highlight, { labelMax, paperSize, conceptSize, paperFont, conceptFont }) {
  return rawNodes.map((n) => {
    const isPaper = n.group === "paper";
    const highlighted = isPaper
      ? highlight.paperKeys.includes(n.key)
      : highlight.conceptLabels.includes(n.label);
    return {
      id: n.id,
      label: truncate(n.label, labelMax),
      shape: "dot",
      size: highlighted ? (isPaper ? 28 : 14) : isPaper ? paperSize : conceptSize,
      color: isPaper ? paperNodeColor(highlighted) : conceptNodeColor(highlighted),
      font: {
        color: "#1b1a17",
        size: isPaper ? paperFont : conceptFont,
        face: "IBM Plex Mono",
      },
    };
  });
}

function buildEdges(rawEdges) {
  return rawEdges.map((e) => ({
    from: e.from,
    to: e.to,
    title: e.relation,
    color: { color: e.relation === "RELATED_TO" ? "#c0b8a8" : "#c9c2ab" },
    width: e.relation === "RELATED_TO" ? 1 : 1.5,
    dashes: e.relation === "RELATED_TO",
  }));
}

// ── Main component ───────────────────────────────────────────────────────────

export default function GraphView({ graphData, highlight, note }) {
  const containerRef = useRef(null);
  const networkRef = useRef(null);
  const [stabilized, setStabilized] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);

  const hasGraph = graphData.nodes.length > 0;

  // (Re)build the network whenever the underlying graph or highlights change.
  useEffect(() => {
    if (!containerRef.current || !hasGraph) return;

    const nodes = new DataSet(
      buildNodes(graphData.nodes, highlight, {
        labelMax: 28,
        paperSize: 22,
        conceptSize: 10,
        paperFont: 12,
        conceptFont: 10,
      })
    );
    const edges = new DataSet(buildEdges(graphData.edges));

    networkRef.current = new Network(
      containerRef.current,
      { nodes, edges },
      {
        physics: {
          stabilization: true,
          barnesHut: { gravitationalConstant: -2200, springLength: 90 },
        },
        interaction: { hover: true, zoomView: true, tooltipDelay: 150 },
        edges: {
          smooth: { type: "continuous" },
          arrows: { to: { enabled: true, scaleFactor: 0.4 } },
        },
      }
    );
    networkRef.current.once("stabilized", () => setStabilized(true));

    return () => {
      networkRef.current?.destroy();
      networkRef.current = null;
    };
  }, [graphData, highlight, hasGraph]);

  const fit = () =>
    networkRef.current?.fit({
      animation: { duration: 400, easingFunction: "easeInOutQuad" },
    });
  const zoom = (factor) => {
    if (!networkRef.current) return;
    networkRef.current.moveTo({
      scale: networkRef.current.getScale() * factor,
      animation: { duration: 300, easingFunction: "easeInOutQuad" },
    });
  };

  return (
    <div className="col col-right">
      <div className="eyebrow">Knowledge Graph</div>
      <div className="graph-toolbar">
        <button className="graph-btn" onClick={fit}>
          ⊞ Fit
        </button>
        <button className="graph-btn" onClick={() => zoom(1.3)}>
          ＋
        </button>
        <button className="graph-btn" onClick={() => zoom(0.7)}>
          －
        </button>
        <button
          className="graph-btn"
          onClick={() => setModalOpen(true)}
          style={{ marginLeft: "auto" }}
          disabled={!hasGraph}
        >
          ⤢ Expand
        </button>
      </div>

      <div className="graph-wrap">
        <div ref={containerRef} className="graph-container" />
        {(!hasGraph || !stabilized) && (
          <div className="graph-empty">
            <span className="ge-glyph">◈</span>
            <span>
              {hasGraph
                ? "Laying out the graph…"
                : "Ask a question to see the graph come alive"}
            </span>
          </div>
        )}
      </div>

      <div className="eyebrow" style={{ marginTop: 16 }}>
        Node Types
      </div>
      <div className="legend">
        <LegendDot color="#9a2b1f" label="Paper" />
        <LegendDot color="#3d6b54" label="Concept" />
        <LegendDot color="#2b5a9a" label="Author" planned />
        <LegendDot color="#6b3d6b" label="Institution" planned />
        <LegendDot color="#9a6b2b" label="Dataset" planned />
        <LegendDot color="#e0a83a" label="Retrieved for query" />
      </div>

      <div className="eyebrow">Relationship Types</div>
      <div className="relation-legend">
        <div className="relation-item">
          <span className="relation-line" /> DISCUSSES
        </div>
        <div className="relation-item">
          <span className="relation-line dashed" /> RELATED_TO
        </div>
        <RelationPlanned label="CITES" />
        <RelationPlanned label="AUTHORED_BY" />
        <RelationPlanned label="INTRODUCES" />
        <RelationPlanned label="USES_DATASET" />
      </div>

      <div className="graph-note">{note}</div>

      {modalOpen && (
        <GraphModal
          graphData={graphData}
          highlight={highlight}
          onClose={() => setModalOpen(false)}
        />
      )}
    </div>
  );
}

function LegendDot({ color, label, planned }) {
  return (
    <div className="legend-item">
      <span className="legend-dot" style={{ background: color }} /> {label}
      {planned && <span className="planned">(planned)</span>}
    </div>
  );
}

function RelationPlanned({ label }) {
  return (
    <div className="relation-item">
      <span className="relation-line" style={{ background: "#b8b0a0" }} /> {label}{" "}
      <span className="planned">(planned)</span>
    </div>
  );
}

// ── Full-screen modal graph ──────────────────────────────────────────────────

function GraphModal({ graphData, highlight, onClose }) {
  const containerRef = useRef(null);
  const networkRef = useRef(null);

  useEffect(() => {
    // Defer creation one tick so the modal is painted and the container has size.
    const id = setTimeout(() => {
      if (!containerRef.current) return;
      const nodes = new DataSet(
        buildNodes(graphData.nodes, highlight, {
          labelMax: 32,
          paperSize: 26,
          conceptSize: 12,
          paperFont: 13,
          conceptFont: 11,
        })
      );
      const edges = new DataSet(buildEdges(graphData.edges));
      networkRef.current = new Network(
        containerRef.current,
        { nodes, edges },
        {
          physics: {
            stabilization: true,
            barnesHut: { gravitationalConstant: -3000, springLength: 130 },
          },
          interaction: { hover: true, zoomView: true, tooltipDelay: 150 },
          edges: {
            smooth: { type: "continuous" },
            arrows: { to: { enabled: true, scaleFactor: 0.5 } },
          },
        }
      );
      networkRef.current.once("stabilized", () =>
        networkRef.current?.fit({
          animation: { duration: 400, easingFunction: "easeInOutQuad" },
        })
      );
    }, 150);

    return () => {
      clearTimeout(id);
      networkRef.current?.destroy();
      networkRef.current = null;
    };
  }, [graphData, highlight]);

  const fit = () =>
    networkRef.current?.fit({
      animation: { duration: 400, easingFunction: "easeInOutQuad" },
    });
  const zoom = (factor) => {
    if (!networkRef.current) return;
    networkRef.current.moveTo({
      scale: networkRef.current.getScale() * factor,
      animation: { duration: 300, easingFunction: "easeInOutQuad" },
    });
  };

  return (
    <div
      className="graph-modal open"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="graph-modal-inner">
        <div className="graph-modal-header">
          <span className="graph-modal-title">Knowledge Graph — Full View</span>
          <div style={{ display: "flex", gap: 6 }}>
            <button className="graph-btn" onClick={fit}>
              ⊞ Fit
            </button>
            <button className="graph-btn" onClick={() => zoom(1.3)}>
              ＋
            </button>
            <button className="graph-btn" onClick={() => zoom(0.7)}>
              －
            </button>
            <button
              className="graph-btn"
              onClick={onClose}
              style={{ borderColor: "var(--accent)", color: "var(--accent)" }}
            >
              ✕ Close
            </button>
          </div>
        </div>
        <div ref={containerRef} className="graph-modal-container" />
      </div>
    </div>
  );
}
