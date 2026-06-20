import { useState } from "react";

// The Gemini API key bar. The key lives only in React state for this tab — it
// is sent straight to the backend per-request and is never persisted anywhere.
export default function KeyBar({ connected, onConnect }) {
  const [value, setValue] = useState("");

  const connect = () => {
    const trimmed = value.trim();
    if (trimmed) onConnect(trimmed);
  };

  return (
    <div className="keybar">
      <span>
        <span className={`dot${connected ? " live" : ""}`} />
        <strong>GEMINI KEY</strong>
      </span>
      <input
        type="password"
        value={value}
        placeholder="Paste your Gemini API key (stored in this tab only)"
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && connect()}
      />
      <button onClick={connect}>Connect</button>
      <span className="status">
        {connected
          ? "connected — key kept in memory only for this session"
          : "not connected — never saved to the server"}
      </span>
    </div>
  );
}
