// Minimal, self-contained markdown renderer ported from the original
// vanilla-JS frontend. It escapes HTML *first*, so the output is XSS-safe to
// inject via dangerouslySetInnerHTML — the LLM answer never reaches the DOM
// as live markup.

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function inlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, "<code>$1</code>");
}

export function renderMarkdown(text) {
  const lines = String(text || "").split("\n");
  let html = "";
  let inUl = false;
  let inOl = false;

  const closeList = () => {
    if (inUl) {
      html += "</ul>";
      inUl = false;
    }
    if (inOl) {
      html += "</ol>";
      inOl = false;
    }
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (/^### /.test(line)) {
      closeList();
      html += `<h4>${inlineMarkdown(line.slice(4))}</h4>`;
      continue;
    }
    if (/^## /.test(line)) {
      closeList();
      html += `<h3>${inlineMarkdown(line.slice(3))}</h3>`;
      continue;
    }
    const bullet = line.match(/^[-*+] (.+)/);
    if (bullet) {
      if (!inUl) {
        closeList();
        html += "<ul>";
        inUl = true;
      }
      html += `<li>${inlineMarkdown(bullet[1])}</li>`;
      continue;
    }
    const num = line.match(/^\d+\. (.+)/);
    if (num) {
      if (!inOl) {
        closeList();
        html += "<ol>";
        inOl = true;
      }
      html += `<li>${inlineMarkdown(num[1])}</li>`;
      continue;
    }
    if (line.trim() === "") {
      closeList();
      continue;
    }
    closeList();
    html += `<p>${inlineMarkdown(line)}</p>`;
  }
  closeList();
  return html;
}
