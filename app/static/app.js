// The page: ask, show the answer with its citations, the sources, what the law was before, and take feedback.
const $ = (id) => document.getElementById(id);
const EXAMPLES = [
  "What is the punishment for issuing a cheque that bounces?",
  "Talaq ka notice kis ko aur kab dena hota hai?",
  "کرایہ دار کو مکان سے کب نکالا جا سکتا ہے؟",
  "Can a minor enter into a valid contract?",
];
let current = null;

function escape(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// The answer's light Markdown: headings, paragraphs, lists, **bold**, quoted provisions ("> "), the [n] citations as
// buttons, and drafted documents (a ```document block) laid out as a document, with Copy and Print.
function render(answer) {
  const inline = (line) => escape(line)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*?)\*(?!\*)/g, "$1<em>$2</em>")
    .replace(/\[(\d{1,2})\]/g, (m, n) => `<button type="button" class="cite" data-n="${n}" aria-label="Source ${n}">${n}</button>`);
  const out = [];
  let list = null, quote = null, doc = null;
  const closeList = () => { if (list) { out.push(`</${list.kind}>`); list = null; } };
  const closeQuote = () => { if (quote) { out.push(`<blockquote class="provision" dir="auto">${quote.map(inline).join("<br>")}</blockquote>`); quote = null; } };
  for (const raw of answer.split(/\n/)) {
    if (doc !== null) {
      if (/^\s*```\s*$/.test(raw)) {
        const text = doc.join("\n").replace(/^\n+|\n+$/g, "");
        out.push(`<div class="document"><div class="doc-tools"><span>Draft document</span>
          <button type="button" class="link" data-copy>Copy</button><button type="button" class="link" data-print>Print</button></div>
          <pre class="doc" dir="auto">${escape(text)}</pre></div>`);
        doc = null;
      } else doc.push(raw);
      continue;
    }
    const line = raw.trim();
    if (/^```/.test(line)) { closeList(); closeQuote(); doc = []; continue; }
    if (/^>/.test(line)) { closeList(); (quote ||= []).push(line.replace(/^>\s?/, "")); continue; }
    closeQuote();
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) { closeList(); out.push(`<h3 class="answer-h">${inline(heading[2])}</h3>`); continue; }
    const bullet = line.match(/^(?:[-*•]|\d+[.)])\s+(.*)$/);
    if (bullet) {
      const kind = /^\d/.test(line) ? "ol" : "ul";
      if (!list || list.kind !== kind) { closeList(); list = { kind }; out.push(`<${kind}>`); }
      out.push(`<li>${inline(bullet[1])}</li>`);
      continue;
    }
    closeList();
    if (line) out.push(`<p>${inline(line)}</p>`);
  }
  closeList(); closeQuote();
  if (doc !== null) out.push(`<div class="document"><pre class="doc" dir="auto">${escape(doc.join("\n"))}</pre></div>`);   // an unclosed block
  return out.join("");
}

function printDocument(text) {
  const page = window.open("", "_blank");
  page.document.write(`<!doctype html><title>Draft</title><style>body{font:12pt/1.6 "Times New Roman",serif;margin:2.5cm;white-space:pre-wrap}</style>${escape(text)}`);
  page.document.close();
  page.print();
}

function badges(item) {
  const out = [`<span class="badge ${escape(item.tier)}">${escape(item.status)}</span>`];
  if (item.jurisdiction) out.push(`<span class="badge">${escape(item.jurisdiction)}</span>`);
  if (item.last_amended) out.push(`<span class="badge">last amended ${escape(item.last_amended)}</span>`);
  for (const place of item.exceptions || []) out.push(`<span class="badge historical">repealed for ${escape(place)}</span>`);
  return `<div class="badges">${out.join("")}</div>`;
}

function sourceCard(item) {
  return `<li class="source card${item.cited ? "" : " uncited"}" id="source-${item.n}">
    <h3>${item.n}. ${escape(item.title)}</h3>
    <div class="where">${escape(item.label)}</div>
    ${badges(item)}
    <details><summary>Read the provision</summary><div class="text" dir="ltr">${escape(item.text)}</div></details>
    <button type="button" class="link" data-law="${escape(item.work_id)}">Open the whole law</button>
  </li>`;
}

function historyItem(item) {
  const replaced = item.replaced_by ? ` — ${item.how === "enacted as an Act" ? "replaced by" : "repealed by"} ${escape(item.replaced_by)}` : "";
  return `<li class="source card">
    <h3>${escape(item.title)}</h3>
    <div class="where">${escape(item.label)}${replaced}</div>
    ${badges(item)}
    <details><summary>Read what it said</summary><div class="text" dir="ltr">${escape(item.text)}</div></details>
    <button type="button" class="link" data-law="${escape(item.work_id)}">Open the whole law</button>
  </li>`;
}

async function ask(event) {
  event?.preventDefault();
  const question = $("question").value.trim();
  if (question.length < 3) return;
  $("go").disabled = true;
  $("loading").hidden = false;
  $("error").hidden = true;
  $("result").hidden = true;
  try {
    const response = await fetch("/api/ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, province: $("province").value || null }),
    });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
    current = await response.json();
    show(current);
  } catch (error) {
    $("error").textContent = `Something went wrong: ${error.message}`;
    $("error").hidden = false;
  } finally {
    $("go").disabled = false;
    $("loading").hidden = true;
  }
}

function show(data) {
  $("answer").innerHTML = render(data.answer);
  const parts = [`${data.sources.length} sections used`];
  if (data.issues?.length) parts.push(`${data.issues.length} issues searched separately`);
  if (data.gap_sources) parts.push(`${data.gap_sources} more found for points first left uncovered`);
  if (data.province_inferred) parts.push(`answered for ${data.province_inferred} (named in your question; choose a province to change it)`);
  $("meta").textContent = [...parts, `${data.seconds}s`].join(" · ");
  $("sources").innerHTML = data.sources.map(sourceCard).join("");
  $("history").hidden = !data.history.length;
  $("history-list").innerHTML = data.history.map(historyItem).join("");
  for (const b of document.querySelectorAll("[data-rating]")) b.classList.remove("chosen");
  $("report-form").hidden = true;
  $("thanks").hidden = true;
  $("result").hidden = false;
}

async function sendFeedback(rating, kind, comment) {
  if (!current) return;
  await fetch("/api/feedback", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question_id: current.id, rating, kind, comment }),
  });
  $("thanks").hidden = false;
}

async function openLaw(workId) {
  const response = await fetch(`/api/law/${encodeURIComponent(workId)}`);
  if (!response.ok) return;
  const law = await response.json();
  $("law-title").textContent = law.title;
  const replaces = law.replaces.length ? ` It replaced: ${law.replaces.map((p) => p.title).join("; ")}.` : "";
  $("law-status").textContent = `${law.status}. ${law.evidence || ""}${replaces}`;
  $("law-body").innerHTML = law.provisions.map((p) => `<h4>${escape(p.label)}</h4><div class="text" dir="ltr">${escape(p.text)}</div>`).join("");
  $("law").showModal();
}

document.addEventListener("click", (event) => {
  const cite = event.target.closest(".cite");
  if (cite) {
    const card = $(`source-${cite.dataset.n}`);
    if (card) {
      card.querySelector("details").open = true;
      card.scrollIntoView({ behavior: "smooth", block: "center" });
      card.classList.add("flash");
      setTimeout(() => card.classList.remove("flash"), 1500);
    }
  }
  const copy = event.target.closest("[data-copy]");
  if (copy) {
    navigator.clipboard.writeText(copy.closest(".document").querySelector(".doc").textContent);
    copy.textContent = "Copied";
    setTimeout(() => (copy.textContent = "Copy"), 1500);
  }
  const print = event.target.closest("[data-print]");
  if (print) printDocument(print.closest(".document").querySelector(".doc").textContent);
  const lawButton = event.target.closest("[data-law]");
  if (lawButton) openLaw(lawButton.dataset.law);
  const rate = event.target.closest("[data-rating]");
  if (rate) {
    for (const b of document.querySelectorAll("[data-rating]")) b.classList.toggle("chosen", b === rate);
    sendFeedback(Number(rate.dataset.rating), null, null);
  }
});

$("ask").addEventListener("submit", ask);
$("question").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) ask(e); });
$("report").addEventListener("click", () => { $("report-form").hidden = false; $("report-text").focus(); });
$("report-form").addEventListener("submit", (e) => {
  e.preventDefault();
  sendFeedback(-1, "wrong", $("report-text").value.trim());
  $("report-form").hidden = true;
});
$("law-close").addEventListener("click", () => $("law").close());
$("examples").innerHTML = EXAMPLES.map((q) => `<button type="button" dir="auto">${escape(q)}</button>`).join("");
$("examples").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (b) { $("question").value = b.textContent; ask(); }
});
