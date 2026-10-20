/* LLM Output Arbitration - front end (vanilla JS, no build step) */
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const VERDICTS = {
  ACCEPT: { label: "Accept", tone: "ok" },
  ACCEPT_WITH_WARNINGS: { label: "Accept with warnings", tone: "warn" },
  REVISE: { label: "Revise", tone: "bad" },
  REJECT: { label: "Reject", tone: "crit" },
};
const SEV_TONE = { low: "none", medium: "warn", high: "bad", critical: "crit" };
const SEV_RANK = { low: 2, medium: 3, high: 4, critical: 5 };
const RANK_CLASS = { 1: "s-ok", 2: "s-note", 3: "s-warn", 4: "s-bad", 5: "s-crit" };
const CRITICS = [
  ["accuracy", "Accuracy", "Is it true?"],
  ["grounding", "Grounding", "Backed by documents?"],
  ["consistency", "Consistency", "Agrees with itself?"],
  ["completeness", "Completeness", "Covers the question?"],
];
const scoreTone = (s) => (s >= 85 ? "ok" : s >= 70 ? "warn" : s >= 50 ? "bad" : "crit");
const pretty = (s) => String(s || "").replace(/_/g, " ");

let lastResult = null;

/* ------------------------------------------------------------------ api */
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch (_) { /* not json */ }
  if (!res.ok) {
    let msg = body && body.detail ? body.detail : `Request failed (${res.status})`;
    if (Array.isArray(msg)) msg = msg.map((e) => `${(e.loc || []).slice(1).join(".")}: ${e.msg}`).join("; ");
    throw new Error(msg);
  }
  return body;
}

/* ------------------------------------------------------------------ status + examples + kb */
async function loadHealth() {
  try {
    const h = await api("/api/health");
    const model = h.llm_mode === "offline" ? "Offline demo (heuristic critics)" : `${h.llm_mode} / ${h.model}`;
    $("status").innerHTML =
      `<div><dt>Critic model</dt><dd>${esc(model)}</dd></div>` +
      `<div><dt>Embeddings</dt><dd>${esc(h.embeddings)}</dd></div>` +
      `<div><dt>Knowledge base</dt><dd>${esc(h.kb_chunks)} passages</dd></div>`;
    const notes = [h.llm_warning, h.embedding_note].filter(Boolean);
    $("banner").hidden = notes.length === 0;
    $("banner").textContent = notes.join(" ");
  } catch (e) {
    $("status").innerHTML = `<div><dt>Status</dt><dd>Server not reachable</dd></div>`;
  }
}

async function loadExamples() {
  const examples = await api("/api/examples");
  const box = $("examples");
  box.innerHTML = "";
  examples.forEach((ex) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.textContent = ex.title;
    b.setAttribute("aria-pressed", "false");
    b.addEventListener("click", () => {
      $("question").value = ex.question;
      $("answer").value = ex.answer;
      box.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", "false"));
      b.setAttribute("aria-pressed", "true");
    });
    box.appendChild(b);
  });
}

async function loadDocs() {
  const d = await api("/api/documents");
  $("kb-summary").textContent = d.sources.length
    ? `${d.sources.length} documents, ${d.chunks} searchable passages.`
    : "Empty. Upload files, paste text, or re-index the sample documents.";
  $("kb-list").innerHTML = d.sources
    .map((s) => `<li><span>${esc(s.source)}</span><span><span class="count">${s.chunks} passages</span>
      <button type="button" data-del="${esc(s.source)}" aria-label="Remove ${esc(s.source)}">Remove</button></span></li>`)
    .join("");
  loadHealth();
}

function kbMsg(text, isError) {
  const el = $("kb-msg");
  el.textContent = text;
  el.style.color = isError ? "var(--crit)" : "";
}

$("kb-list").addEventListener("click", async (e) => {
  const name = e.target.getAttribute && e.target.getAttribute("data-del");
  if (!name) return;
  try {
    await api(`/api/documents/${encodeURIComponent(name)}`, { method: "DELETE" });
    kbMsg(`Removed ${name}.`);
    loadDocs();
  } catch (err) { kbMsg(err.message, true); }
});

$("upload").addEventListener("change", async (e) => {
  const files = [...e.target.files];
  if (!files.length) return;
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  kbMsg("Indexing… the first upload can take a moment.");
  try {
    const r = await api("/api/documents", { method: "POST", body: fd });
    kbMsg(`Indexed ${Object.keys(r.indexed).join(", ")}.`);
    loadDocs();
  } catch (err) { kbMsg(err.message, true); }
  e.target.value = "";
});

$("reindex").addEventListener("click", async () => {
  kbMsg("Re-indexing sample documents…");
  try { await api("/api/reindex", { method: "POST" }); kbMsg("Sample documents re-indexed."); loadDocs(); }
  catch (err) { kbMsg(err.message, true); }
});

$("doc-add").addEventListener("click", async () => {
  const name = $("doc-name").value.trim(), text = $("doc-text").value.trim();
  if (!name || text.length < 10) return kbMsg("Enter a name and at least a sentence of text.", true);
  try {
    await api("/api/documents/text", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, text }) });
    $("doc-name").value = ""; $("doc-text").value = "";
    kbMsg(`Added ${name}.`);
    loadDocs();
  } catch (err) { kbMsg(err.message, true); }
});

/* ------------------------------------------------------------------ run */
$("run").addEventListener("click", async () => {
  const question = $("question").value.trim(), answer = $("answer").value.trim();
  const err = $("error");
  err.hidden = true;
  if (question.length < 3 || answer.length < 3) {
    err.textContent = "Enter both the question and the answer to review.";
    err.hidden = false;
    return;
  }
  const critics = [...document.querySelectorAll("input[name=critic]:checked")].map((c) => c.value);
  if (!critics.length) { err.textContent = "Enable at least one critic in Run settings."; err.hidden = false; return; }
  const options = { critics };
  if ($("samples").value !== "") options.selfcheck_samples = Number($("samples").value);
  if ($("topk").value !== "") options.top_k = Number($("topk").value);
  const payload = { question, answer, options };
  if ($("context").value.trim()) payload.extra_context = $("context").value;

  $("run").disabled = true;
  $("run").textContent = "Deliberating…";
  $("empty").hidden = true; $("result").hidden = true; $("loading").hidden = false;
  try {
    lastResult = await api("/api/evaluate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    render(lastResult, answer);
  } catch (e) {
    err.textContent = e.message;
    err.hidden = false;
    $("empty").hidden = false;
  } finally {
    $("loading").hidden = true;
    $("run").disabled = false;
    $("run").textContent = "Run arbitration";
  }
});

/* ------------------------------------------------------------------ rendering */
const STOP = new Set("a an the and or but of to in on at by for with as is are was were be been it its this that these those there their they he she we you i from into than then so not no can will just".split(" "));
const toks = (t) => new Set((String(t).toLowerCase().match(/[a-z0-9]+/g) || []).filter((w) => w.length > 1 && !STOP.has(w)).map((w) => (w.length > 3 && w.endsWith("s") ? w.slice(0, -1) : w)));
const overlap = (a, b) => { let n = 0; a.forEach((x) => { if (b.has(x)) n++; }); return a.size ? n / a.size : 0; };

function annotate(answer, r) {
  const parts = answer.split(/(?<=[.!?])(\s+)/);
  const sents = [];
  for (let i = 0; i < parts.length; i += 2) sents.push({ text: parts[i], gap: parts[i + 1] || "", rank: 0, notes: [] });
  const sTok = sents.map((s) => toks(s.text));

  const claimRank = {};
  const claimNotes = {};
  r.claims.forEach((c) => { claimRank[c.id] = 0; claimNotes[c.id] = []; });
  r.critic_reports.forEach((rep) => {
    const rows = (rep.details && (rep.details.claims || rep.details.sample_agreement)) || [];
    rows.forEach((row) => {
      const ok = row.verdict === "supported" || row.assessment === "correct" || row.agreement === "consistent";
      if (ok && claimRank[row.claim_id] === 0) claimRank[row.claim_id] = 1;
    });
  });
  r.issues.forEach((i) => {
    if (i.claim_id == null) return;
    claimRank[i.claim_id] = Math.max(claimRank[i.claim_id] || 0, SEV_RANK[i.severity]);
    (claimNotes[i.claim_id] = claimNotes[i.claim_id] || []).push(`${i.severity}: ${i.description}`);
  });

  r.claims.forEach((c) => {
    const ct = toks(c.text);
    let best = -1, bestOv = 0;
    sTok.forEach((st, idx) => { const ov = overlap(ct, st); if (ov > bestOv) { bestOv = ov; best = idx; } });
    if (best >= 0 && bestOv >= 0.5) {
      sents[best].rank = Math.max(sents[best].rank, claimRank[c.id] || 0);
      sents[best].notes.push(...(claimNotes[c.id] || []));
    }
  });

  const cons = r.critic_reports.find((x) => x.critic === "consistency");
  ((cons && cons.details && cons.details.contradictions) || []).forEach((c) => {
    [c.statement_a, c.statement_b].forEach((st) => {
      const t = toks(st);
      sTok.forEach((sk, idx) => {
        if (overlap(t, sk) >= 0.7) {
          sents[idx].rank = Math.max(sents[idx].rank, SEV_RANK[c.severity] || 3);
          sents[idx].notes.push(`contradiction: ${c.explanation || "conflicts with another statement"}`);
        }
      });
    });
  });

  return sents.map((s) => {
    const cls = RANK_CLASS[s.rank];
    const title = s.notes.length ? ` title="${esc(s.notes.join("\n"))}"` : "";
    return (cls ? `<span class="s ${cls}"${title}>${esc(s.text)}</span>` : esc(s.text)) + esc(s.gap);
  }).join("");
}

function verdictBlock(r) {
  const v = VERDICTS[r.verdict] || { label: r.verdict, tone: "none" };
  return `<div class="verdict tone-${v.tone}">
    <div class="verdict-bar"></div>
    <div>
      <p class="verdict-word">${esc(v.label)}</p>
      <p class="verdict-why">${esc(r.verdict_explanation)}</p>
      <div class="gauges">
        <div class="gauge tone-${scoreTone(r.overall_score)}">
          <div class="gauge-head"><span>Overall score</span><strong>${r.overall_score.toFixed(0)} / 100</strong></div>
          <div class="meter"><i data-w="${r.overall_score}"></i></div>
        </div>
        <div class="gauge tone-${r.confidence >= 0.75 ? "ok" : r.confidence >= 0.5 ? "warn" : "bad"}">
          <div class="gauge-head"><span>Confidence in this verdict</span><strong>${Math.round(r.confidence * 100)}%</strong></div>
          <div class="meter"><i data-w="${r.confidence * 100}"></i></div>
        </div>
      </div>
    </div>
  </div>`;
}

function panelBlock(r) {
  const rows = CRITICS.map(([key, name, sub]) => {
    const s = r.scores[key];
    if (!s) return "";
    let mid;
    if (s.status === "ok") {
      mid = `<div class="score">${s.score.toFixed(0)}<span>confidence ${s.confidence.toFixed(2)}, weight ${(s.weight * 100).toFixed(0)}%</span></div>
             <div class="meter tone-${scoreTone(s.score)}"><i data-w="${s.score}"></i></div>`;
    } else {
      const label = { skipped: "Skipped", failed: "Failed to run", not_evaluable: "Could not evaluate" }[s.status];
      mid = `<div class="score">${label}</div>`;
    }
    return `<div class="panel-row"><div class="name">${name}<small>${sub}</small></div><div>${mid}</div><div class="note">${esc(s.summary)}</div></div>`;
  }).join("");
  return `<section class="block"><h3>The panel</h3>
    <p class="sub">Each critic worked alone and never saw the others' findings.</p>${rows}</section>`;
}

function annotatedBlock(r, answer) {
  return `<section class="block"><h3>Annotated answer</h3>
    <p class="sub">Sentences are marked by the most serious issue found on the claims they contain. Hover a marked sentence for the reason.</p>
    <div class="script">${annotate(answer, r)}</div>
    <div class="legend">
      <span><span class="s s-ok">Verified</span></span><span><span class="s s-note">Minor</span></span>
      <span><span class="s s-warn">Medium</span></span><span><span class="s s-bad">High</span></span><span><span class="s s-crit">Critical</span></span>
      <span>Unmarked text had nothing to check.</span>
    </div></section>`;
}

function issuesBlock(r) {
  const body = r.issues.length ? r.issues.map((i) => `
    <div class="issue tone-${SEV_TONE[i.severity]}">
      <div class="edge"></div>
      <div>
        <div class="issue-head"><span class="sev">${esc(i.severity)}</span><span>${esc(pretty(i.category))}</span><span>found by ${esc(i.critic)}</span><span>${esc(i.id)}</span></div>
        <p class="issue-desc">${esc(i.description)}</p>
        ${i.claim ? `<blockquote>${esc(i.claim)}</blockquote>` : ""}
        ${i.suggestion ? `<p class="fix"><b>Fix:</b> ${esc(i.suggestion)}</p>` : ""}
        ${i.corroborated_by.length ? `<p class="corr">Also flagged by ${esc(i.corroborated_by.join(", "))}, so its severity was raised.</p>` : ""}
      </div>
    </div>`).join("") : `<p class="none-found">No issues found.</p>`;
  return `<section class="block"><h3>Issues (${r.issues.length})</h3>${body}</section>`;
}

function recsBlock(r) {
  if (!r.recommendations.length) return "";
  return `<section class="block"><h3>What to do next</h3><ol class="recs">${r.recommendations.map((x) =>
    `<li>${esc(x.action)}<span class="why">${esc(x.rationale)}${x.addresses.length ? ` (${esc(x.addresses.join(", "))})` : ""}</span></li>`).join("")}</ol></section>`;
}

const TAG_TONE = { supported: "ok", correct: "ok", consistent: "ok", partially_supported: "none", unaddressed: "none", uncertain: "none",
  unsupported: "warn", inconsistent: "warn", contradicted: "bad", incorrect: "bad", no_evidence: "none" };
const tag = (v) => v ? `<span class="tag tone-${TAG_TONE[v] || "none"}">${esc(pretty(v))}</span>` : `<span class="hint">not checked</span>`;

function detailsBlock(r) {
  const rep = (n) => r.critic_reports.find((x) => x.critic === n);
  const g = ((rep("grounding") || {}).details || {}).claims || [];
  const a = ((rep("accuracy") || {}).details || {}).claims || [];
  const c = ((rep("consistency") || {}).details || {}).sample_agreement || [];
  const rows = r.claims.map((cl) => {
    const gv = g.find((x) => x.claim_id === cl.id), av = a.find((x) => x.claim_id === cl.id), cv = c.find((x) => x.claim_id === cl.id);
    return `<tr><td>C${cl.id}</td><td>${esc(cl.text)}</td><td>${tag(gv && gv.verdict)}</td><td>${tag(av && av.assessment)}</td><td>${tag(cv && cv.agreement)}</td></tr>`;
  }).join("");

  const usedBy = {};
  g.forEach((x) => (x.evidence_ids || []).forEach((id) => (usedBy[id] = usedBy[id] || []).push("C" + x.claim_id)));
  const ev = r.evidence.map((e) => `<div class="ev"><div class="ev-head">${esc(e.id)}: ${esc(e.source)} (${e.origin === "user_context" ? "pasted context" : "knowledge base"}), relevance ${e.score.toFixed(2)}${usedBy[e.id] ? `, cited for ${esc(usedBy[e.id].join(", "))}` : ""}</div><p>${esc(e.text)}</p></div>`).join("")
    || `<p class="hint">Nothing relevant was retrieved. Add documents to the knowledge base for grounding checks.</p>`;

  const m = r.meta;
  const weights = Object.entries(m.weights).map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`).join(", ");
  const samples = ((rep("consistency") || {}).details || {}).samples || [];
  return `<section class="block"><h3>Working papers</h3>
    <details class="opt" open><summary>Claim-by-claim checks</summary>
      <div class="tablewrap"><table class="grid"><thead><tr><th>#</th><th>Claim</th><th>Grounding</th><th>Accuracy</th><th>Sampled answers</th></tr></thead><tbody>${rows || `<tr><td colspan="5">No claims were extracted.</td></tr>`}</tbody></table></div>
    </details>
    <details class="opt"><summary>Retrieved evidence (${r.evidence.length})</summary>${ev}</details>
    ${samples.length ? `<details class="opt"><summary>Independent samples used for the self-check (${samples.length})</summary>${samples.map((s) => `<div class="ev"><p>${esc(s)}</p></div>`).join("")}</details>` : ""}
    <details class="opt"><summary>How the verdict was reached</summary>
      <dl class="meta-list">
        <dt>Rules applied</dt><dd>${m.rules_triggered.map(esc).join("<br>")}</dd>
        <dt>Critic weights</dt><dd>${esc(weights)}</dd>
        <dt>Critic disagreement</dt><dd>${m.disagreement} points (standard deviation of critic scores)</dd>
        <dt>Model</dt><dd>${esc(m.llm_mode)} / ${esc(m.model)}</dd>
        <dt>Embeddings</dt><dd>${esc(m.embeddings)}</dd>
        <dt>Time taken</dt><dd>${(m.elapsed_ms / 1000).toFixed(1)} s</dd>
      </dl>
      ${m.warnings.length ? `<p class="warnings">${m.warnings.map(esc).join("<br>")}</p>` : ""}
    </details>
    <details class="opt"><summary>Raw JSON response</summary>
      <button type="button" class="quiet" id="copy-json">Copy JSON</button>
      <pre class="json">${esc(JSON.stringify(r, null, 2))}</pre>
    </details></section>`;
}

function render(r, answer) {
  const box = $("result");
  box.innerHTML = verdictBlock(r) + panelBlock(r) + annotatedBlock(r, answer) + issuesBlock(r) + recsBlock(r) + detailsBlock(r);
  box.hidden = false;
  requestAnimationFrame(() => requestAnimationFrame(() => box.querySelectorAll(".meter > i").forEach((el) => { el.style.width = el.dataset.w + "%"; })));
  const copy = $("copy-json");
  if (copy) copy.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(JSON.stringify(lastResult, null, 2)); copy.textContent = "Copied"; }
    catch (_) { copy.textContent = "Copy failed"; }
    setTimeout(() => (copy.textContent = "Copy JSON"), 1500);
  });
  box.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
}

/* ------------------------------------------------------------------ init */
loadHealth();
loadExamples();
loadDocs().catch(() => { $("kb-summary").textContent = "Could not load documents."; });
