import { stroke } from "./evidence.js";
import { closePop, layout, pop, treeRuns } from "./workbench.js";
import { $, S, at, editsText, esc, fd, isB, root } from "../state.js";
/* ---------- share card ---------- */
function renderCard() {
  const T = treeRuns(); if (!T) return;
  const pick = T.runs.find((r) => r.n === S.sel && r.split) || T.runs.filter((r) => r.split).sort((a, b) => b.at - a.at)[0];
  const words = (r, from, n) => r.words.slice(from, from + n).map((x) => x.w + x.sp).join("").trim();
  const si = Math.max(0, pick.words.findIndex((x) => x.split)), model = T.e.model.split("/").pop();
  const base = words(T.runs[0], 0, 9), lead = words(pick, 0, si), splitW = pick.words[si]?.w || "", tail = words(pick, si + 1, 9);
  const edits = editsText(pick.rows);
  $("card").innerHTML = `<div class="ct"><span class="wordmark">lobot</span><span class="meta">${esc(model)}</span></div>
    <p class="meta" style="margin:14px 0 0;font-family:var(--body);font-size:14px">“${esc(T.e.prompt)}”</p>
    <div class="cf" id="cf"><svg id="cfl" aria-hidden="true"></svg><div style="font:400 34px var(--body)" id="cr">${esc(T.shared)}</div>
      <div class="cb"><div id="cb0"><span class="meta" style="display:block">no edit</span>${esc(ell(base))}</div><div id="cb1"><span class="meta" style="display:block">${pick.rows.some(isB) ? `${esc(pick.rows[0].trait)} pushed` : `${pick.rows.length} neuron${pick.rows.length === 1 ? "" : "s"} pushed`}</span>${lead ? esc(lead) + " " : ""}<span class="tok">${esc(splitW)}</span> ${esc(ell(tail))}</div></div></div>
    ${pick.proof ? `<p class="meta" style="margin:0 0 10px;color:var(--ink);font-family:var(--body);font-size:14px">${esc(proofShort(pick.proof))}</p>` : ""}
    <div style="display:flex;gap:16px;align-items:end;justify-content:space-between"><div style="min-width:0"><span class="mono" style="font-size:15px;color:var(--ink)">${esc(edits)}</span><br><span class="meta">same prompt, same seed, split at ${at(pick.split)}${pick.research ? " · research mode, guard off" : ""}</span></div>
      ${pick.rows.length === 1 ? `<span style="font:800 56px/.85 var(--display);font-stretch:75%;letter-spacing:-0.02em">${fd(pick.rows[0].dose)}</span>` : ""}</div>`;
  const post = pick.post || postText(T, pick, model, base, lead, splitW, tail);
  $("post").textContent = post; $("postCount").textContent = `${[...post].length}/280`;
  $("recipe").textContent = pick.recipe || "";
  requestAnimationFrame(() => {
    const box = $("cf").getBoundingClientRect(), rb = $("cr").getBoundingClientRect(), cs = getComputedStyle(root), x0 = rb.right - box.left + 6, y0 = rb.top - box.top + rb.height / 2;
    $("cfl").innerHTML = [["cb0", cs.getPropertyValue("--ink"), 2], ["cb1", cs.getPropertyValue("--cut"), 4]].map(([id, c, w]) => { const b = $(id).getBoundingClientRect(), x1 = b.left - box.left - 6, y1 = b.top - box.top + b.height / 2; return `<path d="M${x0} ${y0} C ${(x0 + x1) / 2} ${y0}, ${(x0 + x1) / 2} ${y1}, ${x1} ${y1}" fill="none" stroke="${c}" stroke-width="${w}"/>`; }).join("");
  });
}
const ell = (t) => (/[.!?]$/.test(t) ? t : t + "…");
const proofShort = (p) => `The reader predicted “${p.word}”; pushing ${p.id} made it ${p.ratio >= 10 ? Math.round(p.ratio) : p.ratio.toFixed(1)}× more likely (measured on 8 neutral prompts).`;
// Post text for runs without a server-built post (behavior runs): only recorded text and measured numbers, under 280 characters.
function postText(T, pick, model, base, lead, splitW, tail) {
  const head = `${model}, same prompt, same seed. I pushed ${editsText(pick.rows)} inside the model, and its answer forked at “${splitW.replace(/[.,;:!?]+$/, "")}”.`;
  const proof = pick.proof ? ` ${proofShort(pick.proof)}` : "";
  for (let n = 60; n >= 10; n -= 5) {
    const cut = (t) => ([...t].length > n ? [...t].slice(0, n).join("").trimEnd() + "…" : /[.!?]$/.test(t) ? t : t + "…");
    const t = `${head}${proof}\n\nno edit: “${cut(T.shared + " " + base)}”\nedited: “${cut([T.shared, lead, splitW, tail].filter(Boolean).join(" "))}”`;
    if ([...t].length <= 280) return t;
  }
  return [...`${head}${proof}`].slice(0, 280).join("");
}
function setView(v) {
  S.view = v; closePop(); $("vWork").hidden = v !== "work"; $("vShare").hidden = v !== "share"; $("vPage").hidden = v !== "page";
  if (v !== "page") S.page = null;
  if (v === "page") return;
  if (v === "share") { renderCard(); $("shareH").setAttribute("tabindex", "-1"); $("shareH").focus(); } else { requestAnimationFrame(layout); $("shareBtn").focus(); }
}
$("shareBtn").addEventListener("click", () => setView("share"));
$("backBtn").addEventListener("click", () => setView("work"));
document.querySelectorAll("[data-copy]").forEach((b) => b.addEventListener("click", async () => {
  const text = $(b.dataset.copy).textContent, label = b.textContent;
  try { await navigator.clipboard.writeText(text); b.textContent = "Copied"; }
  catch { const r = document.createRange(); r.selectNodeContents($(b.dataset.copy)); const s = getSelection(); s.removeAllRanges(); s.addRange(r); b.textContent = "Selected: press ⌘C"; }
  setTimeout(() => (b.textContent = label), 1600);
}));

export { renderCard, ell, proofShort, postText, setView };
