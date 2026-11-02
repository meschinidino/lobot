import { run } from "./screens/run.js";
import { drawN, inspect, layout } from "./screens/workbench.js";
const $ = (id) => document.getElementById(id);
const root = document.documentElement;
const store = {
  get(k, d) { try { const v = localStorage.getItem("lobot." + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("lobot." + k, JSON.stringify(v)); } catch { /* storage blocked or full: the tree then lives for this page only */ } },
};
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
const fd = (d) => (d > 0 ? "+" : d < 0 ? "−" : "±") + Math.abs(d).toFixed(2);
// Accepts "+1.5", "-1,5", "−1.50": the typographic minus and a decimal comma both parse.
const parseDose = (s) => { const t = String(s).trim().replace("−", "-").replace(",", "."); return /^[+-]?\d*\.?\d+$/.test(t) ? parseFloat(t) : NaN; };
const at = (sp) => (sp.token != null ? `token ${sp.token}` : "this word");
const nid = (r) => `L${r.layer}N${r.neuron}`;
// An edit is a neuron row {layer, neuron, dose} or a behavior {trait, dose}. A run uses neurons or one behavior.
const isB = (r) => r.trait != null;
const eid = (r) => (isB(r) ? r.trait : nid(r));
const editsText = (rows) => (rows.length ? rows.map((x) => `${eid(x)} ${fd(x.dose)}`).join(", ") : "no edit");
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
if (!/Mac|iPhone|iPad/.test(navigator.platform)) document.querySelectorAll(".kbd").forEach((k) => (k.textContent = "Ctrl Enter"));

/* ---------- theme, shelf ---------- */
root.dataset.theme = store.get("theme2", "dark");
const syncTheme = () => ($("themeBtn").textContent = root.dataset.theme === "dark" ? "Light" : "Dark");
$("themeBtn").addEventListener("click", () => { root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark"; store.set("theme2", root.dataset.theme); syncTheme(); layout(); });
syncTheme();
function setShelf(open) {
  $("shell").classList.toggle("collapsed", !open); store.set("shelf", open);
  (open ? $("collapse") : $("expand")).focus({ preventScroll: true });
  requestAnimationFrame(layout);
}
$("collapse").addEventListener("click", () => setShelf(false));
$("expand").addEventListener("click", () => setShelf(true));
if (store.get("shelf", true) === false) $("shell").classList.add("collapsed");

/* ---------- state ---------- */
const S = {
  api: false, state: null, running: false, view: "work", inspect: null, focusEdit: 0, sel: null,
  limits: { layers: 32, neurons: 9216, dose_min: -3, dose_max: 3, max_rows: 4 },
  model: null, rows: [], example: false,
  // Guard: on unless research mode (session only, confirmed in a dialog). guardSeen: last guard result per setup.
  guard: true, research: false, guardSeen: {}, checking: false,
  jobs: [], traits: null, families: null, live: null, page: null, read: null,
};
// Experiments: one per (model, prompt). Run 1 is the baseline; every steered result adds a run.
const expKey = (model, prompt) => `${model}\n${prompt.trim()}`;
let EXPS = store.get("exps", {});
const saveExps = () => store.set("exps", EXPS);
const curExp = () => EXPS[expKey(S.model, $("prompt").value)] || null;
const rowsKey = (rows) => rows.filter((r) => r.dose !== 0).map((r) => `${eid(r)}@${r.dose}`).sort().join(" ");

function addResult(res, soft = false) {
  const k = expKey(res.model_id, res.prompt);
  let e = EXPS[k];
  if (e && e.baseline.text !== res.baseline.text && !soft) {
    // Greedy decoding with a fixed seed gives the same baseline; if it changed, the settings changed. Start over.
    $("runMsg").textContent = "The baseline differs from this experiment's run 1, so a new experiment starts.";
    e = null;
  }
  if (!e) e = EXPS[k] = { model: res.model_id, prompt: res.prompt, baseline: res.baseline, runs: [], at: Date.now() };
  const key = rowsKey(res.rows);
  const prev = e.runs.findIndex((r) => rowsKey(r.rows) === key);
  const run = { n: prev >= 0 ? e.runs[prev].n : e.runs.length + 2, rows: res.rows, text: res.steered.text, tokens: res.steered.tokens,
    split: res.split && { token: res.split.token, char: res.split.steered_char, base_char: res.split.base_char },
    seconds: res.seconds, recipe: res.recipe, post: res.post, research: !!res.research,
    dial: res.dial && res.dial.steered && res.dial.steered.length ? { trait: res.dial.trait, toks: res.dial.steered } : undefined, proof: prev >= 0 ? e.runs[prev].proof : undefined, at: Date.now() };
  if (prev >= 0) e.runs[prev] = run; else e.runs.push(run);
  if (res.dial && res.dial.baseline && res.dial.baseline.length) e.baseDial = { trait: res.dial.trait, toks: res.dial.baseline };
  e.at = Date.now();
  saveExps();
  return run.n;
}

function baselineOnly(res) {
  const k = expKey(res.model_id, res.prompt);
  if (!EXPS[k] || EXPS[k].baseline.text !== res.baseline.text) EXPS[k] = { model: res.model_id, prompt: res.prompt, baseline: res.baseline, runs: [], at: Date.now() };
  // At dose 0 the steered pass is the baseline, so its streamed dial values read run 1.
  const bt = res.dial && (res.dial.baseline && res.dial.baseline.length ? res.dial.baseline : res.dial.steered);
  if (bt && bt.length) EXPS[k].baseDial = { trait: res.dial.trait, toks: bt };
  EXPS[k].at = Date.now(); saveExps(); drawN = 1;
}

export { $, root, store, esc, fd, parseDose, at, nid, isB, eid, editsText, reduce, syncTheme, setShelf, S, expKey, EXPS, saveExps, curExp, rowsKey, addResult, baselineOnly };
