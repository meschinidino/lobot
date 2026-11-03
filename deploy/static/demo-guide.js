// Static demo: a guide that lists only what this export can play.
// Loads after static-api.js. It reads the export through window.lobotDemo.
(() => {
  const KEY = "lobot-demo-guide";
  const REPO = "https://github.com/meschinidino/lobot";
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const short = (id) => id.split("/").pop();
  const store = {
    get() { try { return localStorage.getItem(KEY); } catch { return null; } },
    set(v) { try { localStorage.setItem(KEY, v); } catch { /* private window: keep it in memory */ } },
  };

  const css = `
#demoGuide{position:fixed;right:16px;bottom:16px;z-index:50;width:340px;max-height:min(70vh,560px);overflow:auto;
  background:var(--sheet);color:var(--ink);border:1px solid var(--ink);font:1rem/1.5 var(--body);padding:16px}
#demoGuide h2{font:800 28px/1 var(--display);font-stretch:75%;margin:0}
#demoGuide .head{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px}
#demoGuide p{margin:0 0 8px}
#demoGuide ol{margin:0 0 8px;padding-left:20px}
#demoGuide li{margin-bottom:8px}
#demoGuide .muted{color:var(--ink-muted);font-size:14px}
#demoGuide code{font:14px var(--mono);font-feature-settings:"tnum"}
#demoGuide button,#demoGuideOpen{font:700 14px var(--body);background:transparent;color:var(--ink);border:1px solid var(--ink);
  border-radius:0;padding:4px 8px;cursor:pointer;margin:4px 4px 0 0;text-align:left}
#demoGuide button:hover,#demoGuideOpen:hover{background:var(--band)}
#demoGuide a{color:var(--ink)}
#demoGuideOpen{position:fixed;right:16px;bottom:16px;z-index:50;background:var(--sheet);margin:0;padding:8px 16px}
@media (max-width:640px){#demoGuide{left:16px;width:auto;max-height:60vh}}`;

  function body(meta) {
    const model = window.lobotDemo.model() || meta.start_model;
    const setups = window.lobotDemo.setups(meta, model);
    const explained = meta.guide.explained.filter((t) => t.model_id === model);
    const others = Object.keys(meta.states).filter((m) => m !== model && meta.recordings.some((r) => r.model_id === m));
    const steps = [];
    if (setups.length) {
      const runs = setups.map((s) => `<p>Prompt: <i>${esc(s.prompt)}</i><br>${s.recipes.map((r) => `<code>${esc(r)}</code>`).join("<br>")}</p>`).join("");
      steps.push(`<b>Run a recorded setup.</b> To load its prompt and edits, open its run under Experiments in the shelf. Turn the knob to a recorded dose, then press Run. Each run adds a branch where the answer changes.${runs}`);
      steps.push("<b>Read a neuron.</b> Click a word in a run. Under \"Applied edits in this run\", press the neuron to see what it pushes toward.");
    } else {
      steps.push("<b>No recorded runs on this model.</b> Switch the model to see runs.");
    }
    if (explained.length) {
      const btns = explained.map((t) => `<button type="button" data-open="${esc(t.id)}">${esc(t.family)} v${t.version}</button>`).join("");
      steps.push(`<b>See what Muse Spark says.</b> Open a behavior, then scroll to "In plain words" and "Second opinion".<br>${btns}`);
    }
    steps.push("<b>Open any behavior</b> in the shelf to see its evidence page: the tests, the dose dial and the guard.");
    if (others.length) {
      steps.push(`<b>Switch the model.</b> ${others.map((m) => `<code>${esc(short(m))}</code>`).join(", ")} has recorded runs too.`);
    }
    return `<div class="head"><h2>Demo guide</h2><button type="button" data-hide>Hide</button></div>
<p>This demo replays recorded runs of real models. No model runs in your browser, and nothing is saved.</p>
<ol>${steps.map((s) => `<li>${s}</li>`).join("")}</ol>
<p class="muted">These need lobot on your own computer: neuron search, Find with Muse, Tune with Muse, New behavior, chat and install. <a href="${REPO}" target="_blank" rel="noopener">Get lobot</a></p>`;
  }

  // The evidence page renders after its fetches: wait up to 3 s for the Muse section.
  function scrollToMuse(tries = 30) {
    const h = [...document.querySelectorAll("h2")].find((x) => x.textContent.startsWith("Second opinion"));
    if (h) h.scrollIntoView({ block: "start" });
    else if (tries > 0) setTimeout(() => scrollToMuse(tries - 1), 100);
  }

  async function start() {
    const meta = await window.lobotDemo.ready;
    const style = document.createElement("style");
    style.textContent = css;
    document.head.append(style);
    const panel = Object.assign(document.createElement("aside"), { id: "demoGuide" });
    panel.setAttribute("aria-label", "Demo guide");
    const open = Object.assign(document.createElement("button"), { id: "demoGuideOpen", type: "button", textContent: "Demo guide" });
    document.body.append(panel, open);
    const show = (on) => {
      panel.hidden = !on; open.hidden = on;
      if (on) panel.innerHTML = body(meta);
    };
    panel.addEventListener("click", (e) => {
      if (e.target.closest("[data-hide]")) { show(false); store.set("hidden"); open.focus(); return; }
      const t = e.target.closest("[data-open]");
      if (!t) return;
      const shelf = document.querySelector(`[data-beh="${CSS.escape(t.dataset.open)}"]`);
      if (shelf) { shelf.click(); scrollToMuse(); }
    });
    open.addEventListener("click", () => { show(true); store.set("open"); });
    window.addEventListener("lobot-demo:model", () => { if (!panel.hidden) panel.innerHTML = body(meta); });
    // On a phone the open guide covers the page, so it starts as a button there.
    const saved = store.get();
    show(saved ? saved === "open" : !matchMedia("(max-width: 640px)").matches);
  }

  if (window.lobotDemo) start();
})();
