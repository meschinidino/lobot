// Static demo: answer the page's api/ calls from files that export.py wrote.
// Loads before js/app.js. Read-only: writes answer 403, Muse is never called.
(() => {
  const BASE = new URL("static-api/", document.baseURI);
  const READ_ONLY = "This public demo is read-only. Run lobot on your own computer to use this feature.";
  const NO_SAVED = "This public demo shows only saved Muse Spark answers, and this report has none.";
  const STAGE_MS = 400;
  const realFetch = window.fetch.bind(window);
  const load = (name) => realFetch(new URL(name, BASE)).then((r) => r.json());
  const ready = Promise.all([load("meta.json"), load("index.json")]);
  let model = null, running = false, cancelled = false;

  // 32-bit FNV-1a, hex: the same file names as export.py.
  const fnv1a = (text) => {
    let h = 0x811c9dc5;
    for (const b of new TextEncoder().encode(text)) h = Math.imul(h ^ b, 0x01000193) >>> 0;
    return h.toString(16).padStart(8, "0");
  };
  const json = (status, body) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
  const saved = async (index, key, miss) => {
    if (!(key in index)) return json(miss[0], { detail: miss[1] });
    const f = await load(`${fnv1a(key)}.json`);
    return json(f.status, f.body);
  };

  const rowsKey = (rows) => rows.map((r) => `${r.layer}:${r.neuron}:${Number(r.dose).toFixed(4)}`).sort().join("|");
  function findRecording(recs, prompt, rows) {
    if (rows.length && rows.every((r) => Number(r.dose) === 0)) {
      const rec = recs.find((x) => x.model_id === model && x.prompt.trim() === prompt);
      return rec ? { ...rec, rows, steered: rec.baseline, split: null } : null;
    }
    return recs.find((x) => x.model_id === model && x.prompt.trim() === prompt && rowsKey(x.rows) === rowsKey(rows)) || null;
  }
  function replay(rec) {
    const enc = new TextEncoder(), line = (ev) => enc.encode(JSON.stringify(ev) + "\n");
    const wait = (ms) => new Promise((ok) => setTimeout(ok, ms));
    running = true; cancelled = false;
    return new Response(new ReadableStream({
      async start(ctl) {
        for (const stage of ["weights", "baseline", "steered"]) {
          ctl.enqueue(line({ stage }));
          await wait(STAGE_MS);
          if (cancelled) { ctl.enqueue(line({ stage: "stopped", after: stage })); running = false; ctl.close(); return; }
        }
        ctl.enqueue(line({ stage: "complete", result: { ...rec, dial: rec.dial ?? null, cache_hit: true, replay: true } }));
        running = false; ctl.close();
      },
    }), { status: 200, headers: { "Content-Type": "application/x-ndjson" } });
  }

  async function answer(method, path, search, body) {
    const [meta, index] = await ready;
    model = model || meta.start_model;
    if (path === "/api/state") return json(200, meta.states[model]);
    if (method === "GET") {
      if (path === "/api/find") return json(meta.find.status, meta.find.body);
      if (path.startsWith("/api/neuron/")) return saved(index, `GET ${model} ${path}${search}`, [404, "This static demo has no card for this neuron."]);
      return saved(index, `GET ${path}${search}`, [404, "This static demo has no saved answer for this page."]);
    }
    if (path === "/api/model") {
      if (!meta.states[body.model_id]) return json(400, { detail: `Unknown model ${body.model_id}.` });
      model = body.model_id;
      return json(200, meta.states[model]);
    }
    if (path === "/api/stop") { cancelled = true; return json(200, { stopping: running }); }
    if (path === "/api/run") {
      const prompt = (body.prompt || "").trim();
      if (!prompt) return json(400, { detail: "The prompt is empty. Type a prompt first." });
      if (!body.rows?.length) return json(400, { detail: "The recipe is empty. Add at least one row." });
      if (running) return json(409, { detail: "A run is in progress. Wait for it to finish." });
      const rec = findRecording(meta.recordings, prompt, body.rows);
      return rec ? replay(rec) : json(404, { detail: "This demo plays only recorded setups, and this one was not recorded. Run lobot on your own computer to run the model." });
    }
    if (/^\/api\/traits\/[^/]+\/(explain|second-opinion)$/.test(path)) return saved(index, `POST ${path}`, [503, NO_SAVED]);
    return json(403, { detail: READ_ONLY });
  }

  window.fetch = async (input, init = {}) => {
    const url = new URL(typeof input === "string" ? input : input.url, document.baseURI);
    const at = url.pathname.indexOf("/api/");
    if (url.origin !== location.origin || at < 0) return realFetch(input, init);
    const method = (init.method || (typeof input === "string" ? "GET" : input.method) || "GET").toUpperCase();
    let body = {};
    try { body = init.body ? JSON.parse(init.body) : {}; } catch { body = {}; }
    return answer(method, url.pathname.slice(at), url.search, body);
  };
})();
