import { $, EXPS, S } from "./state.js";
import { K } from "./components/dial.js";
import { applyState, seedRecorded, update } from "./screens/workbench.js";
import { loadBehaviors } from "./screens/pages.js";
import { setStatus } from "./screens/run.js";
import "./screens/evidence.js";
import "./screens/share.js";

await document.fonts.ready;
K.draw();
try {
  const r = await fetch("api/state"); if (!r.ok) throw new Error();
  applyState(await r.json()); S.api = true;
  await loadBehaviors();
} catch { S.api = false; }
await seedRecorded();
const last = Object.values(EXPS).filter((e) => !S.model || e.model === S.model).sort((a, b) => b.at - a.at)[0];
if (!S.model) S.model = last ? last.model : "Qwen/Qwen3.5-4B";
$("prompt").value = last ? last.prompt : "";
const lastRun = last && last.runs.filter((x) => x.split).sort((a, b) => b.at - a.at)[0];
S.rows = lastRun ? lastRun.rows.map((x) => ({ ...x })) : (S.state ? S.state.preset_rows.map((x) => ({ ...x })) : []);
update();
if (!S.api) setStatus("err", "OFFLINE", "The lobot server is not reachable. The tree shows saved runs. Start the server with: uv run python server.py");
else if (S.state.replay) setStatus("info", "REPLAY", "Replay mode: no weights are loaded. Run answers only setups that were recorded.");
else if (S.state.no_model) setStatus("warn", "NO WEIGHTS", "The server runs without weights (MODEL_LOBOTOMY_NO_MODEL). Runs return an error.");

export { last, lastRun };
