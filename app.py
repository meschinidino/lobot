"""lobot Gradio UI (T6).

Minimal before/after loop: prompt in, baseline + steered columns out, with
first-divergence highlight, recipe editor (dose slider + numeric fallback
per row), and a Copy-for-X artifact box.

Doc: UI row + D6 (loading state) + D9 (custom HTML highlight, queue,
warm model, input validation) + D10 (bounded dose slider per row).
Render-after (recorded minor R2-4 leaves streaming open): the highlight
renders once both outputs are final.

Out of scope (v1.5): filmstrip, recipe replay/parse, neuron graph.

Run:
    uv run python app.py            # loads rung-1 weights, serves localhost:7860
    MODEL_LOBOTOMY_NO_MODEL=1 uv run python app.py   # layout smoke, no weights
"""

from __future__ import annotations

import difflib
import html
import json
import os
import threading
import time
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import HfApi

import steer
from engine import ENGINE
from steer import RecipeRow

ROOT = Path(__file__).resolve().parent

MAX_ROWS = 4          # recipe editor slots (evil-qwen needs 2)
DOSE_MIN, DOSE_MAX = -3.0, 3.0   # bounded slider range (D10 doubles as validation)
SEED = 0              # greedy ignores the seed; recorded for future sampling (R2-1)


def _load_presets() -> dict:
    """Presets per model id, from presets/*.json (Qwen ships; Llama lands later)."""
    found = {}
    for p in sorted((ROOT / "presets").glob("*.json")):
        d = json.loads(p.read_text())
        if isinstance(d, dict) and "model_id" in d and "rows" in d:
            found[d["model_id"]] = d
    if steer.MODEL_ID not in found:
        raise RuntimeError("default Qwen preset missing from presets/")
    return found


_PRESETS = _load_presets()

# Warm engine state (one model warm at a time; switching unloads first).
_model = None
_tokenizer = None
_model_id: str | None = None
_rung_n: int | None = None
_revision: str | None = None
_run_lock = threading.Lock()
_last_artifact: str = ""
_active_model_id: str = steer.MODEL_ID  # Qwen default; dropdown switches


def active_preset() -> dict:
    return _PRESETS[_active_model_id]


def active_spec():
    return steer.get_spec(_active_model_id)


def unload_engine() -> None:
    """Drop the warm model and free device memory (model switch)."""
    global _model, _tokenizer, _model_id, _rung_n, _revision
    _model, _tokenizer, _model_id, _rung_n, _revision = None, None, None, None, None
    import gc
    gc.collect()
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def load_engine(model_id: str | None = None):
    """Load the selected model once; keep it warm. Qwen rides the ladder."""
    global _model, _tokenizer, _model_id, _rung_n, _revision
    model_id = model_id or _active_model_id
    if _model is not None and _model_id == model_id:
        return _model, _tokenizer, _rung_n
    if _model is not None:
        unload_engine()
    _revision = HfApi().model_info(model_id).sha
    if model_id == steer.MODEL_ID:
        _model, _tokenizer, _rung_n, attempts = steer.load_with_ladder()
        for a in attempts:
            print(f"[engine] rung {a['rung']}: ok={a['ok']} {a['detail']}", flush=True)
        if _rung_n != 1:
            # Presets bind to model id (D2); non-rung-1 needs re-derivation (D8).
            raise RuntimeError(
                f"ladder stopped at rung {_rung_n}, not rung 1; preset rows are "
                f"invalid until re-derived ({steer.RECOVERY_PROCEDURE})"
            )
    else:
        spec = steer.get_spec(model_id)
        _model = steer.load_model(model_id)
        _tokenizer = steer.load_tokenizer(model_id)
        _rung_n = 1
        print(f"[engine] {model_id} loaded direct "
              f"(dims {spec.layers}/{spec.hidden}/{spec.intermediate})", flush=True)
    _model_id = model_id
    preset = _PRESETS[model_id]
    steer.check_preset_binding(preset, model_id)
    steer.assert_text_dims(_model, steer.get_spec(model_id))
    return _model, _tokenizer, _rung_n


# -- diff + highlight ---------------------------------------------------------

def first_divergence(a: list[int], b: list[int]) -> int | None:
    """First token index where the id streams differ; None if identical."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def highlight_steered(tokenizer, steered_text: str, steered_ids: list[int],
                      div: int | None) -> str:
    """HTML for the steered column: mark divergence token onward.

    Token-id to char-span alignment: decode the steered prefix before the
    divergence index and locate it in the full text. If divergence lands at
    token 0 (or the prefix will not align), fall back to a word-level diff
    against the baseline text (doc: semantic/line diff fallback).
    """
    esc = html.escape(steered_text)
    box = ("<div style='border:1px solid #e2e2e2;border-radius:8px;"
           "padding:10px;max-height:420px;overflow-y:auto'>")
    if div is None:
        return f"{box}<p>{esc}</p><p><em>Outputs identical.</em></p></div>"
    prefix = tokenizer.decode(steered_ids[:div], skip_special_tokens=True) if div > 0 else ""
    cut = steered_text.find(prefix) + len(prefix) if prefix and prefix in steered_text else -1
    if div > 0 and cut > 0:
        before = html.escape(steered_text[:cut])
        after = html.escape(steered_text[cut:])
        mark = "<mark style='background:#ffe08a'>"
        return f"{box}<p>{before}{mark}{after}</mark></p></div>"
    return None  # caller falls back to word diff


def highlight_word_diff(baseline_text: str, steered_text: str) -> str:
    """Word-level diff fallback: highlight changed/inserted steered words."""
    a = baseline_text.split()
    b = steered_text.split()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    parts = []
    for tag, _, _, j1, j2 in sm.get_opcodes():
        words = " ".join(html.escape(w) for w in b[j1:j2])
        if tag == "equal":
            parts.append(words)
        elif words:
            parts.append(f"<mark style='background:#ffe08a'>{words}</mark>")
    box = ("<div style='border:1px solid #e2e2e2;border-radius:8px;"
           "padding:10px;max-height:420px;overflow-y:auto'>")
    return box + "<p>" + " ".join(parts) + "</p></div>"


# -- artifact (T7 shared template in steer.py; T10 owns the grammar doc) ------

def build_artifact(prompt: str, rows: list[RecipeRow], calib_id: str,
                   base: dict, steered: dict) -> str:
    preset = active_preset()
    return steer.build_x_artifact(
        prompt=prompt, rows=rows, calib_id=calib_id, base=base,
        steered=steered, rung_n=_rung_n,
        max_new_tokens=int(preset["decode"]["max_new_tokens"]),
        seed=SEED, calib_prompts=list(preset["calibration_prompts"]),
        model_id=_active_model_id)


# -- run ----------------------------------------------------------------------

def collect_rows(n: int, *vals) -> list[RecipeRow]:
    """Collect (layer, neuron, dose) triples from the n active row slots."""
    rows = []
    for i in range(min(n, MAX_ROWS)):
        layer, neuron, dose = vals[i * 3:(i + 1) * 3]
        rows.append(RecipeRow(layer=int(layer), neuron=int(neuron), dose=float(dose)))
    return rows


def run_loop(prompt: str, n: int, *row_vals):
    """One before/after run. Returns (status, baseline_md, steered_html, artifact).

    Signature frozen (tests call it positionally): the model comes from the
    dropdown via _active_model_id, default Qwen.
    """
    global _last_artifact
    spec = active_spec()
    preset = active_preset()
    calib_prompts = list(preset["calibration_prompts"])
    max_new = int(preset["decode"]["max_new_tokens"])
    empty = ("", "", "", _last_artifact or "Run first: no artifact yet.")
    if not prompt or not prompt.strip():
        return ("Empty prompt: type something first.",) + empty[1:]
    try:
        rows = collect_rows(int(n), *row_vals)
    except (TypeError, ValueError):
        return ("Invalid row: layer/neuron must be integers, dose numeric.",) + empty[1:]
    if not rows:
        return ("Empty recipe: enable at least one row.",) + empty[1:]
    try:
        steer.validate_recipe(rows, spec)
    except ValueError as e:
        return (f"Invalid row: {e}",) + empty[1:]
    for r in rows:
        if not DOSE_MIN <= r.dose <= DOSE_MAX:
            return (f"Dose {r.dose:g} out of bounds [{DOSE_MIN:g},{DOSE_MAX:g}]; "
                     "clamp it or move the slider.",) + empty[1:]
    if not _run_lock.acquire(blocking=False):
        return ("A run is already in progress; wait for it to finish.",) + empty[1:]
    try:
        model, tok, _ = load_engine()
        layers = sorted({r.layer for r in rows})
        key = steer.calibration_cache_key(
            _active_model_id, _revision, tok, str(steer.DTYPE), steer.DEVICE)
        with ENGINE.clean():
            medians, cal_info = steer.calibrate_layer_medians(
                model, tok, calib_prompts, layers,
                cache_path=ROOT / steer.calibration_path_for(_active_model_id),
                key=key, spec=spec)
        calib_id = cal_info["key"]["prompts_sha256"]

        t0 = time.perf_counter()
        steer.reset_hook_call_count()
        with ENGINE.clean():
            base = steer.generate_text(model, tok, prompt.strip(),
                                       max_new_tokens=max_new, seed=SEED, spec=spec)
        with ENGINE.apply(model, rows, medians=medians, spec=spec):
            steered = steer.generate_text(model, tok, prompt.strip(),
                                          max_new_tokens=max_new, seed=SEED, spec=spec)
            hook_calls = steer.hook_call_count()
        wall = round(time.perf_counter() - t0, 1)
        if hook_calls < 1:
            return ("Steering hooks did not fire; outputs untrusted.",) + empty[1:]

        div = first_divergence(base["token_ids"], steered["token_ids"])
        steered_html = highlight_steered(tok, steered["text"], steered["token_ids"], div)
        if steered_html is None:  # divergence at 0 or unaligned: word-diff fallback
            steered_html = highlight_word_diff(base["text"], steered["text"])
        _last_artifact = build_artifact(prompt.strip(), rows, calib_id, base, steered)
        div_msg = "identical" if div is None else f"first divergence at token {div}"
        status = (f"Done in {wall}s ({div_msg}; cache_hit={cal_info['cache_hit']}). "
                  f"Artifact ready to copy.")
        return status, base["text"], steered_html, _last_artifact
    except Exception as e:  # noqa: BLE001 — surface engine errors in the UI
        return (f"Run failed: {e}",) + empty[1:]
    finally:
        _run_lock.release()


def copy_artifact() -> str:
    """Copy-for-X: empty-safe before the first run (test plan edge case)."""
    return _last_artifact or "Run first: no artifact yet."


# -- layout -------------------------------------------------------------------

def switch_model(model_id: str):
    """Dropdown handler: swap active model + preset rows, unload warm engine.

    The next Run loads the newly selected model (~10 s). Returns updates
    for [status, n_rows, *row_widgets_all] (layer, neuron, slider, numeric).
    """
    global _active_model_id
    _active_model_id = model_id
    unload_engine()
    spec = active_spec()
    rows = list(active_preset()["rows"])
    n = min(max(1, len(rows)), MAX_ROWS)
    out: list = [f"Switched to {model_id}; the next run loads it (~10 s).", n]
    for i in range(MAX_ROWS):
        d = rows[i] if i < len(rows) else None
        out += [
            gr.update(value=d["layer"] if d else 0, minimum=0,
                      maximum=spec.layers - 1,
                      label=f"Row {i + 1} layer (0-{spec.layers - 1})"),
            gr.update(value=d["neuron"] if d else 0, minimum=0,
                      maximum=spec.intermediate - 1,
                      label=f"Row {i + 1} neuron (0-{spec.intermediate - 1})"),
            gr.update(value=d["dose"] if d else 0.0),
            gr.update(value=d["dose"] if d else 0.0),
        ]
    out += [gr.update(visible=i < n) for i in range(MAX_ROWS)]
    return out


def build_demo() -> gr.Blocks:
    default_preset = _PRESETS[steer.MODEL_ID]
    default_rows: list = list(default_preset["rows"])
    with gr.Blocks(title="lobot") as demo:
        gr.Markdown("# lobot — before/after steering loop")
        gr.Markdown("Pick a model (one stays warm; switching reloads). "
                    "Greedy, seed 0. Highlight marks the "
                    "first-diverging token onward in the steered column.")
        model_pick = gr.Dropdown(choices=sorted(_PRESETS), value=steer.MODEL_ID,
                                 label="Model")
        n_rows = gr.Number(value=min(len(default_rows), MAX_ROWS),
                           visible=False, precision=0)
        row_groups: list[gr.Group] = []
        row_widgets: list = []  # layer, neuron, dose triples per slot
        row_widgets_all: list = []  # + slider, for model-switch updates
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt", lines=4,
                                    placeholder="Ask something the steered model answers badly...")
                with gr.Group():
                    gr.Markdown("### Recipe rows (L = layer, N = neuron)")
                    for i in range(MAX_ROWS):
                        d = default_rows[i] if i < len(default_rows) else None
                        with gr.Group(visible=i < 2) as g:
                            row_groups.append(g)
                            layer = gr.Number(label="Row {0} layer (0-31)".format(i + 1),
                                              value=d["layer"] if d else 0,
                                              precision=0, minimum=0, maximum=31)
                            neuron = gr.Number(label="Row {0} neuron (0-9215)".format(i + 1),
                                               value=d["neuron"] if d else 0,
                                               precision=0, minimum=0, maximum=9215)
                            dose = gr.Slider(label=f"Row {i + 1} dose",
                                             minimum=DOSE_MIN, maximum=DOSE_MAX,
                                             step=0.1, value=d["dose"] if d else 0.0)
                            dose_num = gr.Number(label=f"Row {i + 1} dose (numeric)",
                                                 value=d["dose"] if d else 0.0)
                            dose.change(lambda v: v, dose, dose_num)
                            dose_num.change(lambda v: v, dose_num, dose)
                            row_widgets += [layer, neuron, dose_num]
                            row_widgets_all += [layer, neuron, dose, dose_num]
                    with gr.Row():
                        add_btn = gr.Button("Add row", size="sm")
                        del_btn = gr.Button("Remove row", size="sm")
                run_btn = gr.Button("Run before/after", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=2):
                with gr.Row():
                    base_out = gr.Textbox(label="Baseline", lines=14)
                    with gr.Column():
                        gr.Markdown("**Steered (highlighted)**")
                        steered_out = gr.HTML()
                with gr.Row():
                    copy_btn = gr.Button("Copy for X", size="sm")
                artifact = gr.Textbox(label="X artifact (select + copy)", lines=10)

        def _set_rows(n: int, delta: int):
            n = min(MAX_ROWS, max(1, n + delta))
            return [n] + [gr.update(visible=i < n) for i in range(MAX_ROWS)]

        add_btn.click(_set_rows, [n_rows, gr.State(1)], [n_rows, *row_groups])
        del_btn.click(_set_rows, [n_rows, gr.State(-1)], [n_rows, *row_groups])
        model_pick.change(switch_model, [model_pick],
                          [status, n_rows, *row_widgets_all, *row_groups])
        run_btn.click(run_loop, [prompt, n_rows, *row_widgets],
                      [status, base_out, steered_out, artifact],
                      show_progress="full")
        copy_btn.click(copy_artifact, [], [artifact])
    demo.queue(max_size=1)
    return demo


if __name__ == "__main__":
    if os.environ.get("MODEL_LOBOTOMY_NO_MODEL"):
        print("[app] NO_MODEL set: serving layout without weights", flush=True)
    else:
        load_engine()
        print(f"[app] engine warm ({_model_id} rung {_rung_n}); models: "
              f"{sorted(_PRESETS)}", flush=True)
    build_demo().launch(server_name="127.0.0.1", server_port=7860)