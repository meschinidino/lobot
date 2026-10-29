"""model-lobotomy Gradio UI (T6).

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
from steer import RecipeRow

ROOT = Path(__file__).resolve().parent
PRESET_PATH = ROOT / "presets" / "evil-qwen.json"
CALIBRATION_PATH = ROOT / "calibration.json"

MAX_ROWS = 4          # recipe editor slots (evil-qwen needs 2)
DOSE_MIN, DOSE_MAX = -3.0, 3.0   # bounded slider range (D10 doubles as validation)
SEED = 0              # greedy ignores the seed; recorded for future sampling (R2-1)

_preset = json.loads(PRESET_PATH.read_text())
CALIB_PROMPTS: list[str] = list(_preset["calibration_prompts"])
MAX_NEW_TOKENS: int = int(_preset["decode"]["max_new_tokens"])
PRESET_ROWS: list[dict] = list(_preset["rows"])

# Warm engine state (loaded once, reused across runs).
_model = None
_tokenizer = None
_rung_n: int | None = None
_revision: str | None = None
_run_lock = threading.Lock()
_last_artifact: str = ""


def load_engine():
    """Load the model once via the fallback ladder; keep it warm."""
    global _model, _tokenizer, _rung_n, _revision
    if _model is not None:
        return _model, _tokenizer, _rung_n
    _revision = HfApi().model_info(steer.MODEL_ID).sha
    _model, _tokenizer, _rung_n, attempts = steer.load_with_ladder()
    for a in attempts:
        print(f"[engine] rung {a['rung']}: ok={a['ok']} {a['detail']}", flush=True)
    if _rung_n != 1:
        # Presets bind to model id (D2); non-rung-1 needs re-derivation (D8).
        raise RuntimeError(
            f"ladder stopped at rung {_rung_n}, not rung 1; preset rows are "
            f"invalid until re-derived ({steer.RECOVERY_PROCEDURE})"
        )
    steer.check_preset_binding(_preset, steer.MODEL_ID)
    steer.assert_text_dims(_model)
    return _model, _tokenizer, _rung_n


def mps_mib() -> float | None:
    """Current MPS allocated memory in MiB (None off-MPS)."""
    try:
        if torch.backends.mps.is_available():
            return round(torch.mps.current_allocated_memory() / 1024**2, 1)
    except Exception:
        pass
    return None


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


# -- artifact (basic v1 body; T7 adds measurements, T10 fixes the grammar) ----

def recipe_string(rows: list[RecipeRow], calib_id: str) -> str:
    body = ";".join(f"L{r.layer}N{r.neuron}@{r.dose:g}" for r in rows)
    return (
        f"model-lobotomy/v1 model={steer.MODEL_ID} rows=[{body}] "
        f"calib={calib_id} template=chat+no-think max_tokens={MAX_NEW_TOKENS} "
        f"decode=greedy,seed={SEED} dtype={steer.DTYPE} device={steer.DEVICE}"
    )


def build_artifact(prompt: str, rows: list[RecipeRow], calib_id: str,
                   base: dict, steered: dict, mem_mib: float | None) -> str:
    row_lines = "\n".join(f"  L{r.layer} N{r.neuron} dose {r.dose:g}" for r in rows)
    mem = f"{mem_mib} MiB (MPS current)" if mem_mib is not None else "n/a"
    return (
        f"model-lobotomy before/after\n"
        f"model: {steer.MODEL_ID} (rung {_rung_n})\n"
        f"recipe:\n{row_lines}\n"
        f"calibration: {calib_id}\n"
        f"latency: baseline {base['seconds']}s / steered {steered['seconds']}s "
        f"({base['new_tokens']}/{steered['new_tokens']} new tokens)\n"
        f"peak memory: {mem}\n"
        f"prompt: {prompt}\n"
        f"--- baseline ---\n{base['text']}\n"
        f"--- steered ---\n{steered['text']}\n"
        f"recipe: {recipe_string(rows, calib_id)}"
    )


# -- run ----------------------------------------------------------------------

def collect_rows(n: int, *vals) -> list[RecipeRow]:
    """Collect (layer, neuron, dose) triples from the n active row slots."""
    rows = []
    for i in range(min(n, MAX_ROWS)):
        layer, neuron, dose = vals[i * 3:(i + 1) * 3]
        rows.append(RecipeRow(layer=int(layer), neuron=int(neuron), dose=float(dose)))
    return rows


def run_loop(prompt: str, n: int, *row_vals):
    """One before/after run. Returns (status, baseline_md, steered_html, artifact)."""
    global _last_artifact
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
        steer.validate_recipe(rows)
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
            steer.MODEL_ID, _revision, tok, str(steer.DTYPE), steer.DEVICE)
        medians, cal_info = steer.calibrate_layer_medians(
            model, tok, CALIB_PROMPTS, layers,
            cache_path=CALIBRATION_PATH, key=key)
        calib_id = cal_info["key"]["prompts_sha256"]

        t0 = time.perf_counter()
        steer.reset_hook_call_count()
        base = steer.generate_text(model, tok, prompt.strip(),
                                   max_new_tokens=MAX_NEW_TOKENS, seed=SEED)
        steer.register_steering_hooks(model, rows, medians, mode="add")
        try:
            steered = steer.generate_text(model, tok, prompt.strip(),
                                          max_new_tokens=MAX_NEW_TOKENS, seed=SEED)
            hook_calls = steer.hook_call_count()
        finally:
            steer.remove_steering_hooks()
        wall = round(time.perf_counter() - t0, 1)
        if hook_calls < 1:
            return ("Steering hooks did not fire; outputs untrusted.",) + empty[1:]

        div = first_divergence(base["token_ids"], steered["token_ids"])
        steered_html = highlight_steered(tok, steered["text"], steered["token_ids"], div)
        if steered_html is None:  # divergence at 0 or unaligned: word-diff fallback
            steered_html = highlight_word_diff(base["text"], steered["text"])
        mem = mps_mib()
        _last_artifact = build_artifact(prompt.strip(), rows, calib_id, base, steered, mem)
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

def build_demo() -> gr.Blocks:
    with gr.Blocks(title="model-lobotomy") as demo:
        gr.Markdown("# model-lobotomy — before/after steering loop")
        gr.Markdown(f"Model `{steer.MODEL_ID}` stays warm across runs. "
                    "Greedy, seed 0, "
                    f"{MAX_NEW_TOKENS} max new tokens. Highlight marks the "
                    "first-diverging token onward in the steered column.")
        n_rows = gr.Number(value=2, visible=False, precision=0)
        row_groups: list[gr.Group] = []
        row_widgets: list = []  # layer, neuron, dose triples per slot
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt", lines=4,
                                    placeholder="Ask something the steered model answers badly...")
                with gr.Group():
                    gr.Markdown("### Recipe rows (L = layer, N = neuron)")
                    for i in range(MAX_ROWS):
                        d = PRESET_ROWS[i] if i < len(PRESET_ROWS) else None
                        with gr.Group(visible=i < 2) as g:
                            row_groups.append(g)
                            layer = gr.Number(label=f"Row {i + 1} layer (0-31)",
                                              value=d["layer"] if d else 0,
                                              precision=0, minimum=0, maximum=31)
                            neuron = gr.Number(label=f"Row {i + 1} neuron (0-9215)",
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
        print(f"[app] engine warm (rung {_rung_n}); calibration prompts: "
              f"{len(CALIB_PROMPTS)}", flush=True)
    build_demo().launch(server_name="127.0.0.1", server_port=7860)