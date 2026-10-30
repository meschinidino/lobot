"""lobot engine.

Load recipe (T3) + steering hooks and runner (T1) + fallback ladder (T2) +
calibration (T4) + hardcoded module path (T5). Smoke:
scripts/check_engine_t2t4t5.py [--with-model].

Load recipe (Eng D2 rung 1 + D3 + D9, verified by scripts/verify_load.py):
  model ......... Qwen/Qwen3.5-4B (public, ungated, safetensors, 2 shards)
  class ......... Qwen3_5ForConditionalGeneration (in transformers, no remote code)
  dtype ......... torch.float16 on MPS (config native bf16 noted, rung 1 says fp16)
  device ........ single MPS device, eager (plain .to("mps"), no accelerate dispatch)
  prompt wrap ... tokenizer.apply_chat_template (chat model, verified live at T3)
  weights ....... 9,319,737,856 param bytes (~8.7 GiB); disk budget 15 GiB (T3)
"""

from __future__ import annotations

import torch
from transformers import (AutoTokenizer, LlamaForCausalLM, Qwen3_5ForConditionalGeneration,
                          StoppingCriteria, StoppingCriteriaList)

MODEL_ID = "Qwen/Qwen3.5-4B"
LLAMA_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_REVISION: str | None = None  # float on main; resolved sha recorded by verify_load.py
DTYPE = torch.float16
DEVICE = "mps"
TRUST_REMOTE_CODE = False
MAX_NEW_TOKENS_PROBE = 16


def load_tokenizer(model_id: str = MODEL_ID):
    """Load the text tokenizer and assert the chat template exists."""
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=TRUST_REMOTE_CODE)
    if tok.chat_template is None:
        raise RuntimeError(f"{model_id} has no chat template; T3 load recipe needs one")
    return tok


def wrap_prompt(tokenizer, user_text: str) -> str:
    """Wrap a user prompt with the model's chat template (v1 rule)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def load_model(model_id: str = MODEL_ID, revision: str | None = MODEL_REVISION,
               device: str = DEVICE):
    """Load weights: fp16, eager, onto device. No hooks here (T1 owns hooks).

    device defaults to MPS (rung 1); the ladder passes "cpu" for rung 4 (T2).
    """
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; rung 1 needs MPS (see README ladder)")
    model = model_class_for(model_id).from_pretrained(
        model_id,
        revision=revision,
        dtype=DTYPE,
        trust_remote_code=TRUST_REMOTE_CODE,
    )
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# T1 (Day-0 spike gate): steering engine.
# Doc: D1 (hardcode path + assert), D7 (spike gate), D9 (calibration tap and
# hook isolation). The T3 load recipe above is untouched. T2 owns the ladder,
# T4 owns calibration tests, T5 owns the final path assert; T1 carries the
# minimum of each needed to prove the flip.
# ---------------------------------------------------------------------------

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

CAL_CODE_VERSION = "t1-v2"
# Expected text-tower dims for Qwen/Qwen3.5-4B (verified live at T1 probe).
TEXT_LAYERS = 32
TEXT_HIDDEN = 2560
TEXT_INTERMEDIATE = 9216
# Reference column norms from the sarvkk/two-neurons replication (NF4 model).
# Same order of magnitude expected on fp16; sanity check only, not a gate.
REF_COL_NORMS = {(14, 8949): 0.4632, (13, 7956): 0.4196}


@dataclass(frozen=True)
class ModelSpec:
    """Per-model engine facts: dims, block path, loader class, template flags."""
    model_id: str
    layers: int
    hidden: int
    intermediate: int
    block_path: str  # "qwen" (model.model.language_model.layers) or "llama" (model.model.layers)
    thinking_kw: bool  # whether the chat template takes enable_thinking


QWEN_SPEC = ModelSpec(MODEL_ID, 32, 2560, 9216, "qwen", True)
LLAMA_SPEC = ModelSpec(LLAMA_MODEL_ID, 28, 3072, 8192, "llama", False)

MODEL_SPECS: dict[str, ModelSpec] = {
    MODEL_ID: QWEN_SPEC,
    LLAMA_MODEL_ID: LLAMA_SPEC,
}


def get_spec(model_id: str) -> ModelSpec:
    """Spec for a supported model id; unknown ids fail loudly."""
    try:
        return MODEL_SPECS[model_id]
    except KeyError:
        raise RuntimeError(
            f"unsupported model {model_id!r}; supported: {sorted(MODEL_SPECS)}"
        ) from None


def model_class_for(model_id: str):
    """HF model class: registry spec first, family fallback for ladder rungs."""
    if model_id in MODEL_SPECS:
        return (Qwen3_5ForConditionalGeneration if MODEL_SPECS[model_id].block_path == "qwen"
                else LlamaForCausalLM)
    if "Qwen" in model_id:
        return Qwen3_5ForConditionalGeneration
    if "Llama" in model_id or "llama" in model_id:
        return LlamaForCausalLM
    raise RuntimeError(f"no model class for {model_id!r}")


def calibration_path_for(model_id: str) -> str:
    """Cache file per model. Qwen keeps the original calibration.json."""
    if model_id == MODEL_ID:
        return "calibration.json"
    slug = model_id.lower().replace("/", "-").replace("_", "-")
    return f"calibration.{slug}.json"


@dataclass(frozen=True)
class RecipeRow:
    layer: int
    neuron: int
    dose: float


def text_config(model):
    """Return the text-tower config (multimodal class wraps it)."""
    cfg = getattr(model.config, "text_config", model.config)
    return cfg


def assert_text_dims(model, spec: ModelSpec | None = None):
    """Loud check that the loaded tower matches the preset's dims (D1)."""
    spec = spec or QWEN_SPEC
    cfg = text_config(model)
    got = (cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size)
    want = (spec.layers, spec.hidden, spec.intermediate)
    if got != want:
        raise RuntimeError(f"text dims {got} != preset dims {want}; preset invalid for this model")


def validate_recipe(rows: list[RecipeRow], spec: ModelSpec | None = None) -> None:
    """Bounds check with clear errors (0-based; intermediate dim)."""
    spec = spec or QWEN_SPEC
    for r in rows:
        if not 0 <= r.layer < spec.layers:
            raise ValueError(f"layer {r.layer} out of bounds [0,{spec.layers})")
        if not 0 <= r.neuron < spec.intermediate:
            raise ValueError(f"neuron {r.neuron} out of bounds [0,{spec.intermediate})")


def resolve_block(model, layer: int, spec: ModelSpec):
    """Hardcoded decoder-block path per model family (D1)."""
    label = ("model.model.language_model.layers" if spec.block_path == "qwen"
             else "model.model.layers")
    try:
        base = model.model.language_model if spec.block_path == "qwen" else model.model
        return base.layers[layer]
    except (AttributeError, IndexError) as e:
        raise RuntimeError(
            f"module path mismatch: {label}[{layer}] unreachable ({e}); "
            "hardcoded path invalid for this model"
        ) from e


def resolve_down_proj(model, layer: int, spec: ModelSpec | None = None):
    """Hardcoded down_proj path per spec (D1).

    Fails loudly on any mismatch: missing attrs, wrong module type, or a
    weight layout that is not [hidden, intermediate].
    """
    spec = spec or QWEN_SPEC
    block = resolve_block(model, layer, spec)
    try:
        mod = block.mlp.down_proj
    except AttributeError as e:
        raise RuntimeError(f"module path mismatch: no mlp.down_proj at layer {layer}") from e
    if not isinstance(mod, torch.nn.Linear):
        raise RuntimeError(f"module path mismatch: down_proj at L{layer} is {type(mod)}, not Linear")
    cfg = text_config(model)
    want = (cfg.hidden_size, cfg.intermediate_size)
    if tuple(mod.weight.shape) != want:
        raise RuntimeError(f"weight layout mismatch at L{layer}: {tuple(mod.weight.shape)} != {want}")
    return mod


def column_norm(model, layer: int, neuron: int, spec: ModelSpec | None = None) -> float:
    """||W_down[:, n]|| for one target (float32 norm of the fp16 column)."""
    mod = resolve_down_proj(model, layer, spec)
    with torch.no_grad():
        return float(mod.weight.detach().float()[:, neuron].norm().item())


def wrap_prompt_notk(tokenizer, user_text: str, spec: ModelSpec | None = None) -> str:
    """Chat-template wrap with thinking disabled where the template has it.

    Qwen (T1 spike choice): the replication ground truth ran with thinking
    disabled; the default template opens a <think> block that pollutes
    rubric judging. Llama templates take no such flag. Recorded per preset.
    """
    spec = spec or QWEN_SPEC
    kwargs: dict = {}
    if spec.thinking_kw:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


# -- steering hook registry ---------------------------------------------------

_active_steering: list = []
_hook_calls: int = 0


def steering_hook_count() -> int:
    return len(_active_steering)


def hook_call_count() -> int:
    return _hook_calls


def reset_hook_call_count() -> None:
    global _hook_calls
    _hook_calls = 0


def register_steering_hooks(model, rows: list[RecipeRow], medians: dict, mode: str = "add",
                            spec: ModelSpec | None = None):
    """Register one pre-hook per target layer for the full generation.

    mode="add" (doc binding): x[..., n] += dose * median_L / ||W_down[:, n]||.
    mode="clamp" (fallback lane): x[..., n] = dose * median_L / ||W_down[:, n]||.
    """
    if mode not in ("add", "clamp"):
        raise ValueError(f"unknown steering mode {mode!r}")
    spec = spec or QWEN_SPEC
    validate_recipe(rows, spec)
    assert_text_dims(model, spec)
    by_layer: dict[int, list[RecipeRow]] = {}
    for r in rows:
        by_layer.setdefault(r.layer, []).append(r)
    for layer, lrows in by_layer.items():
        if layer not in medians:
            raise KeyError(f"no cached median for layer {layer}")
        med = float(medians[layer])
        deltas = {r.neuron: r.dose * med / column_norm(model, layer, r.neuron, spec) for r in lrows}

        def _prehook(_m, args, *, _d=dict(deltas), _mode=mode):
            global _hook_calls
            _hook_calls += 1
            x = args[0].clone()
            for n, d in _d.items():
                if _mode == "add":
                    x[..., n] = x[..., n] + d
                else:
                    x[..., n] = d
            return (x,) + args[1:]

        _active_steering.append(resolve_down_proj(model, layer, spec).register_forward_pre_hook(_prehook))


def remove_steering_hooks() -> None:
    while _active_steering:
        _active_steering.pop().remove()


# -- calibration ---------------------------------------------------------------

def _block_hidden(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        return out[0]
    for attr in ("last_hidden_state", "hidden_states"):
        if hasattr(out, attr):
            return getattr(out, attr)
    raise RuntimeError(f"cannot read block output from {type(out)}")


def calibration_cache_key(model_id, revision, tokenizer, dtype_str, device, quant=None):
    """Cache key: model id + revision + template + dtype/quant + device + cal code (D9)."""
    tmpl = tokenizer.chat_template or ""
    return {
        "model_id": model_id,
        "revision": revision,
        "template_sha256": hashlib.sha256(tmpl.encode()).hexdigest()[:16],
        "thinking": False,
        "dtype": dtype_str,
        "quant": quant,
        "device": device,
        "cal_code": CAL_CODE_VERSION,
    }


def calibrate_layer_medians(model, tokenizer, prompts: list[str], layers: list[int],
                             *, cache_path: str | Path, key: dict,
                             spec: ModelSpec | None = None):
    """Median residual norm leaving each layer L, pooled over all prompts' tokens 1:.

    Tap = decoder block output (D9). Guards: filter empties + 2-plus wrapped
    tokens per prompt (D4), steering hooks OFF with assert (D9 isolation).
    Reuses calibration.json on key match, else computes and writes it.
    """
    spec = spec or QWEN_SPEC
    if steering_hook_count() != 0:
        raise RuntimeError("steering hooks must be OFF during calibration (D9 isolation)")
    prompts = [p for p in prompts if p and p.strip()]
    if not prompts:
        raise ValueError("calibration needs at least one non-empty prompt (D4 filter empties)")
    cache_path = Path(cache_path)
    prompt_hashes = [hashlib.sha256(p.encode()).hexdigest()[:16] for p in prompts]
    full_key = dict(key)
    full_key["prompts_sha256"] = hashlib.sha256("\n".join(prompts).encode()).hexdigest()[:16]
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("key") == full_key and all(str(L) in cached.get("medians", {}) for L in layers):
            medians = {int(k): float(v) for k, v in cached["medians"].items()}
            return medians, {"cache_hit": True, "key": full_key}
    captured: dict[int, list] = {L: [] for L in layers}
    pooled: dict[int, list] = {L: [] for L in layers}
    counts: dict[int, list[int]] = {L: [] for L in layers}
    handles = []
    for L in layers:
        block = resolve_block(model, L, spec)

        def _cap(_m, _args, out, *, _L=L):
            captured[_L].append(_block_hidden(out).detach().float())

        handles.append(block.register_forward_hook(_cap))
    try:
        with torch.no_grad():
            for p in prompts:
                wrapped = wrap_prompt_notk(tokenizer, p, spec)
                ids = tokenizer(wrapped, return_tensors="pt")["input_ids"]
                if ids.numel() < 2:
                    raise ValueError(f"calibration prompt too short (<2 tokens): {p!r}")
                ids = ids.to(DEVICE)
                for L in layers:
                    captured[L].clear()
                model(ids)
                for L in layers:
                    hs = captured[L][-1][0, 1:, :]  # tokens 1:, skip BOS (D9)
                    counts[L].append(int(hs.shape[0]))
                    pooled[L].append(hs.norm(dim=-1))  # accumulate: one median over all prompts
    finally:
        for h in handles:
            h.remove()
    medians = {}
    for L in layers:
        norms = torch.cat(pooled[L])
        medians[L] = float(norms.median().item())
    merged_medians = {str(k): v for k, v in medians.items()}
    merged_counts = {str(k): v for k, v in counts.items()}
    if cache_path.exists():
        try:
            old = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            old = None
        if old and old.get("key") == full_key:
            # Same prompts: keep layers calibrated by earlier runs so a
            # subset run never drops them (clobber fixed 2026-10-29).
            for k, v in old.get("medians", {}).items():
                merged_medians.setdefault(k, v)
            for k, v in old.get("token_counts", {}).items():
                merged_counts.setdefault(k, v)
    payload = {
        "key": full_key,
        "medians": merged_medians,
        "prompt_hashes": prompt_hashes,
        "token_counts": merged_counts,
    }
    cache_path.write_text(json.dumps(payload, indent=2))
    return medians, {"cache_hit": False, "key": full_key}


# -- runner ---------------------------------------------------------------------

class CancelCriteria(StoppingCriteria):
    """Stops generate() once `event` is set (UI Stop). Checked after each token."""

    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.event.is_set(),
                          dtype=torch.bool, device=input_ids.device)


class RowStreamer:
    """Per-row token queue for one generate call (single or batched).

    generate() calls put() once per step with the new token ids of
    every row, shape (batch,) or (batch, 1); consumers take (row,
    token_id) events from the queue until the (None, None) sentinel
    from end(). Duck-typed as a transformers streamer (put/end only).
    """

    def __init__(self):
        import queue
        self.queue: queue.Queue = queue.Queue()

    def put(self, values):
        toks = values.tolist() if hasattr(values, "tolist") else list(values)
        if toks and isinstance(toks[0], list):
            # generate() first echoes the whole prompt (rows longer
            # than one token); only single-token rows are new tokens.
            # Row order is preserved: row r keeps index r.
            for r, row in enumerate(toks):
                if len(row) == 1:
                    self.queue.put((r, int(row[0])))
            return
        for r, t in enumerate(toks):
            self.queue.put((r, int(t)))

    def end(self):
        self.queue.put((None, None))


def generate_text(model, tokenizer, user_text: str, *, max_new_tokens: int, seed: int,
                  spec: ModelSpec | None = None, cancel=None, wrapped: str | None = None):
    """One greedy generation under whatever hooks are registered (T7 measured).

    Runs never exceed MAX_NEW_TOKENS_CAP (D6 budget). Returns the T1 keys
    (text, new_tokens, token_ids, seconds) plus the T7 split: prefill_s (a
    timed forward pass over the wrapped prompt under the same hook state),
    decode_s (generate total minus prefill, clamped), tok_per_s, and peak
    memory sampled via the torch.mps allocation API (max of post-prefill
    and post-decode current_allocated_memory; MPS exposes no peak counter).

    `cancel` (a threading.Event, optional) stops decoding early when set;
    the result then carries cancelled=True and a partial text.
    `wrapped` (optional) is a pre-wrapped prompt, used as-is.
    """
    max_new_tokens = min(max_new_tokens, MAX_NEW_TOKENS_CAP)
    torch.manual_seed(seed)
    try:
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except Exception:
        pass
    if wrapped is None:
        wrapped = wrap_prompt_notk(tokenizer, user_text, spec)
    device = str(next(model.parameters()).device)
    input_ids = tokenizer(wrapped, return_tensors="pt")["input_ids"].to(device)
    mem_samples: list[float] = []
    with torch.no_grad():
        _sync_device()
        t0 = time.perf_counter()
        model(input_ids)
        _sync_device()
        prefill_s = time.perf_counter() - t0
    mem_samples.append(mps_current_mib() or 0.0)
    _sync_device()
    t0 = time.perf_counter()
    with torch.no_grad():
        stop = StoppingCriteriaList([CancelCriteria(cancel)]) if cancel is not None else None
        out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                             stopping_criteria=stop)
    _sync_device()
    total_s = time.perf_counter() - t0
    mem_samples.append(mps_current_mib() or 0.0)
    decode_s = max(0.0, total_s - prefill_s)
    new_ids = out[0][input_ids.shape[1]:]
    n_new = len(new_ids)
    return {
        "text": tokenizer.decode(new_ids, skip_special_tokens=True),
        "new_tokens": n_new,
        "token_ids": new_ids.tolist(),
        "seconds": round(total_s, 1),
        "prefill_s": round(prefill_s, 2),
        "decode_s": round(decode_s, 2),
        "tok_per_s": round(n_new / total_s, 1) if total_s > 0 else 0.0,
        "peak_mib": round(max(mem_samples), 1),
        "driver_mib": mps_driver_mib(),
        "cancelled": bool(cancel is not None and cancel.is_set()),
    }


def generate_text_with_streamer(model, tokenizer, user_text: str, *,
                                max_new_tokens: int, seed: int,
                                spec: ModelSpec | None = None,
                                cancel=None, streamer=None,
                                wrapped: str | None = None):
    """generate_text that also feeds a streamer (same result keys).

    The returned dict is built from the generate output exactly like
    generate_text, so streamed runs carry identical token ids; the
    caller consumes streamer events concurrently in another thread.
    wrapped is a pre-wrapped prompt (chat history).
    """
    max_new_tokens = min(max_new_tokens, MAX_NEW_TOKENS_CAP)
    torch.manual_seed(seed)
    try:
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except Exception:
        pass
    if wrapped is None:
        wrapped = wrap_prompt_notk(tokenizer, user_text, spec)
    device = str(next(model.parameters()).device)
    input_ids = tokenizer(wrapped, return_tensors="pt")["input_ids"].to(device)
    mem_samples: list[float] = []
    with torch.no_grad():
        _sync_device()
        t0 = time.perf_counter()
        model(input_ids)
        _sync_device()
        prefill_s = time.perf_counter() - t0
    mem_samples.append(mps_current_mib() or 0.0)
    _sync_device()
    t0 = time.perf_counter()
    with torch.no_grad():
        stop = StoppingCriteriaList([CancelCriteria(cancel)]) if cancel is not None else None
        out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                             stopping_criteria=stop, streamer=streamer)
    _sync_device()
    total_s = time.perf_counter() - t0
    mem_samples.append(mps_current_mib() or 0.0)
    decode_s = max(0.0, total_s - prefill_s)
    new_ids = out[0][input_ids.shape[1]:]
    n_new = len(new_ids)
    return {
        "text": tokenizer.decode(new_ids, skip_special_tokens=True),
        "new_tokens": n_new,
        "token_ids": new_ids.tolist(),
        "seconds": round(total_s, 1),
        "prefill_s": round(prefill_s, 2),
        "decode_s": round(decode_s, 2),
        "tok_per_s": round(n_new / total_s, 1) if total_s > 0 else 0.0,
        "peak_mib": round(max(mem_samples), 1),
        "driver_mib": mps_driver_mib(),
        "cancelled": bool(cancel is not None and cancel.is_set()),
    }


# ---------------------------------------------------------------------------
# T2 (fallback ladder + recovery). Doc: D2 (ladder order + tests + preset
# binding), D8 (re-derivation procedure + verify-first rung checks). Rung 1
# passed at T1; the ladder below is the ordered map the doc requires.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LadderRung:
    n: int
    name: str
    model_id: str | None  # None = unresolved, rung does not count yet (D8)
    device: str
    test: str  # pass/fail test for this rung


# Rung-2 id verified to exist on the HF Hub at T3 (public, ungated).
RUNG2_MODEL_ID = "Qwen/Qwen3.5-2B"

LADDER: tuple[LadderRung, ...] = (
    LadderRung(1, "qwen3.5-4B fp16 on MPS eager", MODEL_ID, "mps",
               "loads + generates + hooks fire"),
    LadderRung(2, "same-family smaller", RUNG2_MODEL_ID, "mps",
               "loads + generates + hooks fire"),
    LadderRung(3, "GGUF-in-transformers", None, "mps",
               "hooks fire + flip judged (verify-first: hook path + weight "
               "layout + norm scale before this rung counts)"),
    LadderRung(4, "CPU run", MODEL_ID, "cpu",
               "completes; slow accepted"),
)

# D8 recovery: any fallback invalidates the old rows (preset binds to model
# id), so a new model needs re-derived neurons before the gate re-runs.
RECOVERY_PROCEDURE = (
    "contrast-prompt activation diff on the new model to re-derive "
    "(layer, neuron, dose) rows, then re-run the Day-0 spike gate, "
    "timeboxed half a day"
)

# D2 quantized-rung rule: any quantized rung needs all three before it counts.
QUANT_RUNG_RULE = "hook-compat check + recalibration + re-judged flip"


def check_preset_binding(preset: dict, model_id: str) -> None:
    """Presets bind to model id (D2): a fallback invalidates rows until re-derived."""
    want = preset.get("model_id")
    if want != model_id:
        raise RuntimeError(
            f"preset {preset.get('name', '?')!r} binds to {want!r}, not {model_id!r}; "
            f"re-derive rows first ({RECOVERY_PROCEDURE})"
        )


def hooks_fire_probe(model, tokenizer, spec: ModelSpec | None = None) -> int:
    """One forward pass with a canary pre-hook on L0 down_proj input.

    Returns the call count (>=1 means hooks fire). Uses resolve_down_proj,
    so a wrong path fails loudly here too (T5). No steering math involved.
    """
    if steering_hook_count() != 0:
        raise RuntimeError("probe needs steering hooks OFF (isolation)")
    calls = 0

    def _canary(_m, args):
        nonlocal calls
        calls += 1
        return args

    handle = resolve_down_proj(model, 0, spec).register_forward_pre_hook(_canary)
    try:
        device = str(next(model.parameters()).device)
        ids = tokenizer("Hello.", return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            model(ids)
    finally:
        handle.remove()
    return calls


def attempt_rung(rung: LadderRung, *, revision: str | None = None) -> dict:
    """Run one rung's pass/fail test. Returns {ok, rung, detail, seconds}."""
    t0 = time.perf_counter()
    if rung.model_id is None:
        return {"ok": False, "rung": rung.n,
                "detail": f"rung {rung.n} unresolved: no verified checkpoint; {rung.test}",
                "seconds": 0.0}
    try:
        tok = load_tokenizer(rung.model_id)
        model = load_model(rung.model_id, revision=revision, device=rung.device)
    except Exception as e:
        return {"ok": False, "rung": rung.n, "detail": f"load failed: {e}",
                "seconds": round(time.perf_counter() - t0, 1)}
    try:
        if rung.device == "cpu":
            ids = tok("Hello.", return_tensors="pt")["input_ids"]
            with torch.no_grad():
                model(ids)
            return {"ok": True, "rung": rung.n,
                    "detail": "CPU forward pass completed (slow accepted)",
                    "seconds": round(time.perf_counter() - t0, 1)}
        calls = hooks_fire_probe(model, tok)
        if calls < 1:
            return {"ok": False, "rung": rung.n, "detail": "hooks did not fire",
                    "seconds": round(time.perf_counter() - t0, 1)}
        with torch.no_grad():
            ids = tok("Hello.", return_tensors="pt")["input_ids"].to(rung.device)
            out = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS_PROBE, do_sample=False)
        new_tok = out.shape[1] - ids.shape[1]
        if new_tok < 1:
            return {"ok": False, "rung": rung.n, "detail": "generation empty",
                    "seconds": round(time.perf_counter() - t0, 1)}
        return {"ok": True, "rung": rung.n,
                "detail": f"loads + generates ({new_tok} tok) + hooks fire ({calls} calls)",
                "seconds": round(time.perf_counter() - t0, 1)}
    except Exception as e:
        return {"ok": False, "rung": rung.n, "detail": f"probe failed: {e}",
                "seconds": round(time.perf_counter() - t0, 1)}
    finally:
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def load_with_ladder(*, revision: str | None = None) -> tuple:
    """Try rungs top-first; stop at the first passing rung (D2).

    Returns (model, tokenizer, rung_n, attempts). Raises RuntimeError with the
    recovery procedure if every rung fails.
    """
    attempts = []
    for rung in LADDER:
        pin = revision if rung.n == 1 else None
        result = attempt_rung(rung, revision=pin)
        attempts.append(result)
        if result["ok"]:
            # O8: the pinned revision rides the final load, not just the probe.
            model = load_model(rung.model_id, revision=pin, device=rung.device)
            tok = load_tokenizer(rung.model_id)
            return model, tok, rung.n, attempts
    lines = "; ".join(f"rung {a['rung']}: {a['detail']}" for a in attempts)
    raise RuntimeError(f"ALL rungs failed ({lines}). Recovery: {RECOVERY_PROCEDURE}")


# ---------------------------------------------------------------------------
# T7 (runner budget + measurements + artifact template). Doc: D6 (max-tokens
# cap + per-run latency/memory), D9 (prefill-vs-decode latency per side,
# peak memory via the torch.mps allocation API, X artifact template fields,
# recipe-string fields). V1 writes the recipe string; v1.5 parses it (T10
# owns the grammar document; the field set below is the conformance target).
# ---------------------------------------------------------------------------

# D6 budget: no v1 run generates more than this, whatever the caller asks.
MAX_NEW_TOKENS_CAP = 256

ARTIFACT_VERSION = "v1"


def _sync_device() -> None:
    """Block until queued device work finishes, so timings are honest."""
    try:
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
    except Exception:
        pass


def mps_current_mib() -> float | None:
    """Current MPS allocated memory in MiB (None off-MPS)."""
    try:
        if torch.backends.mps.is_available():
            return round(torch.mps.current_allocated_memory() / 1024**2, 1)
    except Exception:
        pass
    return None


def mps_driver_mib() -> float | None:
    """Driver-side MPS allocation in MiB (None off-MPS)."""
    try:
        if torch.backends.mps.is_available():
            return round(torch.mps.driver_allocated_memory() / 1024**2, 1)
    except Exception:
        pass
    return None


def recipe_string(rows: list[RecipeRow], calib_id: str, *, max_new_tokens: int,
                  seed: int, calib_prompts: list[str],
                  model_id: str = MODEL_ID) -> str:
    """One-line recipe string (v1 writes, v1.5 parses).

    Fields (doc binding): model, rows, calibration id, template,
    max-tokens, decode settings, dtype/device, calibration prompts.
    """
    body = ";".join(f"L{r.layer}N{r.neuron}@{r.dose:g}" for r in rows)
    calib_hashes = ",".join(
        hashlib.sha256(p.encode()).hexdigest()[:8] for p in calib_prompts)
    tmpl = "chat+no-think" if MODEL_SPECS.get(model_id, QWEN_SPEC).thinking_kw else "chat"
    return (
        f"model-lobotomy/{ARTIFACT_VERSION} model={model_id} rows=[{body}] "
        f"calib={calib_id} template={tmpl} "
        f"max_tokens={max_new_tokens} decode=greedy,seed={seed} "
        f"dtype={DTYPE} device={DEVICE} calib_prompts=[{calib_hashes}]"
    )


def build_x_artifact(*, prompt: str, rows: list[RecipeRow], calib_id: str,
                     base: dict, steered: dict, rung_n: int,
                     max_new_tokens: int, seed: int,
                     calib_prompts: list[str],
                     model_id: str = MODEL_ID) -> str:
    """Paste-ready X artifact (v1 template): model, recipe, calibration id,
    per-side latency (prefill vs decode) + peak memory, prompt, both
    outputs, and the one-line recipe string."""
    row_lines = "\n".join(f"  L{r.layer} N{r.neuron} dose {r.dose:g}" for r in rows)

    def _side(d: dict) -> str:
        mem = f"{d.get('peak_mib', '?')} MiB peak (MPS current)" \
            if d.get("peak_mib") else "n/a"
        return (f"{d['seconds']}s total = {d.get('prefill_s', '?')}s prefill + "
                f"{d.get('decode_s', '?')}s decode "
                f"({d['new_tokens']} tok, {d.get('tok_per_s', '?')} tok/s; {mem})")

    return (
        f"model-lobotomy before/after\n"
        f"model: {model_id} (rung {rung_n})\n"
        f"recipe:\n{row_lines}\n"
        f"calibration: {calib_id}\n"
        f"latency baseline: {_side(base)}\n"
        f"latency steered: {_side(steered)}\n"
        f"prompt: {prompt}\n"
        f"--- baseline ---\n{base['text']}\n"
        f"--- steered ---\n{steered['text']}\n"
        f"recipe: {recipe_string(rows, calib_id, max_new_tokens=max_new_tokens, seed=seed, calib_prompts=calib_prompts, model_id=model_id)}"
    )


def build_post_text(*, rows: list[RecipeRow], base: dict, steered: dict,
                    calib_id: str, max_new_tokens: int, seed: int,
                    calib_prompts: list[str],
                    model_id: str = MODEL_ID) -> str:
    """Suggested X post text (<=280 chars). The human posts; code never posts.

    Shape: hook line + compact recipe line + before/after quotes. The full
    one-line recipe string lives in artifact.txt (it alone is ~220 chars,
    so it cannot fit here); the compact line is a human-readable subset,
    not a second grammar. Quotes shrink to fit; the recipe line never cuts.
    """
    short_model = model_short_name(model_id)
    doses = " ".join(f"L{r.layer}N{r.neuron}@{r.dose:+g}" for r in rows)
    recipe_line = (f"{doses} on {short_model} "
                   f"(calib {calib_id[:8]}, greedy s{seed}, <={max_new_tokens}tok)")

    def _quote(text: str, n: int) -> str:
        one_line = " ".join(text.split())
        return one_line if len(one_line) <= n else one_line[: max(0, n - 1)] + "…"

    head = f"steered {short_model} with {len(rows)} neurons:\n{recipe_line}\n"
    room = 280 - len(head) - len("before: '' after: ''")
    per = max(10, room // 2)
    post = (f"{head}before: '{_quote(base['text'], per)}' "
            f"after: '{_quote(steered['text'], per)}'")
    assert len(post) <= 280, f"post text {len(post)} chars exceeds 280"
    return post


def model_short_name(model_id: str = MODEL_ID) -> str:
    """Short model name for the post hook (org prefix + dots stripped)."""
    return model_id.split("/")[-1].replace(".", "").lower()


def save_artifact_dir(root: str | Path, *, prompt: str, rows: list[RecipeRow],
                      calib_id: str, base: dict, steered: dict, rung_n: int,
                      max_new_tokens: int, seed: int, calib_prompts: list[str],
                      divergence: int | None,
                      model_id: str = MODEL_ID) -> Path:
    """Write a versioned artifact dir: artifact.txt + post.txt + run.json.

    Dir name: <root>/<ARTIFACT_VERSION>-<UTC timestamp>. Returns the dir.
    """
    import datetime as _dt

    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    outdir = Path(root) / f"{ARTIFACT_VERSION}-{stamp}"
    outdir.mkdir(parents=True, exist_ok=False)
    (outdir / "artifact.txt").write_text(build_x_artifact(
        prompt=prompt, rows=rows, calib_id=calib_id, base=base,
        steered=steered, rung_n=rung_n, max_new_tokens=max_new_tokens,
        seed=seed, calib_prompts=calib_prompts, model_id=model_id))
    (outdir / "post.txt").write_text(build_post_text(
        rows=rows, base=base, steered=steered, calib_id=calib_id,
        max_new_tokens=max_new_tokens, seed=seed,
        calib_prompts=calib_prompts, model_id=model_id))
    (outdir / "run.json").write_text(json.dumps({
        "version": ARTIFACT_VERSION,
        "model_id": model_id,
        "rung": rung_n,
        "recipe": [{"layer": r.layer, "neuron": r.neuron, "dose": r.dose}
                   for r in rows],
        "calibration_id": calib_id,
        "prompt": prompt,
        "max_new_tokens": min(max_new_tokens, MAX_NEW_TOKENS_CAP),
        "seed": seed,
        "divergence_token": divergence,
        "baseline": {k: base[k] for k in (
            "new_tokens", "seconds", "prefill_s", "decode_s", "tok_per_s",
            "peak_mib", "driver_mib")},
        "steered": {k: steered[k] for k in (
            "new_tokens", "seconds", "prefill_s", "decode_s", "tok_per_s",
            "peak_mib", "driver_mib")},
    }, indent=2))
    return outdir
