"""lobot engine (D2, D13): one owner for model state and every hook.

Engine owns the warm model, tokenizer, run lock and a single registry of
every hook kind (neuron add, vector add, ablation, read). Generations go
through Engine.run(EditSet): per layer, neuron edits apply at the MLP
input, then one block-output hook applies ablation -> vector adds ->
read-outs (D13). Engine.clean() asserts zero hooks of any kind before
calibration and judging. Engine.capture (D6) is the one activation
tap for eval code; scripts that manage their own model use the
module-level capture() / apply_edits() twins, which share the same
hook builders (no second implementation).

Covered by tests/test_engine.py. steer.py keeps its primitives for
scripts and the model lane; product paths (app.py, server.py) run here.
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch

from model_lobotomy import steer

# Gated engine features. Engine.supports() is True for all of them;
# the llama backend (llama_backend.LlamaEngine) supports none and its
# routes say so via server._need_torch. _state().torch_only lists the
# gaps for the active backend.
FEATURES = ("dial", "capture", "reader", "sae", "create", "verify",
            "ablate")

# -- edits --------------------------------------------------------------------


@dataclass(frozen=True)
class NeuronEdit:
    """One neuron row: dose in neuron-scale units (x median/col-norm)."""
    layer: int
    neuron: int
    dose: float


@dataclass(frozen=True)
class VectorEdit:
    """Residual-stream add: out += dose * scale * unit(direction).

    direction is a unit tuple (immutable, testable); scale is the
    layer median residual norm (same dose unit as the spike).
    """
    layer: int
    direction: tuple
    scale: float
    dose: float


@dataclass(frozen=True)
class Ablation:
    """Directional ablation: out -= (out.u) u. D13: runs before adds."""
    layer: int
    direction: tuple


@dataclass(frozen=True)
class Read:
    """Read-out: capture the layer output after all edits (D13 last)."""
    layer: int


@dataclass(frozen=True)
class EditSet:
    """Immutable set of typed edits for one Engine.run call."""
    neuron: tuple = ()
    vector: tuple = ()
    ablate: tuple = ()
    read: tuple = ()

    @classmethod
    def from_rows(cls, rows) -> "EditSet":
        """Neuron rows (RecipeRow) -> EditSet (neuron site only)."""
        return cls(neuron=tuple(
            NeuronEdit(layer=r.layer, neuron=r.neuron, dose=r.dose)
            for r in rows))


EMPTY_EDITS = EditSet()


def overlap_warning(edits: EditSet) -> str | None:
    """Warn when a neuron knob pushes along an ablated direction (D13).

    A neuron edit (L, n) overlaps an ablation (L, u) when |u[n]| is more
    than 5% of the direction's largest component: the ablation partly
    removes what the knob adds. Returns a UI string, or None.
    """
    by_layer: dict[int, list] = {}
    for a in edits.ablate:
        by_layer.setdefault(a.layer, []).append(a)
    for e in edits.neuron:
        for a in by_layer.get(e.layer, []):
            u = a.direction
            if e.neuron < len(u) and abs(u[e.neuron]) > 0.05 * max(
                    (abs(x) for x in u), default=0.0):
                return (f"Row L{e.layer}N{e.neuron} pushes along a deleted "
                        f"direction; the ablation partly removes it.")
    return None


# -- hook builders (shared by Engine.run and apply_edits) ----------------------


def _neuron_deltas(model, spec, rows: list[NeuronEdit],
                   medians: dict) -> dict[int, dict[int, float]]:
    by_layer: dict[int, list[NeuronEdit]] = {}
    for r in rows:
        by_layer.setdefault(r.layer, []).append(r)
    out = {}
    for layer, lrows in by_layer.items():
        if layer not in medians:
            raise KeyError(f"no cached median for layer {layer}")
        med = float(medians[layer])
        out[layer] = {
            r.neuron: r.dose * med / steer.column_norm(model, layer, r.neuron, spec)
            for r in lrows}
    return out


def dial_value(h, unit, b0: float, b1: float) -> float:
    """Trait-dial projection: 0 = neutral mean, 1 = trait mean.

    h is one residual vector, unit the trait direction. A zero span
    reads 0.0 (uncalibrated) instead of exploding.
    """
    span = float(b1) - float(b0)
    if abs(span) < 1e-6:
        return 0.0
    p = float(torch.as_tensor(h, dtype=torch.float32)
              @ torch.as_tensor(unit, dtype=torch.float32))
    return (p - float(b0)) / span


def dial_for_index(tap: list, k: int, unit, b0: float, b1: float):
    """Dial value for generated token k, or None when the tap lags.

    tap[0] is the prefill entry (prompt's last token); decode step k
    appends tap[k + 1]. A short tap means extra forwards ran that the
    tap cannot map, so the token ships without a value.
    """
    j = k + 1
    if j >= len(tap):
        return None
    return dial_value(tap[j], unit, b0, b1)


def _build_hooks(model, spec, edits: EditSet, medians: dict,
                 counter: list, reads: dict, taps: dict | None = None):
    """Build (mlp_hooks, out_hooks): {layer: hook_fn} per D13 order.

    counter is a one-item list bumped on every hook call; reads collects
    {layer: output copy} for Read edits; taps (optional {layer: list})
    appends the last-position vector per forward for live dials.
    No registration here.
    """
    deltas = _neuron_deltas(model, spec, list(edits.neuron), medians)
    mlp_hooks = {}
    for layer, dd in deltas.items():
        def _pre(_m, args, *, _d=dict(dd)):
            counter[0] += 1
            x = args[0].clone()
            for n, d in _d.items():
                x[..., n] = x[..., n] + d
            return (x,) + args[1:]

        mlp_hooks[layer] = _pre
    abl: dict[int, list] = {}
    for a in edits.ablate:
        abl.setdefault(a.layer, []).append(
            torch.tensor(a.direction, dtype=torch.float32))
    vec: dict[int, list] = {}
    for v in edits.vector:
        u = torch.tensor(v.direction, dtype=torch.float32)
        vec.setdefault(v.layer, []).append((u, float(v.scale) * float(v.dose)))
    want_read = {r.layer for r in edits.read}
    tap_layers = set(taps or {})
    out_hooks = {}
    for layer in set(abl) | set(vec) | want_read | tap_layers:
        def _post(_m, _a, out, *, _L=layer):
            counter[0] += 1
            h = steer._block_hidden(out)
            dev, dt = h.device, h.dtype
            x = h.float()
            for u in abl.get(_L, []):
                uu = u.to(x.device)
                x = x - (x @ uu).unsqueeze(-1) * uu
            for uu, s in vec.get(_L, []):
                x = x + s * uu.to(x.device)
            if _L in want_read:
                reads[_L] = x.detach().cpu()
            if taps is not None and _L in taps:
                taps[_L].append(x[0, -1, :].detach().cpu())
            x = x.to(dt)
            if isinstance(out, torch.Tensor):
                return x
            return (x,) + tuple(out[1:])

        out_hooks[layer] = _post
    return mlp_hooks, out_hooks


# -- capture (D6) -----------------------------------------------------------------


def capture(model, tok, spec, texts: list[str], layers: list[int],
            point: str = "mlp_in", reduce: str = "mean_1") -> dict:
    """One activation tap for eval code. No model state, no registry.

    point: "mlp_in" (down_proj input, intermediate width, dim-asserted)
        or "block_out" (decoder block output, residual width).
    reduce: "mean_1" (mean over tokens 1: per text), "last" (last
        token per text), or "none" (raw token rows per text).
    Returns {layer: [P, W] tensor} for mean_1/last, {layer: list of
    [T, W]} for none. Guards: non-empty texts, 2+ tokens per prompt.
    """
    if point not in ("mlp_in", "block_out"):
        raise ValueError(f"unknown capture point {point!r}")
    if reduce not in ("mean_1", "last", "none"):
        raise ValueError(f"unknown capture reduce {reduce!r}")
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        raise ValueError("capture needs at least one non-empty text")
    per_layer: dict[int, list] = {L: [] for L in layers}
    grabbed: dict[int, list] = {L: [] for L in layers}
    handles = []
    for L in layers:
        if point == "mlp_in":
            mod = steer.resolve_down_proj(model, L, spec)

            def _pre(_m, args, *, _L=L):
                x = args[0].detach().float().cpu()
                if x.shape[-1] != spec.intermediate:
                    raise RuntimeError(
                        f"capture width mismatch at L{_L}: {x.shape[-1]} != "
                        f"intermediate {spec.intermediate}")
                grabbed[_L].append(x)

            handles.append(mod.register_forward_pre_hook(_pre))
        else:
            block = steer.resolve_block(model, L, spec)

            def _post(_m, _a, out, *, _L=L):
                grabbed[_L].append(
                    steer._block_hidden(out).detach().float().cpu())

            handles.append(block.register_forward_hook(_post))
    dev = "cpu"
    params = getattr(model, "parameters", None)
    if params is not None:
        first = next(params(), None)
        if first is not None:
            dev = first.device
    try:
        with torch.no_grad():
            for t in texts:
                wrapped = steer.wrap_prompt_notk(tok, t, spec)
                ids = tok(wrapped, return_tensors="pt")["input_ids"]
                if ids.numel() < 2:
                    raise ValueError(f"capture text too short (<2 tokens): {t!r}")
                for L in layers:
                    grabbed[L].clear()
                model(ids.to(dev))
                for L in layers:
                    h = grabbed[L][-1][0]
                    if reduce == "mean_1":
                        per_layer[L].append(h[1:].mean(dim=0))
                    elif reduce == "last":
                        per_layer[L].append(h[-1])
                    else:
                        per_layer[L].append(h[1:])
    finally:
        for h in handles:
            h.remove()
    if reduce == "none":
        return per_layer
    return {L: torch.stack(per_layer[L]) for L in layers}


@contextmanager
def apply_edits(model, tok, spec, edits: EditSet, medians: dict):
    """Script twin of Engine.run's hook path (own local handles).

    For scripts that manage their own model (spike, derive): asserts no
    steer-level hooks are active, registers the EditSet under D13 order,
    yields a reads dict, and always removes its handles. Product code
    uses Engine.run instead.
    """
    if steer.steering_hook_count() != 0:
        raise RuntimeError("apply_edits needs steer hooks OFF (isolation)")
    _validate_edits(model, spec, edits)
    counter: list = [0]
    reads: dict = {}
    mlp_hooks, out_hooks = _build_hooks(model, spec, edits, medians, counter, reads)
    handles = []
    for L, fn in mlp_hooks.items():
        handles.append(steer.resolve_down_proj(model, L, spec)
                       .register_forward_pre_hook(fn))
    for L, fn in out_hooks.items():
        handles.append(steer.resolve_block(model, L, spec)
                       .register_forward_hook(fn))
    try:
        yield reads
    finally:
        for h in handles:
            h.remove()


def _validate_edits(model, spec, edits: EditSet) -> None:
    from model_lobotomy.steer import RecipeRow
    steer.validate_recipe(
        [RecipeRow(r.layer, r.neuron, r.dose) for r in edits.neuron], spec)
    for v in list(edits.vector) + list(edits.ablate):
        if not 0 <= v.layer < spec.layers:
            raise ValueError(f"layer {v.layer} out of bounds [0,{spec.layers})")
    for r in edits.read:
        if not 0 <= r.layer < spec.layers:
            raise ValueError(f"layer {r.layer} out of bounds [0,{spec.layers})")
    steer.assert_text_dims(model, spec)


# -- Engine -----------------------------------------------------------------------


class Engine:
    """Owns the warm model, lock and every hook (D2). One in memory."""

    backend = "torch"

    def supports(self, feature: str) -> bool:
        """True for every FEATURES member (the torch backend is full)."""
        return feature in FEATURES

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_id: str | None = None
        self.revision: str | None = None
        self.rung: int | None = None
        self.lock = threading.Lock()
        self._hooks: list = []  # (kind, layer, handle); every hook kind

    @property
    def spec(self):
        if self.model_id is None:
            raise RuntimeError("no model loaded")
        return steer.get_spec(self.model_id)

    @property
    def warm(self) -> bool:
        return self.model is not None

    def load(self, model_id: str, revision: str | None = None) -> int:
        """Load (and pin) a model; Qwen rides the fallback ladder.

        Resolves the revision first and passes it to the loaders (O8):
        the same revision loads every time. Returns the rung number.
        Takes this process's GPU slot first (gpu_lock; GpuBusy when
        another model job holds every slot).
        """
        from huggingface_hub import HfApi
        from model_lobotomy import gpu_lock
        gpu_lock.acquire(f"torch {model_id}")
        if self.model is not None:
            self.unload()
        sha = revision or HfApi().model_info(model_id).sha
        if model_id == steer.MODEL_ID:
            model, tok, rung, attempts = steer.load_with_ladder(revision=sha)
            for a in attempts:
                print(f"[engine] rung {a['rung']}: ok={a['ok']} "
                      f"{a['detail']}", flush=True)
            if rung != 1:
                raise RuntimeError(
                    f"ladder stopped at rung {rung}, not rung 1; preset rows "
                    f"invalid until re-derived ({steer.RECOVERY_PROCEDURE})")
        else:
            model = steer.load_model(model_id, revision=sha)
            tok = steer.load_tokenizer(model_id)
            rung = 1
            spec = steer.get_spec(model_id)
            print(f"[engine] {model_id} loaded direct "
                  f"(dims {spec.layers}/{spec.hidden}/{spec.intermediate})",
                  flush=True)
        self.model, self.tokenizer = model, tok
        self.model_id, self.revision, self.rung = model_id, sha, rung
        return rung

    def unload(self) -> None:
        self.remove_all()
        self.model, self.tokenizer = None, None
        self.model_id, self.revision, self.rung = None, None, None
        import gc
        gc.collect()
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    def fingerprint(self) -> dict:
        """Execution fingerprint binding runs to the exact stack (O8)."""
        if self.model is None:
            raise RuntimeError("no model loaded")
        return dict(steer.calibration_cache_key(
            self.model_id, self.revision, self.tokenizer,
            str(steer.DTYPE), str(next(self.model.parameters()).device)))

    # -- registry --

    def hook_count(self, kind: str | None = None) -> int:
        if kind is None:
            return len(self._hooks)
        return sum(1 for k, _, _ in self._hooks if k == kind)

    def remove_all(self) -> None:
        while self._hooks:
            _, _, h = self._hooks.pop()
            h.remove()

    @contextmanager
    def clean(self):
        """Assert zero hooks of any kind (before calibration/judging)."""
        if self._hooks or steer.steering_hook_count() != 0:
            kinds = sorted({k for k, _, _ in self._hooks})
            raise RuntimeError(
                f"hooks active during clean section: engine={kinds} "
                f"steer={steer.steering_hook_count()}")
        yield

    # -- capture + run --

    def capture(self, texts: list[str], layers: list[int],
                point: str = "mlp_in", reduce: str = "mean_1") -> dict:
        """D6 helper bound to the warm model. Registry must be empty."""
        if self.model is None:
            raise RuntimeError("no model loaded")
        with self.clean():
            return capture(self.model, self.tokenizer, self.spec,
                           texts, layers, point, reduce)

    def capture_with_edits(self, texts: list[str], layers: list[int],
                           edits: EditSet, medians: dict,
                           point: str = "mlp_in",
                           reduce: str = "none") -> dict:
        """capture() with an EditSet active (click-a-word reads the
        steered activations, not the clean ones). Registers, captures,
        always unregisters."""
        if self.model is None:
            raise RuntimeError("no model loaded")
        spec = self.spec
        _validate_edits(self.model, spec, edits)
        counter: list = [0]
        reads: dict = {}
        self._register_hooks(edits, medians, counter, reads, spec)
        try:
            return capture(self.model, self.tokenizer, spec,
                           texts, layers, point, reduce)
        finally:
            self.remove_all()

    def prefill(self, user_text: str, edits: EditSet | None = None, *,
                medians: dict | None = None):
        """One forward pass; return last-position logits (1D vocab tensor).

        Uses the same chat wrapper as generate_text, so the first-token
        distribution matches real generation. With edits, registers
        them under D13 order (medians required for neuron rows) and
        always removes them. The guard (D8) reads first-token refusal
        probability from these logits.
        """
        if self.model is None:
            raise RuntimeError("no model loaded")
        spec = self.spec
        wrapped = steer.wrap_prompt_notk(self.tokenizer, user_text, spec)
        tok_out = self.tokenizer(wrapped, return_tensors="pt")
        ids = (tok_out["input_ids"] if isinstance(tok_out, dict)
               else tok_out.input_ids)
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = "cpu"
        ids = ids.to(device)
        if edits is None:
            with torch.no_grad():
                out = self.model(ids)
            return out.logits[0, -1, :].detach()
        if medians is None:
            raise ValueError("prefill with edits needs medians")
        _validate_edits(self.model, spec, edits)
        counter: list = [0]
        reads: dict = {}
        mlp_hooks, out_hooks = _build_hooks(
            self.model, spec, edits, medians, counter, reads)
        for L, fn in mlp_hooks.items():
            self._hooks.append(("neuron", L, steer.resolve_down_proj(
                self.model, L, spec).register_forward_pre_hook(fn)))
        for L, fn in out_hooks.items():
            self._hooks.append(("guard", L, steer.resolve_block(
                self.model, L, spec).register_forward_hook(fn)))
        try:
            with torch.no_grad():
                out = self.model(ids)
        finally:
            self.remove_all()
        return out.logits[0, -1, :].detach()

    def _register_hooks(self, edits: EditSet, medians: dict,
                        counter: list, reads: dict, spec,
                        taps: dict | None = None) -> None:
        """Shared hook path for run/run_stream (D13 order, registered)."""
        mlp_hooks, out_hooks = _build_hooks(
            self.model, spec, edits, medians, counter, reads, taps)
        for L, fn in mlp_hooks.items():
            self._hooks.append(("neuron", L, steer.resolve_down_proj(
                self.model, L, spec).register_forward_pre_hook(fn)))
        for L, fn in out_hooks.items():
            kinds = ("ablate" if any(a.layer == L for a in edits.ablate)
                     else "vector" if any(v.layer == L for v in edits.vector)
                     else "read")
            self._hooks.append((kinds, L, steer.resolve_block(
                self.model, L, spec).register_forward_hook(fn)))

    def run(self, prompt: str, edits: EditSet, *, medians: dict,
            max_new_tokens: int, seed: int, cancel=None,
            wrapped: str | None = None) -> dict:
        """One greedy generation under an EditSet (D13 order).

        Registers every hook in the registry, generates, collects
        read-outs and always unregisters. Returns the generate_text
        keys plus hook_calls and reads ({layer: tensor}). wrapped is
        a pre-wrapped prompt (chat history); otherwise the prompt is
        wrapped as one user turn.
        """
        if self.model is None:
            raise RuntimeError("no model loaded")
        spec = self.spec
        _validate_edits(self.model, spec, edits)
        counter: list = [0]
        reads: dict = {}
        t0 = time.perf_counter()
        self._register_hooks(edits, medians, counter, reads, spec)
        try:
            gen = steer.generate_text(
                self.model, self.tokenizer, prompt,
                max_new_tokens=max_new_tokens, seed=seed, spec=spec,
                cancel=cancel, wrapped=wrapped)
        finally:
            self.remove_all()
        gen["hook_calls"] = counter[0]
        gen["reads"] = reads
        gen["engine_seconds"] = round(time.perf_counter() - t0, 1)
        return gen

    def run_stream(self, prompt: str, edits: EditSet, *, medians: dict,
                   max_new_tokens: int, seed: int, cancel=None,
                   dial: dict | None = None,
                   dials: list[dict] | None = None,
                   wrapped: str | None = None):
        """run() as token events, then one done event (filmstrip live).

        Yields {"type": "token", "text": delta} per generated token
        (cumulative decode, so concatenated deltas equal the text) and
        finally {"type": "done", "result": <run() dict>}. Generation
        runs in a worker thread; closing the generator (disconnect)
        sets cancel, joins the worker and removes the hooks. dial is
        one {"trait", "layer", "unit", "b0", "b1"} read-dial spec;
        dials is a list of them (dial stays as the one-spec
        shorthand; both may combine). Token events then carry
        {"dial": {trait: value}} per spec whose tap kept up.
        wrapped is a pre-wrapped prompt (chat history).
        """
        if self.model is None:
            raise RuntimeError("no model loaded")
        spec = self.spec
        _validate_edits(self.model, spec, edits)
        if cancel is None:
            cancel = threading.Event()
        specs = list(dials or [])
        if dial is not None:
            specs.append(dial)
        streamer = steer.RowStreamer()
        counter: list = [0]
        reads: dict = {}
        taps: dict | None = None
        tap_lists: dict[int, list] = {}
        if specs:
            for entry in specs:
                tap_lists.setdefault(int(entry["layer"]), [])
            taps = tap_lists
        t0 = time.perf_counter()
        self._register_hooks(edits, medians, counter, reads, spec, taps)
        result: dict = {}

        def _target():
            try:
                result["gen"] = steer.generate_text_with_streamer(
                    self.model, self.tokenizer, prompt,
                    max_new_tokens=max_new_tokens, seed=seed, spec=spec,
                    cancel=cancel, streamer=streamer, wrapped=wrapped)
            except Exception as e:  # noqa: BLE001 - re-raised after join
                result["error"] = e
            finally:
                streamer.end()

        worker = threading.Thread(target=_target, daemon=True)
        worker.start()
        try:
            ids: list[int] = []
            prev = ""
            while True:
                row, tok = streamer.queue.get()
                if row is None:
                    break
                ids.append(tok)
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
                delta = text[len(prev):]
                prev = text
                if delta:
                    ev = {"type": "token", "text": delta}
                    if specs:
                        per = {}
                        for entry in specs:
                            v = dial_for_index(
                                tap_lists[int(entry["layer"])],
                                len(ids) - 1, entry["unit"],
                                entry["b0"], entry["b1"])
                            if v is not None:
                                per[entry["trait"]] = v
                        if per:
                            ev["dial"] = per
                    yield ev
        except GeneratorExit:
            cancel.set()
            worker.join(timeout=120)
            self.remove_all()
            raise
        worker.join()
        self.remove_all()
        if "error" in result:
            raise result["error"]
        gen = result["gen"]
        gen["hook_calls"] = counter[0]
        gen["reads"] = reads
        gen["engine_seconds"] = round(time.perf_counter() - t0, 1)
        yield {"type": "done", "result": gen}


    def run_batched(self, prompt: str, edit_sets: list[EditSet], *,
                    medians: dict, max_new_tokens: int, seed: int,
                    cancel=None, fallback: bool = True) -> dict:
        """One greedy batch: same prompt, one EditSet per row (E6/D3).

        All rows must touch the same layers (filmstrip varies doses, not
        sites); ablations must match across rows. One pre-hook per MLP
        layer adds per-row deltas; one output hook per block layer
        applies per-row ablate -> vector add and stacks read-outs.
        On any failure (or non-finite lm_head output) with fallback on,
        reruns as N single Engine.run calls and reports fallback_used.
        Returns {"rows": [generate_text dicts], "fallback_used",
        "hook_calls", "reads"}.
        """
        if self.model is None:
            raise RuntimeError("no model loaded")
        spec = self.spec
        self._check_batched(edit_sets, spec)
        if fallback:
            try:
                return self._run_batched_inner(
                    prompt, edit_sets, spec, medians, max_new_tokens,
                    seed, cancel, fallback_used=False)
            except Exception:
                self.remove_all()
        else:
            return self._run_batched_inner(
                prompt, edit_sets, spec, medians, max_new_tokens,
                seed, cancel, fallback_used=False)
        rows = [self.run(prompt, es, medians=medians,
                         max_new_tokens=max_new_tokens, seed=seed,
                         cancel=cancel) for es in edit_sets]
        return {"rows": rows, "fallback_used": True,
                "hook_calls": sum(r["hook_calls"] for r in rows),
                "reads": {}}

    def run_batched_stream(self, prompt: str, edit_sets: list[EditSet], *,
                           medians: dict, max_new_tokens: int, seed: int,
                           cancel=None, fallback: bool = True):
        """run_batched() as per-column token events, then one done event.

        Yields {"type": "token", "col": b, "text": delta} as columns
        generate (cumulative decode per column) and finally
        {"type": "done", "result": <run_batched() dict>}. On fallback
        each column arrives as one full-text token event instead;
        result["fallback_used"] tells them apart. Closing the
        generator (disconnect) sets cancel, joins the worker and
        removes the hooks.
        """
        if self.model is None:
            raise RuntimeError("no model loaded")
        spec = self.spec
        self._check_batched(edit_sets, spec)
        if cancel is None:
            cancel = threading.Event()
        B = len(edit_sets)
        streamer = steer.RowStreamer()
        result: dict = {}

        def _target():
            try:
                result["out"] = self._run_batched_inner(
                    prompt, edit_sets, spec, medians, max_new_tokens,
                    seed, cancel, fallback_used=False, streamer=streamer)
            except Exception as e:  # noqa: BLE001 - fallback or re-raise
                result["error"] = e
            finally:
                streamer.end()

        worker = threading.Thread(target=_target, daemon=True)
        worker.start()
        try:
            ids = [[] for _ in range(B)]
            prev = ["" for _ in range(B)]
            while True:
                row, tok = streamer.queue.get()
                if row is None:
                    break
                ids[row].append(tok)
                text = self.tokenizer.decode(ids[row],
                                             skip_special_tokens=True)
                delta = text[len(prev[row]):]
                prev[row] = text
                if delta:
                    yield {"type": "token", "col": row, "text": delta}
        except GeneratorExit:
            cancel.set()
            worker.join(timeout=120)
            self.remove_all()
            raise
        worker.join()
        if "error" in result and fallback:
            self.remove_all()
            rows = [self.run(prompt, es, medians=medians,
                             max_new_tokens=max_new_tokens, seed=seed,
                             cancel=cancel) for es in edit_sets]
            for b, r in enumerate(rows):
                yield {"type": "token", "col": b, "text": r["text"],
                       "fallback": True}
            yield {"type": "done", "result": {
                "rows": rows, "fallback_used": True,
                "hook_calls": sum(r["hook_calls"] for r in rows),
                "reads": {}}}
            return
        if "error" in result:
            raise result["error"]
        yield {"type": "done", "result": result["out"]}

    @staticmethod
    def _touched_layers(es: EditSet) -> set:
        return ({e.layer for e in es.neuron} | {e.layer for e in es.vector}
                | {e.layer for e in es.ablate} | {e.layer for e in es.read})

    @staticmethod
    def _check_ablations_match(edit_sets: list[EditSet]) -> None:
        first = sorted((a.layer, a.direction) for a in edit_sets[0].ablate)
        for es in edit_sets[1:]:
            if sorted((a.layer, a.direction) for a in es.ablate) != first:
                raise ValueError("batched rows must share ablations")

    def _check_batched(self, edit_sets: list[EditSet], spec) -> None:
        """Shared restrictions for run_batched/run_batched_stream."""
        if not edit_sets:
            raise ValueError("run_batched needs at least one EditSet")
        for es in edit_sets:
            _validate_edits(self.model, spec, es)
        layer_sets = [self._touched_layers(es) for es in edit_sets]
        if any(s != layer_sets[0] for s in layer_sets):
            raise ValueError("batched rows must touch the same layers")
        self._check_ablations_match(edit_sets)

    def _run_batched_inner(self, prompt, edit_sets, spec, medians,
                           max_new_tokens, seed, cancel, fallback_used,
                           streamer=None):
        from transformers import StoppingCriteriaList
        B = len(edit_sets)
        # Per-row deltas: neuron {L: {n: [B]}}, vector {L: [(u, [B scales])]}.
        neuron: dict[int, dict[int, list]] = {}
        for b, es in enumerate(edit_sets):
            for r in es.neuron:
                if r.layer not in medians:
                    raise KeyError(f"no cached median for layer {r.layer}")
                med = float(medians[r.layer])
                d = (r.dose * med / steer.column_norm(
                    self.model, r.layer, r.neuron, spec))
                neuron.setdefault(r.layer, {}).setdefault(r.neuron, [0.0] * B)[b] = d
        vector: dict[int, list] = {}
        for b, es in enumerate(edit_sets):
            for v in es.vector:
                key = (v.layer, v.direction)
                slot = vector.setdefault(key, None)
                if slot is None:
                    slot = vector[key] = [torch.tensor(
                        v.direction, dtype=torch.float32), [0.0] * B]
                slot[1][b] = float(v.scale) * float(v.dose)
        abl = sorted({(a.layer, a.direction) for a in edit_sets[0].ablate})
        want_read = {r.layer for es in edit_sets for r in es.read}
        counter, reads, finite = [0], {}, [True]

        def _pre(_m, args, *, _dd):
            counter[0] += 1
            x = args[0].clone()
            for n, dd in _dd.items():
                x[..., n] = x[..., n] + torch.tensor(
                    dd, dtype=x.dtype, device=x.device).view(-1, *([1] * (x.dim() - 2)))
            return (x,) + args[1:]

        def _post(_m, _a, out, *, _L):
            counter[0] += 1
            h = steer._block_hidden(out)
            dev, dt = h.device, h.dtype
            x = h.float()
            for layer, u in abl:
                if layer != _L:
                    continue
                uu = torch.tensor(u, dtype=torch.float32, device=dev)
                x = x - (x @ uu).unsqueeze(-1) * uu
            for (layer, _u), (uu, ss) in vector.items():
                if layer != _L:
                    continue
                x = x + torch.tensor(ss, dtype=torch.float32,
                                     device=dev).view(
                    -1, *([1] * (x.dim() - 1))) * uu.to(dev)
            if _L in want_read:
                reads[_L] = x.detach().cpu()
            return x.to(dt) if isinstance(out, torch.Tensor) else (
                x.to(dt),) + tuple(out[1:])

        for L, dd in neuron.items():
            self._hooks.append(("neuron", L, steer.resolve_down_proj(
                self.model, L, spec).register_forward_pre_hook(
                    lambda m, a, *, _dd=dd: _pre(m, a, _dd=_dd))))
        out_layers = ({L for L, _ in abl} | {L for L, _ in vector}
                      | want_read)
        for L in out_layers:
            self._hooks.append(("batched", L, steer.resolve_block(
                self.model, L, spec).register_forward_hook(
                    lambda m, a, o, *, _L=L: _post(m, a, o, _L=_L))))
        head = self._lm_head()
        if head is not None:
            def _finite(_m, _a, out, *, _f=finite):
                t = out[0] if isinstance(out, tuple) else out
                if isinstance(t, torch.Tensor) and not bool(torch.isfinite(t).all()):
                    _f[0] = False

            self._hooks.append(("probe", -1, head.register_forward_hook(_finite)))
        try:
            max_new_tokens = min(max_new_tokens, steer.MAX_NEW_TOKENS_CAP)
            torch.manual_seed(seed)
            wrapped = steer.wrap_prompt_notk(self.tokenizer, prompt, spec)
            device = str(next(self.model.parameters()).device)
            one = self.tokenizer(wrapped, return_tensors="pt")["input_ids"]
            input_ids = one.repeat(B, 1).to(device)
            t_all = time.perf_counter()
            with torch.no_grad():
                stop = None
                if cancel is not None:
                    stop = StoppingCriteriaList([steer.CancelCriteria(cancel)])
                out = self.model.generate(
                    input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=stop, streamer=streamer)
            total_s = time.perf_counter() - t_all
            if not finite[0]:
                raise FloatingPointError("non-finite lm_head output")
            rows = []
            for b in range(B):
                new_ids = out[b][one.shape[1]:]
                rows.append({
                    "text": self.tokenizer.decode(new_ids,
                                                  skip_special_tokens=True),
                    "new_tokens": len(new_ids),
                    "token_ids": new_ids.tolist(),
                    "seconds": round(total_s, 1),
                    "cancelled": bool(cancel is not None and cancel.is_set()),
                })
        finally:
            self.remove_all()
        return {"rows": rows, "fallback_used": fallback_used,
                "hook_calls": counter[0], "reads": reads}

    def _lm_head(self):
        for obj in (self.model, getattr(self.model, "model", None),
                    getattr(getattr(self.model, "model", None),
                            "language_model", None)):
            head = getattr(obj, "lm_head", None)
            if head is not None:
                return head
        return None


_ENGINE: Engine | None = None


def get_engine() -> Engine:
    """The process-wide engine (single warm model).

    Reads LOBOT_BACKEND once ("torch" default, "llama" for the
    llama.cpp daemon backend) and caches the choice. Any other
    value raises ValueError.
    """
    global _ENGINE
    if _ENGINE is None:
        which = os.environ.get("LOBOT_BACKEND", "torch")
        if which == "torch":
            _ENGINE = Engine()
        elif which == "llama":
            from model_lobotomy import llama_backend
            _ENGINE = llama_backend.LlamaEngine()
        else:
            raise ValueError(
                f"unknown LOBOT_BACKEND {which!r}; want torch or llama")
    return _ENGINE
