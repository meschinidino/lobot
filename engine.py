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

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch

import steer

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


def _build_hooks(model, spec, edits: EditSet, medians: dict,
                 counter: list, reads: dict):
    """Build (mlp_hooks, out_hooks): {layer: hook_fn} per D13 order.

    counter is a one-item list bumped on every hook call; reads collects
    {layer: output copy} for Read edits. No registration here.
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
    out_hooks = {}
    for layer in set(abl) | set(vec) | want_read:
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
    from steer import RecipeRow
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
        """
        from huggingface_hub import HfApi
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

    def run(self, prompt: str, edits: EditSet, *, medians: dict,
            max_new_tokens: int, seed: int, cancel=None) -> dict:
        """One greedy generation under an EditSet (D13 order).

        Registers every hook in the registry, generates, collects
        read-outs and always unregisters. Returns the generate_text
        keys plus hook_calls and reads ({layer: tensor}).
        """
        if self.model is None:
            raise RuntimeError("no model loaded")
        spec = self.spec
        _validate_edits(self.model, spec, edits)
        counter: list = [0]
        reads: dict = {}
        mlp_hooks, out_hooks = _build_hooks(
            self.model, spec, edits, medians, counter, reads)
        t0 = time.perf_counter()
        for L, fn in mlp_hooks.items():
            self._hooks.append(("neuron", L, steer.resolve_down_proj(
                self.model, L, spec).register_forward_pre_hook(fn)))
        for L, fn in out_hooks.items():
            kinds = ("ablate" if any(a.layer == L for a in edits.ablate)
                     else "vector" if any(v.layer == L for v in edits.vector)
                     else "read")
            self._hooks.append((kinds, L, steer.resolve_block(
                self.model, L, spec).register_forward_hook(fn)))
        try:
            gen = steer.generate_text(
                self.model, self.tokenizer, prompt,
                max_new_tokens=max_new_tokens, seed=seed, spec=spec,
                cancel=cancel)
        finally:
            self.remove_all()
        gen["hook_calls"] = counter[0]
        gen["reads"] = reads
        gen["engine_seconds"] = round(time.perf_counter() - t0, 1)
        return gen


_ENGINE: Engine | None = None


def get_engine() -> Engine:
    """The process-wide Engine (single warm model)."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Engine()
    return _ENGINE
