"""lobot engine: one owner for every hook on the model.

D2: one registry holds every hook kind (neuron add, vector add, ablation,
read, capture). Engine.clean() asserts it is empty before calibration and
judging. D13: two hook sites per layer, applied in one fixed order:

  1. MLP input (down_proj pre-hook): neuron adds, the recipe rows. Same
     math as steer.register_steering_hooks, so recorded recipes replay
     exactly.
  2. Block output (forward hook): ablate -> vector add -> read.

So a run applies neuron add -> ablate -> vector add -> read in every layer.
D6: Engine.capture() is the one activation-capture helper for the spike and
derive scripts; steer.calibrate_layer_medians keeps its own tested loop.

Model state (weights, tokenizer, run lock) still lives in app.py.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch

import steer
from steer import RecipeRow


@dataclass(frozen=True, eq=False)
class VectorEdit:
    """Add `vector` to the block output of `layer` at every position."""
    layer: int
    vector: torch.Tensor


@dataclass(frozen=True, eq=False)
class Ablation:
    """Remove the component along `direction` from the block output of `layer`."""
    layer: int
    direction: torch.Tensor


@dataclass(frozen=True, eq=False)
class Read:
    """Append the per-token projection of the block output on `direction` to `sink`.

    Each forward pass appends one 1-D CPU tensor (one value per position).
    """
    layer: int
    direction: torch.Tensor
    sink: list


POINTS = ("mlp_in", "block_in", "block_out")
REDUCES = ("mean", "last", "norms")


def _unit(v: torch.Tensor) -> torch.Tensor:
    n = v.float().norm()
    if n == 0:
        raise ValueError("a direction must be non-zero")
    return v.float() / n


class Engine:
    """Owns the hook registry: every hook on the model goes through here."""

    def __init__(self) -> None:
        self._hooks: list[tuple[frozenset, int, object]] = []  # (kinds, layer, handle)
        self.block_calls = 0

    # -- registry -----------------------------------------------------------

    def hook_count(self, kind: str | None = None) -> int:
        """Active hooks of `kind`, or of every kind. Neuron hooks live in steer's list."""
        neuron = steer.steering_hook_count()
        if kind == "neuron":
            return neuron
        mine = sum(1 for kinds, _, _ in self._hooks if kind is None or kind in kinds)
        return mine + (neuron if kind is None else 0)

    @contextmanager
    def clean(self):
        """Run a block with zero hooks of any kind; raise if one is active or leaks."""
        if self.hook_count():
            raise RuntimeError(f"{self.hook_count()} hooks are active; calibration, capture "
                               "and judging need none (D2)")
        yield
        if self.hook_count():
            raise RuntimeError("a hook leaked out of a clean block (D2)")

    def _register(self, kinds, layer, handle) -> None:
        self._hooks.append((frozenset(kinds), layer, handle))

    def _remove(self, handle) -> None:
        handle.remove()
        self._hooks = [h for h in self._hooks if h[2] is not handle]

    # -- edits ----------------------------------------------------------------

    @contextmanager
    def apply(self, model, edits, *, medians: dict | None = None, spec=None):
        """Hold `edits` on the model for the block; remove every hook after, even on error.

        edits: RecipeRow (neuron add), VectorEdit, Ablation and Read, in any order;
        the order inside a layer is fixed (module docstring). Neuron rows need
        `medians` (calibration) as in steer.register_steering_hooks.
        """
        spec = spec or steer.QWEN_SPEC
        edits = list(edits)
        unknown = [e for e in edits if not isinstance(e, (RecipeRow, VectorEdit, Ablation, Read))]
        if unknown:
            raise TypeError(f"unknown edit type {type(unknown[0]).__name__}")
        rows = [e for e in edits if isinstance(e, RecipeRow)]
        by_layer: dict[int, dict[str, list]] = {}
        for e in edits:
            if isinstance(e, RecipeRow):
                continue
            if not 0 <= e.layer < spec.layers:
                raise ValueError(f"layer {e.layer} out of bounds [0,{spec.layers})")
            slot = by_layer.setdefault(e.layer, {"ablation": [], "vector": [], "read": []})
            if isinstance(e, Ablation):
                slot["ablation"].append(_unit(e.direction))
            elif isinstance(e, VectorEdit):
                slot["vector"].append(e.vector)
            else:
                slot["read"].append((_unit(e.direction), e.sink))
        handles = []
        try:
            if rows:
                if medians is None:
                    raise ValueError("neuron rows need calibration medians")
                steer.register_steering_hooks(model, rows, medians, mode="add", spec=spec)
            for layer, slot in by_layer.items():
                hook = self._block_hook(slot)
                h = steer.resolve_block(model, layer, spec).register_forward_hook(hook)
                handles.append(h)
                self._register([k for k, v in slot.items() if v], layer, h)
            yield
        finally:
            for h in handles:
                self._remove(h)
            if rows:
                steer.remove_steering_hooks()

    def _block_hook(self, slot):
        abl, vec, reads = slot["ablation"], slot["vector"], slot["read"]

        def _hook(_m, _args, out):
            self.block_calls += 1
            h = steer._block_hidden(out)
            x = h
            for d in abl:  # 1. ablate
                d = d.to(device=x.device, dtype=x.dtype)
                x = x - (x @ d).unsqueeze(-1) * d
            for v in vec:  # 2. vector add
                x = x + v.to(device=x.device, dtype=x.dtype)
            for d, sink in reads:  # 3. read what the next layer will see
                sink.append((x[0].float() @ d.to(x.device)).detach().cpu())
            if x is h:
                return None
            return x if isinstance(out, torch.Tensor) else (x,) + tuple(out[1:])

        return _hook

    # -- capture (D6) ---------------------------------------------------------

    def capture(self, model, tok, spec, texts: list[str], layers, *, point: str, reduce: str):
        """Activations at `point` per layer and text, reduced per text.

        point:  "mlp_in" (down_proj input, spec.intermediate wide),
                "block_in" or "block_out" (residual stream, spec.hidden wide).
        reduce: "mean" (mean over tokens 1:, BOS skipped), "last" (last token),
                "norms" (per-token L2 norms over tokens 1:).
        Returns {layer: [tensor per text]} on CPU, float32. Runs with no other
        hooks active (clean) and fails loudly on a width that does not match
        the point.
        """
        if point not in POINTS or reduce not in REDUCES:
            raise ValueError(f"point must be one of {POINTS}, reduce one of {REDUCES}")
        width = spec.intermediate if point == "mlp_in" else spec.hidden
        layers = list(layers)
        grabbed: dict[int, torch.Tensor] = {}
        out: dict[int, list] = {L: [] for L in layers}
        handles = []

        def _keep(L, x):
            grabbed[L] = x.detach().float().cpu()

        with self.clean():
            try:
                for L in layers:
                    if point == "mlp_in":
                        mod = steer.resolve_down_proj(model, L, spec)
                        h = mod.register_forward_pre_hook(lambda _m, a, *, _L=L: _keep(_L, a[0]))
                    elif point == "block_in":
                        mod = steer.resolve_block(model, L, spec)
                        h = mod.register_forward_pre_hook(lambda _m, a, *, _L=L: _keep(_L, a[0]))
                    else:
                        mod = steer.resolve_block(model, L, spec)
                        h = mod.register_forward_hook(
                            lambda _m, _a, o, *, _L=L: _keep(_L, steer._block_hidden(o)))
                    handles.append(h)
                    self._register(["capture"], L, h)
                with torch.no_grad():
                    for t in texts:
                        ids = tok(steer.wrap_prompt_notk(tok, t, spec), return_tensors="pt")["input_ids"]
                        if reduce != "last" and ids.numel() < 2:
                            raise ValueError(f"capture text too short (<2 tokens): {t!r}")
                        grabbed.clear()
                        model(ids.to(steer.DEVICE))
                        for L in layers:
                            x = grabbed[L][0]
                            if x.shape[-1] != width:
                                raise RuntimeError(f"capture at L{L} {point} is {x.shape[-1]} wide, "
                                                   f"expected {width}; wrong hook point")
                            out[L].append(x[1:].mean(0) if reduce == "mean"
                                          else x[-1] if reduce == "last" else x[1:].norm(dim=-1))
            finally:
                for h in handles:
                    self._remove(h)
        return out


ENGINE = Engine()
