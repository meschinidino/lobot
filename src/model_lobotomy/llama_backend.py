"""llama.cpp/Metal backend (docs/plans/llama-backend.md): LlamaEngine.

Mirrors the Engine methods the product paths use (run, run_stream,
prefill) over the lobot-llamad daemon: a persistent Metal model with
per-request control vectors. VectorEdits map straight to the cvec
buffer; NeuronEdits convert exactly (down_proj has no bias, so a
dose d at MLP input n equals d * W_down[:, n] on the residual, the
site where llama.cpp applies the vector).

Out of scope, loudly: ablation (no static-vector equivalent),
Read/capture/dials (no activation access), create/train/derive.
Those raise UnsupportedError naming the torch backend.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import torch

from model_lobotomy import engine as engine_mod
from model_lobotomy import steer

ROOT = Path(__file__).resolve().parent.parent.parent
DAEMON_BIN = (ROOT / "third_party" / "lobot-llamad-build"
              / "lobot-llamad")


class UnsupportedError(RuntimeError):
    """A feature the llama backend cannot run (torch-only)."""


class DaemonError(RuntimeError):
    """The lobot-llamad child died or spoke garbage (never silent)."""


def _need_torch_msg(feature: str) -> str:
    return (f"{feature} needs LOBOT_BACKEND=torch. "
            f"This server runs the llama backend.")


# -- strict model handle (judge/guard go through daemon prefill) --------------


class _LastLogits:
    """Logits readable only at [0, -1] (what judge_score reads).

    Anything else raises: full-sequence logits are torch-only, and a
    wrong slice must never pass as data (spike.nll fails loudly).
    """

    def __init__(self, vec):
        self._vec = vec

    def __getitem__(self, key):
        try:
            tup = tuple(key)
        except TypeError:
            tup = None
        if tup != (0, -1):
            raise UnsupportedError(
                "full-sequence logits are torch-only")
        return self._vec


class LlamaModelHandle:
    """Strict stand-in for eng.model on the llama backend.

    __call__ answers only the judge pattern (one [1, T] forward,
    logits read at [0, -1]) through daemon prefill with the cvec
    cleared. Every other attribute raises UnsupportedError, so a
    missed torch-only call site fails loudly instead of computing
    a wrong number.
    """

    def __init__(self, engine):
        object.__setattr__(self, "_llama_engine", engine)

    def __call__(self, input_ids):
        if (not torch.is_tensor(input_ids) or input_ids.dim() != 2
                or input_ids.shape[0] != 1):
            raise UnsupportedError(
                "eng.model() on llama takes only a [1, T] tensor; "
                "run with LOBOT_BACKEND=torch")
        ids = [int(x) for x in
               input_ids[0].detach().cpu().tolist()]
        vec = object.__getattribute__(
            self, "_llama_engine")._prefill_ids(ids, cvec=None)
        return SimpleNamespace(logits=_LastLogits(vec))

    def __getattr__(self, name: str):
        if name.startswith("_llama_engine"):
            raise AttributeError(name)
        raise UnsupportedError(
            f"eng.model.{name} is torch-only; "
            f"run with LOBOT_BACKEND=torch")


# -- down_proj columns from HF safetensors (never the GGUF) --------------------


def down_proj_name(spec: steer.ModelSpec, layer: int) -> str:
    """Safetensors key for one layer's down_proj weight."""
    if spec.block_path == "qwen":
        return (f"model.language_model.layers.{layer}."
                f"mlp.down_proj.weight")
    return f"model.layers.{layer}.mlp.down_proj.weight"


class DownProjReader:
    """One layer's down_proj block at a time, fp32 on CPU.

    Reads the pinned HF snapshot with safe_open (mmap, no torch
    model load) and caches touched layers per (model, revision).
    """

    def __init__(self, model_id: str, revision: str):
        from huggingface_hub import snapshot_download
        self.snapshot = Path(snapshot_download(
            model_id, revision=revision))
        index = self.snapshot / "model.safetensors.index.json"
        self._map = (json.loads(index.read_text())["weight_map"]
                     if index.is_file() else None)
        self._cache: dict[int, torch.Tensor] = {}

    def column_block(self, spec: steer.ModelSpec,
                     layer: int) -> torch.Tensor:
        """[hidden, intermediate] fp32 block for one layer."""
        if layer in self._cache:
            return self._cache[layer]
        from safetensors import safe_open
        name = down_proj_name(spec, layer)
        if self._map is not None:
            try:
                fname = self._map[name]
            except KeyError:
                raise RuntimeError(
                    f"{name} missing from the HF snapshot; "
                    f"revision mismatch?") from None
        else:
            found = sorted(self.snapshot.glob("*.safetensors"))
            if len(found) != 1:
                raise RuntimeError(
                    f"no safetensors index and {len(found)} "
                    f"shards in {self.snapshot}")
            fname = found[0].name
        with safe_open(self.snapshot / fname, framework="pt",
                       device="cpu") as f:
            try:
                w = f.get_tensor(name)
            except Exception as e:
                raise RuntimeError(
                    f"{name} unreadable in {fname}: {e}") from e
        if tuple(w.shape) != (spec.hidden, spec.intermediate):
            raise RuntimeError(
                f"{name} shape {tuple(w.shape)} != "
                f"({spec.hidden}, {spec.intermediate})")
        w = w.detach().float().contiguous()
        self._cache[layer] = w
        return w


# -- EditSet -> cvec buffer -----------------------------------------------------


def edits_to_cvec(edits: engine_mod.EditSet, spec: steer.ModelSpec,
                  medians: dict, reader: DownProjReader,
                  ) -> tuple[bytes, int, int] | None:
    """Merged F32 cvec buffer for one request, or None when empty.

    Returns (raw LE bytes, il_start, il_end): a full
    n_layer*n_embd buffer with slot L-1 per touched torch layer L,
    every strength baked into the floats (daemon scale is 1.0).
    Ablation, Read, layer-0, and missing-median EditSets raise
    UnsupportedError as a whole; nothing is ever partial.
    """
    if edits.ablate:
        raise UnsupportedError(_need_torch_msg("ablation"))
    if edits.read:
        raise UnsupportedError(_need_torch_msg("capture"))
    meds = {int(k): float(v) for k, v in (medians or {}).items()}
    acc: dict[int, torch.Tensor] = {}

    def _slot(layer: int) -> torch.Tensor:
        slot = acc.get(layer)
        if slot is None:
            slot = acc[layer] = torch.zeros(
                spec.hidden, dtype=torch.float32)
        return slot

    for e in edits.neuron:
        if e.layer < 1:
            raise UnsupportedError(
                "layer-0 edits cannot run on the llama backend "
                "(llama.cpp forbids direction.0); "
                "run with LOBOT_BACKEND=torch")
        if e.layer not in meds:
            raise UnsupportedError(
                f"no cached median for layer {e.layer}; run once "
                f"with LOBOT_BACKEND=torch")
        w = reader.column_block(spec, e.layer)
        delta = (float(e.dose) * meds[e.layer]
                 / steer.column_norm_of(w, e.neuron))
        _slot(e.layer).add_(w[:, e.neuron] * delta)
    for v in edits.vector:
        if not 0 <= v.layer < spec.layers:
            raise ValueError(
                f"layer {v.layer} out of bounds "
                f"[0,{spec.layers})")
        if v.layer < 1:
            raise UnsupportedError(
                "layer-0 edits cannot run on the llama backend "
                "(llama.cpp forbids direction.0); "
                "run with LOBOT_BACKEND=torch")
        u = torch.as_tensor(list(v.direction), dtype=torch.float32)
        if u.numel() != spec.hidden:
            raise ValueError(
                f"vector has {u.numel()} dims, model needs "
                f"{spec.hidden}")
        _slot(v.layer).add_(u * (float(v.scale) * float(v.dose)))
    if not acc:
        return None
    buf = torch.zeros(spec.layers * spec.hidden,
                      dtype=torch.float32)
    for layer, vec in acc.items():
        buf[(layer - 1) * spec.hidden:layer * spec.hidden] = vec
    return (buf.numpy().tobytes(), min(acc), max(acc))


# -- GGUF resolution ------------------------------------------------------------


def model_slug(model_id: str) -> str:
    """models/ dir name for one model id (same rule as calib files)."""
    return model_id.lower().replace("/", "-").replace("_", "-")


def torch_vocab_size(model_id: str, tok) -> int:
    """Logit dim the torch model produces (HF text config).

    This is the config vocab size (Qwen pads it past the
    tokenizer length), NOT len(tokenizer): the daemon must match
    what torch emits, row for row.
    """
    try:
        from transformers import AutoConfig
        tcfg = AutoConfig.from_pretrained(
            model_id).get_text_config()
        return int(tcfg.vocab_size)
    except Exception:
        return len(tok)


def resolve_gguf(model_id: str, revision: str,
                 ) -> tuple[Path, dict]:
    """(GGUF path, sidecar) for one model + revision + quant.

    Layout: models/<slug>/<rev12>-<quant>.gguf plus a .json sidecar
    (HF repo, revision, llamacpp commit, quant, sha256). Quant comes
    from LOBOT_LLAMA_QUANT (default f16). Missing files fail with
    the exact convert command; nothing converts inside load().
    """
    quant = os.environ.get("LOBOT_LLAMA_QUANT", "f16")
    gguf = (ROOT / "models" / model_slug(model_id)
            / f"{revision[:12]}-{quant}.gguf")
    if not gguf.is_file():
        raise RuntimeError(
            f"no GGUF for {model_id} {quant}: {gguf} missing; run "
            f"uv run python scripts/convert_gguf.py --model "
            f"{model_id} --quant {quant}")
    sidecar_path = gguf.with_name(gguf.stem + ".json")
    if not sidecar_path.is_file():
        raise RuntimeError(
            f"GGUF sidecar missing: {sidecar_path}; reconvert with "
            f"scripts/convert_gguf.py")
    try:
        sidecar = json.loads(sidecar_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(
            f"GGUF sidecar unreadable: {sidecar_path}: {e}") from e
    if sidecar.get("revision") != revision:
        raise RuntimeError(
            f"GGUF {gguf} pins revision "
            f"{sidecar.get('revision')}, want {revision}")
    return gguf, sidecar


# -- daemon child process -------------------------------------------------------


class DaemonProc:
    """One lobot-llamad child: JSON lines over stdio, stderr drained.

    Methods: info dict (from load), prefill(ids, cvec) -> logits
    bytes, generate(...) yielding ("token", id) then ("done", dict),
    close(). A dead child or a garbage line raises DaemonError.
    """

    def __init__(self, binary: str | Path, gguf: str | Path,
                 extra_args: list[str] | None = None):
        binary = str(binary)
        if not (Path(binary).is_file()
                and os.access(binary, os.X_OK)):
            raise RuntimeError(
                f"lobot-llamad missing: {binary}; run "
                f"scripts/build_llamacpp.sh")
        args = [binary, "-m", str(gguf), *(extra_args or [])]
        # KV prefix reuse across requests with the same cvec (daemon
        # default). LOBOT_LLAMA_PREFIX_CACHE=0 makes every request cold.
        if os.environ.get("LOBOT_LLAMA_PREFIX_CACHE", "1") == "0":
            args.append("--no-prefix-cache")
        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        self._io_lock = threading.Lock()
        try:
            info = self._request({"op": "info"})
        except DaemonError as e:
            self._kill()
            raise DaemonError(
                f"lobot-llamad failed to start ({gguf}): {e}; "
                f"stderr: {' | '.join(self._stderr_tail)}") from e
        if not info.get("ok"):
            self._kill()
            raise DaemonError(
                f"lobot-llamad info failed: {info.get('error')}")
        self.info = info

    def _drain_stderr(self) -> None:
        try:
            for line in self._proc.stderr:
                self._stderr_tail.append(
                    line.decode("utf-8", "replace").rstrip())
        except (ValueError, OSError):
            pass

    def _dead(self) -> bool:
        return self._proc.poll() is not None

    def _send(self, obj: dict) -> None:
        if self._dead():
            raise DaemonError(
                "lobot-llamad died; stderr: "
                f"{' | '.join(self._stderr_tail)}")
        try:
            self._proc.stdin.write(
                (json.dumps(obj) + "\n").encode("utf-8"))
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise DaemonError(
                f"lobot-llamad write failed: {e}; stderr: "
                f"{' | '.join(self._stderr_tail)}") from e

    def _readline(self) -> dict:
        try:
            raw = self._proc.stdout.readline()
        except (ValueError, OSError) as e:
            raise DaemonError(
                f"lobot-llamad read failed: {e}") from e
        if not raw:
            raise DaemonError(
                "lobot-llamad died mid-request; stderr: "
                f"{' | '.join(self._stderr_tail)}")
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise DaemonError(
                f"lobot-llamad spoke garbage: {e}") from e

    def _request(self, obj: dict) -> dict:
        """One request, one reply line.

        Skips bare {"ok": true} lines: older daemons ack cancel,
        and an ack that lands after a generate's done line would
        otherwise answer this request (and every later reply would
        belong to the request before it).
        """
        with self._io_lock:
            self._send(obj)
            while True:
                resp = self._readline()
                if resp != {"ok": True}:
                    return resp

    def prefill(self, ids: list[int],
                cvec: tuple[str, int, int] | None) -> bytes:
        """Full last-position logits as raw f32 LE bytes."""
        req: dict = {"op": "prefill", "ids": ids}
        if cvec is not None:
            req["cvec_b64"], req["il_start"], req["il_end"] = cvec
        resp = self._request(req)
        if not resp.get("ok"):
            raise DaemonError(
                f"lobot-llamad prefill failed: "
                f"{resp.get('error')}")
        if "logits_b64" not in resp:
            raise DaemonError(
                f"lobot-llamad prefill reply has no logits: "
                f"{str(resp)[:200]}")
        return base64.b64decode(resp["logits_b64"])

    def _cancel_watcher(self, cancel: threading.Event | None):
        """(stop event, thread) sending one cancel when cancel is set."""
        stop = threading.Event()
        if cancel is None:
            return stop, None

        def _watch():
            while not stop.wait(0.01):
                if cancel.is_set():
                    try:
                        self._send({"op": "cancel"})
                    except DaemonError:
                        pass
                    return
        t = threading.Thread(target=_watch, daemon=True)
        t.start()
        return stop, t

    def _batch_request(self, req: dict,
                       cancel: threading.Event | None) -> dict:
        """One batch op with a cancel watcher; one reply line."""
        with self._io_lock:
            self._send(req)
            stop, watcher = self._cancel_watcher(cancel)
            try:
                while True:
                    resp = self._readline()
                    if resp != {"ok": True}:
                        return resp
            finally:
                stop.set()
                if watcher is not None:
                    watcher.join(timeout=5)

    def generate_batch(self, prompts: list[list[int]], max_new: int,
                       stop_ids: list[int],
                       cvec: tuple[str, int, int] | None,
                       cancel: threading.Event | None = None) -> dict:
        """Greedy ids per prompt, one sequence each (--parallel).

        Returns the done_batch dict: outs, stop, prefill_ms,
        decode_ms. A daemon error raises DaemonError; a cancel
        returns stop == "cancel" with partial outs.
        """
        req: dict = {"op": "generate_batch", "prompts": prompts,
                     "max_new": max_new, "stop_ids": stop_ids}
        if cvec is not None:
            req["cvec_b64"], req["il_start"], req["il_end"] = cvec
        resp = self._batch_request(req, cancel)
        if resp.get("type") != "done_batch":
            raise DaemonError(
                f"lobot-llamad spoke garbage: {str(resp)[:200]}")
        if resp.get("stop") == "error":
            raise DaemonError(
                f"lobot-llamad generate_batch failed: "
                f"{resp.get('error')}")
        return resp

    def score_batch(self, items: list[dict],
                    cvec: tuple[str, int, int] | None = None,
                    cancel: threading.Event | None = None,
                    ) -> list[dict] | None:
        """Per item: {"lp": [...], "last": [...]} (see the daemon).

        Items are {"ids", "from"?, "last_ids"?}. Returns None when
        cancelled; any other failure raises DaemonError.
        """
        req: dict = {"op": "score_batch", "items": items}
        if cvec is not None:
            req["cvec_b64"], req["il_start"], req["il_end"] = cvec
        resp = self._batch_request(req, cancel)
        if resp.get("cancelled"):
            return None
        if not resp.get("ok") or "items" not in resp:
            raise DaemonError(
                f"lobot-llamad score_batch failed: "
                f"{resp.get('error') or str(resp)[:200]}")
        return resp["items"]

    def generate(self, ids: list[int], max_new: int,
                 stop_ids: list[int],
                 cvec: tuple[str, int, int] | None,
                 cancel: threading.Event | None = None):
        """Yields ("token", id), then ("done", done-dict).

        A set cancel event sends {"op":"cancel"} from a watcher
        thread; the daemon stops between tokens. Closing the
        generator early cancels and drains to the done line so
        the next request starts clean.
        """
        with self._io_lock:
            req: dict = {"op": "generate", "ids": ids,
                         "max_new": max_new, "stop_ids": stop_ids}
            if cvec is not None:
                req["cvec_b64"], req["il_start"], req["il_end"] = cvec
            self._send(req)
            stop_watcher = threading.Event()
            watcher = None
            if cancel is not None:
                def _watch():
                    while not stop_watcher.wait(0.01):
                        if cancel.is_set():
                            try:
                                self._send({"op": "cancel"})
                            except DaemonError:
                                pass
                            return
                watcher = threading.Thread(target=_watch,
                                           daemon=True)
                watcher.start()
            done = None
            try:
                while True:
                    ev = self._readline()
                    if ev.get("type") == "token":
                        yield ("token", int(ev["id"]))
                    elif ev.get("type") == "done":
                        done = ev
                        yield ("done", ev)
                        return
                    elif ev == {"ok": True}:
                        continue  # older daemons ack cancel
                    else:
                        raise DaemonError(
                            f"lobot-llamad spoke garbage: "
                            f"{str(ev)[:200]}")
            finally:
                stop_watcher.set()
                if watcher is not None:
                    watcher.join(timeout=5)
                if done is None:
                    # Closed early: cancel and drain to done so
                    # the next request starts at a clean line.
                    try:
                        self._send({"op": "cancel"})
                        while True:
                            ev = self._readline()
                            if ev.get("type") == "done":
                                break
                    except DaemonError:
                        pass

    def _kill(self) -> None:
        try:
            self._proc.kill()
        except OSError:
            pass

    def close(self) -> None:
        """Shutdown op, then SIGKILL fallback. Idempotent."""
        with self._io_lock:
            if self._dead():
                return
            try:
                self._send({"op": "shutdown"})
                self._proc.wait(timeout=10)
            except Exception:
                self._kill()
        try:
            self._proc.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass


# -- LlamaEngine ---------------------------------------------------------------


def _touched_layers(edits: engine_mod.EditSet) -> set[int]:
    return ({e.layer for e in edits.neuron}
            | {e.layer for e in edits.vector})


def _validate_edits(spec: steer.ModelSpec,
                    edits: engine_mod.EditSet) -> None:
    """Bounds checks (no torch model: dims check at load instead)."""
    from model_lobotomy.steer import RecipeRow
    steer.validate_recipe(
        [RecipeRow(r.layer, r.neuron, r.dose)
         for r in edits.neuron], spec)
    for v in list(edits.vector) + list(edits.ablate):
        if not 0 <= v.layer < spec.layers:
            raise ValueError(
                f"layer {v.layer} out of bounds "
                f"[0,{spec.layers})")
    for r in edits.read:
        if not 0 <= r.layer < spec.layers:
            raise ValueError(
                f"layer {r.layer} out of bounds "
                f"[0,{spec.layers})")


class LlamaEngine:
    """The llama.cpp backend: same run/prefill shapes as Engine.

    One lobot-llamad child per engine, chosen by LOBOT_BACKEND=llama
    in engine.get_engine(). proc_factory injects a fake daemon for
    unit tests (same methods as DaemonProc).
    """

    backend = "llama"

    def supports(self, feature: str) -> bool:
        """False for every FEATURES member (torch-only gaps)."""
        return False

    def __init__(self, proc_factory=None):
        self.model = None
        self.tokenizer = None
        self.model_id: str | None = None
        self.revision: str | None = None
        self.rung: int | None = None
        self.lock = threading.Lock()
        self._proc = None
        self._proc_factory = proc_factory or DaemonProc
        self._reader: DownProjReader | None = None
        self._sidecar: dict | None = None
        self._stop_ids: list[int] | None = None

    @property
    def spec(self):
        if self.model_id is None:
            raise RuntimeError("no model loaded")
        return steer.get_spec(self.model_id)

    @property
    def warm(self) -> bool:
        return self._proc is not None

    def load(self, model_id: str,
             revision: str | None = None) -> int:
        """Start the daemon on the resolved GGUF (no convert here).

        Resolves the revision like Engine.load, finds
        models/<slug>/<rev12>-<quant>.gguf, checks daemon dims
        against the HF tokenizer and spec, and warms. Returns 1.
        """
        from huggingface_hub import HfApi
        if self._proc is not None:
            self.unload()
        sha = revision or HfApi().model_info(model_id).sha
        gguf, sidecar = resolve_gguf(model_id, sha)
        tok = steer.load_tokenizer(model_id)
        binary = os.environ.get("LOBOT_LLAMAD_BIN",
                                str(DAEMON_BIN))
        proc = self._proc_factory(binary, gguf)
        spec = steer.get_spec(model_id)
        info = proc.info
        if (info.get("n_layer") != spec.layers
                or info.get("n_embd") != spec.hidden):
            proc.close()
            raise RuntimeError(
                f"GGUF dims {info.get('n_layer')}/"
                f"{info.get('n_embd')} != spec "
                f"{spec.layers}/{spec.hidden}")
        want_vocab = torch_vocab_size(model_id, tok)
        if info.get("n_vocab") != want_vocab:
            proc.close()
            raise RuntimeError(
                f"GGUF vocab {info.get('n_vocab')} != torch "
                f"logit dim {want_vocab}")
        self._proc = proc
        self._sidecar = sidecar
        self.model = LlamaModelHandle(self)
        self.tokenizer = tok
        self.model_id, self.revision, self.rung = (
            model_id, sha, 1)
        self._reader = DownProjReader(model_id, sha)
        self._stop_ids = self._resolve_stop_ids(model_id, tok)
        print(f"[llama] {model_id} {sidecar.get('quant')} warm "
              f"({gguf.name}, llama.cpp "
              f"{info.get('commit', '?')[:12]})", flush=True)
        return 1

    @staticmethod
    def _resolve_stop_ids(model_id: str, tok) -> list[int]:
        """EOS ids from the HF generation config (generate's source).

        Falls back to the tokenizer EOS when the config carries
        none. Always a list, even for a single id.
        """
        try:
            from transformers import AutoConfig
            eos = AutoConfig.from_pretrained(
                model_id).get_text_config().eos_token_id
        except Exception:
            eos = None
        if eos is None:
            eos = tok.eos_token_id
        if isinstance(eos, int):
            return [eos]
        return [int(x) for x in eos]

    def unload(self) -> None:
        if self._proc is not None:
            try:
                self._proc.close()
            finally:
                self._proc = None
        self.model, self.tokenizer = None, None
        self.model_id, self.revision, self.rung = None, None, None
        self._reader = None
        self._sidecar = None
        self._stop_ids = None

    def fingerprint(self) -> dict:
        """Torch keys plus backend/gguf/quant/commit (never shared)."""
        if self._proc is None or self._sidecar is None:
            raise RuntimeError("no model loaded")
        key = steer.calibration_cache_key(
            self.model_id, self.revision, self.tokenizer,
            str(self._sidecar.get("quant")), "metal")
        key["backend"] = "llama"
        key["gguf_sha256"] = self._sidecar.get("gguf_sha256")
        key["quant"] = self._sidecar.get("quant")
        key["llamacpp_commit"] = self._sidecar.get(
            "llamacpp_commit")
        return key

    # -- no-op registry (no hooks on this backend) --

    def hook_count(self, kind: str | None = None) -> int:
        return 0

    def remove_all(self) -> None:
        pass

    def clean(self):
        from contextlib import contextmanager

        @contextmanager
        def _clean():
            yield

        return _clean()

    # -- unsupported (torch-only) --

    def capture(self, texts: list[str], layers: list[int],
                point: str = "mlp_in",
                reduce: str = "mean_1") -> dict:
        raise UnsupportedError(_need_torch_msg("capture"))

    def capture_with_edits(self, texts: list[str],
                           layers: list[int], edits, medians: dict,
                           point: str = "mlp_in",
                           reduce: str = "none") -> dict:
        raise UnsupportedError(_need_torch_msg("capture"))

    # -- cvec + ids --

    def _cvec_for(self, edits: engine_mod.EditSet, medians: dict,
                  ) -> tuple[str, int, int] | None:
        conv = edits_to_cvec(edits, self.spec, medians,
                             self._reader)
        if conv is None:
            return None
        raw, lo, hi = conv
        return (base64.b64encode(raw).decode("ascii"), lo, hi)

    def _encode(self, prompt: str,
                wrapped: str | None = None) -> list[int]:
        text = (wrapped if wrapped is not None
                else steer.wrap_prompt_notk(
                    self.tokenizer, prompt, self.spec))
        out = self.tokenizer(text, return_tensors="pt")
        ids = (out["input_ids"] if isinstance(out, dict)
               else out.input_ids)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, list) and ids and isinstance(
                ids[0], list):
            ids = ids[0]
        return [int(i) for i in ids]

    def _prefill_ids(self, ids: list[int], cvec) -> torch.Tensor:
        if self._proc is None:
            raise RuntimeError("no model loaded")
        raw = self._proc.prefill(ids, cvec)
        import numpy as np
        vec = np.frombuffer(raw, dtype=np.float32).copy()
        if vec.size != self._proc.info["n_vocab"]:
            raise DaemonError(
                f"prefill returned {vec.size} logits, want "
                f"{self._proc.info['n_vocab']}")
        return torch.from_numpy(vec)

    def prefill(self, user_text: str, edits=None, *,
                medians: dict | None = None):
        """Last-position logits (1-D f32 vocab tensor), like Engine."""
        if self._proc is None:
            raise RuntimeError("no model loaded")
        ids = self._encode(user_text)
        cvec = None
        if edits is not None:
            if medians is None:
                raise ValueError(
                    "prefill with edits needs medians")
            _validate_edits(self.spec, edits)
            cvec = self._cvec_for(edits, medians)
        return self._prefill_ids(ids, cvec)

    # -- run --

    def _result(self, ids: list[int], done: dict, total_s: float,
                n_layers: int) -> dict:
        n_new = len(ids)
        prefill_s = float(done.get("prefill_ms", 0.0)) / 1000.0
        decode_s = float(done.get("decode_ms", 0.0)) / 1000.0
        return {
            "text": self.tokenizer.decode(
                ids, skip_special_tokens=True),
            "new_tokens": n_new,
            "token_ids": [int(i) for i in ids],
            "seconds": round(total_s, 1),
            "prefill_s": round(prefill_s, 2),
            "decode_s": round(decode_s, 2),
            "tok_per_s": (round(n_new / total_s, 1)
                          if total_s > 0 else 0.0),
            "peak_mib": None,
            "driver_mib": None,
            "cancelled": bool(done.get("cancelled", False)),
            "hook_calls": n_layers * (1 + n_new),
            "reads": {},
            "engine_seconds": round(total_s, 1),
        }

    def run(self, prompt: str, edits: engine_mod.EditSet, *,
            medians: dict, max_new_tokens: int, seed: int,
            cancel=None, wrapped: str | None = None) -> dict:
        """One greedy generation (same keys as Engine.run).

        Seed is accepted and ignored: greedy decode is
        deterministic. hook_calls counts touched layers times
        forward passes (1 prefill + one decode each), honestly 0
        for an empty EditSet.
        """
        if self._proc is None:
            raise RuntimeError("no model loaded")
        _validate_edits(self.spec, edits)
        max_new_tokens = min(max_new_tokens,
                             steer.MAX_NEW_TOKENS_CAP)
        cvec = self._cvec_for(edits, medians)
        layers = _touched_layers(edits)
        ids = self._encode(prompt, wrapped)
        t0 = time.perf_counter()
        if cancel is not None and cancel.is_set():
            done: dict = {"cancelled": True, "prefill_ms": 0.0,
                          "decode_ms": 0.0}
            return self._result([], done,
                                time.perf_counter() - t0, 0)
        out: list[int] = []
        done = {}
        for kind, payload in self._proc.generate(
                ids, max_new_tokens, self._stop_ids, cvec,
                cancel):
            if kind == "token":
                out.append(payload)
            else:
                done = payload
        if done.get("stop") == "error":
            raise DaemonError(
                f"lobot-llamad generate failed: "
                f"{done.get('error')}")
        return self._result(out, done,
                            time.perf_counter() - t0,
                            len(layers))

    def run_stream(self, prompt: str, edits: engine_mod.EditSet, *,
                   medians: dict, max_new_tokens: int, seed: int,
                   cancel=None, dial: dict | None = None,
                   dials: list[dict] | None = None,
                   wrapped: str | None = None):
        """run() as token events, then one done event.

        Same event shapes as Engine.run_stream. dial/dials raise
        UnsupportedError (no activation taps on this backend).
        Closing the generator cancels the daemon request.
        """
        if self._proc is None:
            raise RuntimeError("no model loaded")
        if dial is not None or dials:
            raise UnsupportedError(_need_torch_msg("dial"))
        _validate_edits(self.spec, edits)
        max_new_tokens = min(max_new_tokens,
                             steer.MAX_NEW_TOKENS_CAP)
        cvec = self._cvec_for(edits, medians)
        layers = _touched_layers(edits)
        ids = self._encode(prompt, wrapped)
        t0 = time.perf_counter()
        if cancel is None:
            cancel = threading.Event()
        if cancel.is_set():
            done: dict = {"cancelled": True, "prefill_ms": 0.0,
                          "decode_ms": 0.0}
            yield {"type": "done", "result": self._result(
                [], done, time.perf_counter() - t0, 0)}
            return
        stream = self._proc.generate(
            ids, max_new_tokens, self._stop_ids, cvec, cancel)
        out: list[int] = []
        prev = ""
        done = {}
        try:
            for kind, payload in stream:
                if kind == "token":
                    out.append(payload)
                    text = self.tokenizer.decode(
                        out, skip_special_tokens=True)
                    delta = text[len(prev):]
                    prev = text
                    if delta:
                        yield {"type": "token", "text": delta}
                else:
                    done = payload
        except GeneratorExit:
            cancel.set()
            stream.close()
            raise
        if done.get("stop") == "error":
            raise DaemonError(
                f"lobot-llamad generate failed: "
                f"{done.get('error')}")
        yield {"type": "done", "result": self._result(
            out, done, time.perf_counter() - t0, len(layers))}

    def run_batched(self, prompt: str,
                    edit_sets: list[engine_mod.EditSet], *,
                    medians: dict, max_new_tokens: int, seed: int,
                    cancel=None, fallback: bool = True) -> dict:
        """Sequential singles (one cvec per context, no fallback).

        Same output shape as Engine.run_batched; fallback_used is
        always False because there is no batched fast path.
        """
        if self._proc is None:
            raise RuntimeError("no model loaded")
        if not edit_sets:
            raise ValueError(
                "run_batched needs at least one EditSet")
        rows = [self.run(prompt, es, medians=medians,
                         max_new_tokens=max_new_tokens,
                         seed=seed, cancel=cancel)
                for es in edit_sets]
        return {"rows": rows, "fallback_used": False,
                "hook_calls": sum(r["hook_calls"] for r in rows),
                "reads": {}}

    def run_batched_stream(self, prompt: str,
                           edit_sets: list[engine_mod.EditSet], *,
                           medians: dict, max_new_tokens: int,
                           seed: int, cancel=None,
                           fallback: bool = True):
        """run_batched() as live per-column events, then done."""
        if self._proc is None:
            raise RuntimeError("no model loaded")
        if not edit_sets:
            raise ValueError(
                "run_batched needs at least one EditSet")
        if cancel is None:
            cancel = threading.Event()
        rows = []
        try:
            for b, es in enumerate(edit_sets):
                for ev in self.run_stream(
                        prompt, es, medians=medians,
                        max_new_tokens=max_new_tokens,
                        seed=seed, cancel=cancel):
                    if ev["type"] == "token":
                        yield {"type": "token", "col": b,
                               "text": ev["text"]}
                    else:
                        rows.append(ev["result"])
        except GeneratorExit:
            cancel.set()
            raise
        yield {"type": "done", "result": {
            "rows": rows, "fallback_used": False,
            "hook_calls": sum(r["hook_calls"] for r in rows),
            "reads": {}}}


# -- Create lane daemon ----------------------------------------------------------


class CreateDaemon:
    """A batched lobot-llamad for Create (trait_eval.LlamaLane).

    Separate from LlamaEngine: Create keeps the torch engine for
    capture and the guard, and sends generation, judge and NLL here.
    The daemon runs --parallel sequences in one unified KV buffer;
    quant comes from LOBOT_CREATE_QUANT (default f16, the parity
    lane: 12/12 byte-exact generations vs torch).
    """

    PARALLEL = 32
    # 32 prompts x (prompt + 60 new) fit in ~4k tokens; score groups
    # cap at n_batch (2048). Torch shares the GPU, so stay small.
    CTX = 8192

    def __init__(self, model_id: str, revision: str, tokenizer,
                 quant: str | None = None, proc_factory=None):
        self.quant = quant or os.environ.get("LOBOT_CREATE_QUANT", "f16")
        self.model_id, self.revision = model_id, revision
        self.tokenizer = tokenizer
        self.spec = steer.get_spec(model_id)
        prev = os.environ.get("LOBOT_LLAMA_QUANT")
        os.environ["LOBOT_LLAMA_QUANT"] = self.quant
        try:
            gguf, self.sidecar = resolve_gguf(model_id, revision)
        finally:
            if prev is None:
                os.environ.pop("LOBOT_LLAMA_QUANT", None)
            else:
                os.environ["LOBOT_LLAMA_QUANT"] = prev
        binary = os.environ.get("LOBOT_LLAMAD_BIN", str(DAEMON_BIN))
        factory = proc_factory or DaemonProc
        self.proc = factory(binary, gguf, [
            "--parallel", str(self.PARALLEL), "--ctx-size", str(self.CTX),
            "--no-prefix-cache"])
        info = self.proc.info
        if (info.get("n_layer") != self.spec.layers
                or info.get("n_embd") != self.spec.hidden):
            self.proc.close()
            raise RuntimeError(
                f"GGUF dims {info.get('n_layer')}/{info.get('n_embd')} "
                f"!= spec {self.spec.layers}/{self.spec.hidden}")
        self.stop_ids = LlamaEngine._resolve_stop_ids(model_id, tokenizer)

    def info(self) -> dict:
        return {"quant": self.quant,
                "gguf_sha256": self.sidecar.get("gguf_sha256"),
                "llamacpp_commit": self.sidecar.get("llamacpp_commit"),
                "parallel": self.PARALLEL}

    def _cvec(self, edits: engine_mod.EditSet):
        conv = edits_to_cvec(edits, self.spec, {}, None)
        if conv is None:
            return None
        raw, lo, hi = conv
        return (base64.b64encode(raw).decode("ascii"), lo, hi)

    def gen_ids_many(self, prompts: list[list[int]],
                     edits: engine_mod.EditSet, max_new: int,
                     cancel: threading.Event | None = None,
                     ) -> list[list[int]]:
        """Greedy reply ids per prompt (stop id kept, like run()).

        Vector-only or empty EditSets; chunks of PARALLEL prompts
        within the context budget. A cancel returns what is done
        (the caller checks its cancel event).
        """
        if edits.ablate or edits.read or edits.neuron:
            raise UnsupportedError(
                "Create lane batches vector edits only")
        max_new = min(max_new, steer.MAX_NEW_TOKENS_CAP)
        cvec = self._cvec(edits)
        outs: list[list[int]] = []
        i = 0
        while i < len(prompts):
            j, budget = i, 0
            while (j < len(prompts) and j - i < self.PARALLEL
                   and (j == i or budget + len(prompts[j]) + max_new
                        <= self.CTX)):
                budget += len(prompts[j]) + max_new
                j += 1
            resp = self.proc.generate_batch(
                prompts[i:j], max_new, self.stop_ids, cvec, cancel)
            outs.extend([[int(t) for t in o] for o in resp["outs"]])
            if resp.get("stop") == "cancel":
                break
            i = j
        return outs

    def score_many(self, items: list[dict],
                   cancel: threading.Event | None = None) -> list[dict]:
        """score_batch with no cvec (judge and NLL read the base model)."""
        if not items:
            return []
        res = self.proc.score_batch(items, None, cancel)
        if res is None:
            return []
        return res

    def close(self) -> None:
        self.proc.close()
