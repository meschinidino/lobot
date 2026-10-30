"""Smoke checks for T2 (ladder), T4 (calibration), T5 (module path).

Unit-level by default (no model, seconds). Pass --with-model to also run the
model-backed section on the warmed ladder (minutes; rung 1 must pass).
"""

from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import steer  # noqa: E402
from model_lobotomy.steer import RecipeRow  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name, fn):
    try:
        detail = fn() or ""
    except Exception as e:  # noqa: BLE001
        results.append((name, FAIL, f"{e}\n{traceback.format_exc(limit=2)}"))
    else:
        results.append((name, PASS, detail))


# -- T2: ladder ---------------------------------------------------------------

def t2_order():
    assert [r.n for r in steer.LADDER] == [1, 2, 3, 4], "rung order"
    r1, r2, r3, r4 = steer.LADDER
    assert r1.model_id == steer.MODEL_ID and r1.device == "mps"
    assert "loads + generates + hooks fire" in r1.test
    assert r2.model_id == steer.RUNG2_MODEL_ID == "Qwen/Qwen3.5-2B"
    assert r3.model_id is None and "verify-first" in r3.test, "rung 3 unresolved"
    assert r4.model_id == steer.MODEL_ID and r4.device == "cpu"
    assert "slow accepted" in r4.test
    return "1:4B-mps 2:2B-mps 3:gguf-unresolved 4:cpu"


def t2_rung2_exists_live():
    from huggingface_hub import model_info

    info = model_info(steer.RUNG2_MODEL_ID)
    assert info.gated is None or info.gated is False, f"gated={info.gated}"
    return f"{info.id} sha={info.sha[:12]} gated={info.gated}"


def t2_rung3_unresolved_no_model():
    r = steer.attempt_rung(steer.LADDER[2])
    assert r["ok"] is False and "unresolved" in r["detail"], r
    return r["detail"][:60]


def t2_preset_binding():
    preset = json.loads((ROOT / "presets" / "evil-qwen.json").read_text())
    steer.check_preset_binding(preset, steer.MODEL_ID)  # must not raise
    try:
        steer.check_preset_binding(preset, "Other/Model")
    except RuntimeError as e:
        assert "re-derive" in str(e), str(e)
        return "match ok; mismatch raises with recovery"
    raise AssertionError("mismatched preset did not raise")


def t2_recovery_text():
    assert "contrast-prompt" in steer.RECOVERY_PROCEDURE, steer.RECOVERY_PROCEDURE
    assert "Day-0" in steer.RECOVERY_PROCEDURE
    assert "half a day" in steer.RECOVERY_PROCEDURE
    for word in ("hook-compat", "recalibration", "re-judged"):
        assert word in steer.QUANT_RUNG_RULE, steer.QUANT_RUNG_RULE
    return "recovery + quantized-rung rule pinned"


def t2_all_fail_raises_with_recovery():
    with patch.object(steer, "attempt_rung",
                       lambda rung, **k: {"ok": False, "rung": rung.n,
                                          "detail": "nope", "seconds": 0.0}):
        try:
            steer.load_with_ladder()
        except RuntimeError as e:
            assert "ALL rungs failed" in str(e), str(e)
            assert "contrast-prompt" in str(e), str(e)
            return "ALL-fail raises + names recovery"
    raise AssertionError("load_with_ladder did not raise")


# -- T4: calibration ----------------------------------------------------------

class _FakeTok:
    chat_template = "fake-template"

    def apply_chat_template(self, *a, **k):
        return "wrapped"

    def __call__(self, *a, **k):
        class _Ids:
            def numel(self):
                return 1

            def to(self, device):
                return self

        return {"input_ids": _Ids()}


def _fake_layer():
    class _H:
        def remove(self):
            pass

    class _L:
        def register_forward_hook(self, fn):
            return _H()

    return _L()


def _fake_model():
    inner = SimpleNamespace(layers={13: _fake_layer(), 14: _fake_layer()})
    return SimpleNamespace(model=SimpleNamespace(language_model=inner))


def t4_key_shape():
    key = steer.calibration_cache_key("m", "r", _FakeTok(), "torch.float16", "mps")
    for f in ("model_id", "revision", "template_sha256", "thinking", "dtype",
              "quant", "device", "cal_code"):
        assert f in key, f"missing {f}"
    assert key["quant"] is None
    assert len(key["template_sha256"]) == 16
    return "8-field D9 key incl quant"


def t4_stored_cache_valid():
    d = json.loads((ROOT / "docs" / "calibration.json").read_text())
    for f in ("model_id", "revision", "template_sha256", "thinking", "dtype",
              "quant", "device", "cal_code", "prompts_sha256"):
        assert f in d["key"], f"missing {f}"
    assert d["medians"]["13"] > 0 and d["medians"]["14"] > 0
    assert len(d["prompt_hashes"]) == 3
    return f"medians L13={d['medians']['13']:.2f} L14={d['medians']['14']:.2f}"


def t4_isolation_assert():
    steer._active_steering.append(object())  # noqa: SLF001
    try:
        steer.calibrate_layer_medians(_fake_model(), _FakeTok(), ["x"], [13],
                                      cache_path=Path("/nonexistent/x.json"), key={})
    except RuntimeError as e:
        assert "OFF" in str(e), str(e)
        return "hooks-ON raises before any model use"
    finally:
        steer._active_steering.pop()  # noqa: SLF001
    raise AssertionError("isolation assert did not fire")


def t4_empty_filter():
    try:
        steer.calibrate_layer_medians(_fake_model(), _FakeTok(), ["", "   "], [13],
                                      cache_path=Path("/nonexistent/x.json"), key={})
    except ValueError as e:
        assert "non-empty" in str(e), str(e)
        return "all-empty prompts rejected"
    raise AssertionError("empty filter did not fire")


def t4_short_guard():
    with tempfile.TemporaryDirectory() as td:
        try:
            steer.calibrate_layer_medians(_fake_model(), _FakeTok(), ["hello"], [13],
                                          cache_path=Path(td) / "c.json", key={"k": 1})
        except ValueError as e:
            assert "<2 tokens" in str(e), str(e)
            return "1-token wrapped prompt rejected"
    raise AssertionError("short guard did not fire")


def t4_cache_hit_no_model():
    import hashlib as _hl

    prompts = ["a prompt here"]
    key = {"model_id": "m"}
    full = dict(key)
    full["prompts_sha256"] = _hl.sha256("\n".join(prompts).encode()).hexdigest()[:16]
    payload = {"key": full, "medians": {"13": 9.5}, "prompt_hashes": ["h"],
               "token_counts": {"13": [3]}}
    with tempfile.TemporaryDirectory() as td:
        cp = Path(td) / "c.json"
        cp.write_text(json.dumps(payload))
        med, info = steer.calibrate_layer_medians(
            None, _FakeTok(), prompts, [13], cache_path=cp, key=dict(key))
    assert info["cache_hit"] is True and med == {13: 9.5}, (med, info)
    return "hit reuses medians without model"


# -- T5: module path ----------------------------------------------------------

def t5_assert_fires_no_path():
    try:
        steer.resolve_down_proj(SimpleNamespace(), 0)
    except RuntimeError as e:
        assert "module path mismatch" in str(e), str(e)
        assert "model.model.language_model.layers[0]" in str(e), str(e)
        return "missing path fails loudly"
    raise AssertionError("assert did not fire")


def t5_assert_fires_bad_layout():
    block = SimpleNamespace(mlp=SimpleNamespace(down_proj=torch.nn.Linear(10, 10)))
    inner = SimpleNamespace(layers=[block])
    m = SimpleNamespace(model=SimpleNamespace(language_model=inner),
                        config=SimpleNamespace(hidden_size=2560, intermediate_size=9216))
    try:
        steer.resolve_down_proj(m, 0)
    except RuntimeError as e:
        assert "weight layout mismatch" in str(e), str(e)
        return "wrong layout fails loudly"
    raise AssertionError("layout assert did not fire")


def t5_assert_fires_non_linear():
    block = SimpleNamespace(mlp=SimpleNamespace(down_proj="not-a-module"))
    inner = SimpleNamespace(layers=[block])
    m = SimpleNamespace(model=SimpleNamespace(language_model=inner),
                        config=SimpleNamespace(hidden_size=2560, intermediate_size=9216))
    try:
        steer.resolve_down_proj(m, 0)
    except RuntimeError as e:
        assert "not Linear" in str(e), str(e)
        return "non-Linear fails loudly"
    raise AssertionError("type assert did not fire")


def t5_resolves_good_shape():
    block = SimpleNamespace(mlp=SimpleNamespace(down_proj=torch.nn.Linear(9216, 2560)))
    inner = SimpleNamespace(layers=[block])
    m = SimpleNamespace(model=SimpleNamespace(language_model=inner),
                        config=SimpleNamespace(hidden_size=2560, intermediate_size=9216))
    assert steer.resolve_down_proj(m, 0) is block.mlp.down_proj
    return "correct [2560, 9216] Linear resolves"


def t5_recipe_bounds():
    steer.validate_recipe([RecipeRow(0, 0, 1.0), RecipeRow(31, 9215, -1.0)])
    for bad in (RecipeRow(32, 0, 1.0), RecipeRow(0, 9216, 1.0), RecipeRow(-1, 0, 1.0)):
        try:
            steer.validate_recipe([bad])
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} accepted")
    return "0-based bounds enforced"


# -- model-backed (opt-in) ----------------------------------------------------

def model_backed():
    from huggingface_hub import model_info

    preset = json.loads((ROOT / "presets" / "evil-qwen.json").read_text())
    rows = [RecipeRow(**r) for r in preset["rows"]]

    def m_ladder_stops_at_rung1():
        model, tok, rung_n, attempts = steer.load_with_ladder()
        assert rung_n == 1 and attempts[0]["ok"], attempts
        return attempts[0]["detail"], (model, tok)

    t0_all = __import__("time").perf_counter()
    detail, (model, tok) = m_ladder_stops_at_rung1()
    results.append(("m_ladder_stops_at_rung1", PASS, detail))

    def m_binding_and_dims():
        steer.check_preset_binding(preset, steer.MODEL_ID)
        steer.assert_text_dims(model)
        return "evil-qwen binds; dims 32/2560/9216"

    check("m_binding_and_dims", m_binding_and_dims)

    def m_path_and_colnorms():
        for r in rows:
            steer.resolve_down_proj(model, r.layer)
        out = []
        for (L, N), ref in steer.REF_COL_NORMS.items():  # noqa: N806
            cn = steer.column_norm(model, L, N)
            assert abs(cn - ref) / ref < 0.2, f"L{L}N{N}: {cn} vs ref {ref}"
            out.append(f"L{L}N{N}={cn:.4f}")
        return " ".join(out)

    check("m_path_and_colnorms", m_path_and_colnorms)

    def m_calibration_cache_hit():
        info = model_info(steer.MODEL_ID)
        key = steer.calibration_cache_key(
            steer.MODEL_ID, info.sha, tok, str(steer.DTYPE), steer.DEVICE)
        med, ci = steer.calibrate_layer_medians(
            model, tok, preset["calibration_prompts"], [13, 14],
            cache_path=ROOT / "docs" / "calibration.json", key=key)
        assert ci["cache_hit"] is True, ci
        stored = json.loads((ROOT / "docs" / "calibration.json").read_text())["medians"]
        assert abs(med[13] - stored["13"]) < 1e-9
        return f"hit L13={med[13]:.2f} L14={med[14]:.2f}"

    check("m_calibration_cache_hit", m_calibration_cache_hit)

    def m_hooks_fire():
        n = steer.hooks_fire_probe(model, tok)
        assert n >= 1, n
        return f"{n} canary calls on L0 down_proj"

    check("m_hooks_fire", m_hooks_fire)

    def m_dose0_noop():
        ids = tok("Hello.", return_tensors="pt")["input_ids"].to(steer.DEVICE)
        with torch.no_grad():
            base = model(ids).logits
        steer.register_steering_hooks(
            model, [RecipeRow(14, 8949, 0.0)], {14: 11.2})
        try:
            with torch.no_grad():
                steered = model(ids).logits
        finally:
            steer.remove_steering_hooks()
        assert torch.equal(base, steered), "dose-0 changed logits"
        return "dose-0 logits bit-identical"

    check("m_dose0_noop", m_dose0_noop)
    results.append(("m_wall", PASS,
                    f"{__import__('time').perf_counter() - t0_all:.0f}s total"))


def main():
    check("t2_order", t2_order)
    check("t2_rung2_exists_live", t2_rung2_exists_live)
    check("t2_rung3_unresolved_no_model", t2_rung3_unresolved_no_model)
    check("t2_preset_binding", t2_preset_binding)
    check("t2_recovery_text", t2_recovery_text)
    check("t2_all_fail_raises_with_recovery", t2_all_fail_raises_with_recovery)
    check("t4_key_shape", t4_key_shape)
    check("t4_stored_cache_valid", t4_stored_cache_valid)
    check("t4_isolation_assert", t4_isolation_assert)
    check("t4_empty_filter", t4_empty_filter)
    check("t4_short_guard", t4_short_guard)
    check("t4_cache_hit_no_model", t4_cache_hit_no_model)
    check("t5_assert_fires_no_path", t5_assert_fires_no_path)
    check("t5_assert_fires_bad_layout", t5_assert_fires_bad_layout)
    check("t5_assert_fires_non_linear", t5_assert_fires_non_linear)
    check("t5_resolves_good_shape", t5_resolves_good_shape)
    check("t5_recipe_bounds", t5_recipe_bounds)
    if "--with-model" in sys.argv:
        model_backed()
    width = max(len(n) for n, _, _ in results)
    failed = 0
    for name, status, detail in results:
        failed += status == FAIL
        first = detail.splitlines()[0] if detail else ""
        print(f"{status} {name:{width}s} {first}")
        if status == FAIL:
            print(detail)
    print(f"{len(results) - failed}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
