"""Signed patch registry, audit trail and behavior CI (1000X RG/AU/CI).

Keys, the trust list and the audit log live in ~/.lobot (override
with LOBOT_HOME / LOBOT_AUDIT_LOG). A registry is a directory.

Run:
    uv run python scripts/lobot_registry.py keygen
    uv run python scripts/lobot_registry.py publish REG knob.lobot --key ~/.lobot/keys/<id>.pem
    uv run python scripts/lobot_registry.py trust <key_id>
    uv run python scripts/lobot_registry.py list REG
    uv run python scripts/lobot_registry.py install REG <name|sha> DEST
    uv run python scripts/lobot_registry.py audit
    uv run python scripts/lobot_registry.py ci REG --quick   # loads weights

ci re-runs every entry's shipped evidence on the current model build
(LOBOT_BACKEND / LOBOT_LLAMA_QUANT pick the build) and exits 1 when
any entry fails its signature or its evidence interval.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model_lobotomy import audit  # noqa: E402
from model_lobotomy import registry as reg  # noqa: E402


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str), flush=True)


def cmd_ci(args) -> int:
    """Behavior CI: signatures + evidence for every entry, one model."""
    from model_lobotomy import engine as engine_mod
    from model_lobotomy import reverify
    entries = reg.load_index(args.registry)["entries"]
    if args.model:
        entries = [e for e in entries if e["model_id"] == args.model]
    if not entries:
        print("no entries to check", flush=True)
        return 2
    trusted = reg.load_trusted() if args.trusted_only else None
    eng, loaded_id, rows, failed = None, None, [], 0
    for e in sorted(entries, key=lambda e: e["model_id"]):
        row = {"name": e["name"], "model_id": e["model_id"]}
        try:
            reg.verify_entry(args.registry, e, trusted)
        except reg.RegistryError as err:
            row.update(ok=False, stage="signature", error=str(err))
            rows.append(row)
            failed += 1
            continue
        if eng is None or loaded_id != e["model_id"]:
            eng = engine_mod.get_engine()
            eng.load(e["model_id"])
            loaded_id = e["model_id"]
        path = Path(args.registry) / "patches" / f"{e['patch_sha256']}.lobot"
        code, res = reverify.run_verify_patch(path, quick=args.quick,
                                              engine=eng)
        ok = code == 0
        row.update(ok=ok, stage="evidence", backend=eng.backend,
                   **{k: res.get(k) for k in
                      ("shipped_ci", "measured_mean", "n", "error")})
        audit.record("verify", patch_sha256=e["patch_sha256"],
                     name=e["name"], ok=ok, backend=eng.backend,
                     measured_mean=res.get("measured_mean"),
                     shipped_ci=res.get("shipped_ci"))
        rows.append(row)
        failed += 0 if ok else 1
    _print({"ok": failed == 0, "failed": failed, "entries": rows})
    return 0 if failed == 0 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("keygen")
    p = sub.add_parser("trust")
    p.add_argument("key_id")
    p = sub.add_parser("publish")
    p.add_argument("registry")
    p.add_argument("patch")
    p.add_argument("--key", required=True)
    p.add_argument("--name")
    p = sub.add_parser("list")
    p.add_argument("registry")
    p = sub.add_parser("verify")
    p.add_argument("registry")
    p.add_argument("ref")
    p = sub.add_parser("install")
    p.add_argument("registry")
    p.add_argument("ref")
    p.add_argument("dest")
    p = sub.add_parser("uninstall")
    p.add_argument("path")
    sub.add_parser("audit")
    p = sub.add_parser("ci")
    p.add_argument("registry")
    p.add_argument("--model")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--trusted-only", action="store_true",
                   help="also fail entries from untrusted publishers")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "keygen":
            _print(reg.generate_key())
        elif args.cmd == "trust":
            _print(sorted(reg.trust(args.key_id)))
        elif args.cmd == "publish":
            _print(reg.publish(args.registry, args.patch,
                               reg.load_key(args.key), name=args.name))
        elif args.cmd == "list":
            _print([{k: e.get(k) for k in ("name", "trait", "model_id",
                                           "key_id", "evidence")}
                    for e in reg.load_index(args.registry)["entries"]])
        elif args.cmd == "verify":
            reg.verify_entry(args.registry,
                             reg.find(args.registry, args.ref), None)
            print("signature and patch OK", flush=True)
        elif args.cmd == "install":
            _print(reg.install(args.registry, args.ref, args.dest))
        elif args.cmd == "uninstall":
            reg.uninstall(args.path)
            print("removed", flush=True)
        elif args.cmd == "audit":
            h = audit.history()
            _print({"chain": h["chain"],
                    "active": {p: {"name": a.get("name"),
                                   "patch_sha256": a["patch_sha256"],
                                   "verifies": len(a["verifies"])}
                               for p, a in h["active"].items()}})
        elif args.cmd == "ci":
            return cmd_ci(args)
    except (reg.RegistryError, ValueError) as e:
        print(f"FAIL: {e}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
