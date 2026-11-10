"""Settings: every environment read in one place (MOD phase 1).

Each module reads `get_settings().<name>`. `get_settings()` builds a
new frozen `Settings` from the current environment on every call.
Callers that patch the environment in tests keep the old behavior.
Module-level values (for example `server.NO_MODEL`) still read once
at import, exactly as before.

`ENV_VARS` is the contract with docs/reference/configuration.md.
A test fails when the two lists differ.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# (name, default text as documented) for every variable lobot reads.
ENV_VARS = (
    ("PORT", "7860"),
    ("LOBOT_HOST", "127.0.0.1"),
    ("LOBOT_PUBLIC_DEMO", "off"),
    ("LOBOT_ALLOW_RESEARCH", "off"),
    ("LOBOT_MODEL", "Qwen/Qwen3.5-4B"),
    ("MODEL_LOBOTOMY_REPLAY", "off"),
    ("MODEL_LOBOTOMY_REPLAY_STAGE_S", "0.4"),
    ("MODEL_LOBOTOMY_NO_MODEL", "off"),
    ("MODEL_LOBOTOMY_INTEGRATION", "off"),
    ("LOBOT_BACKEND", "torch"),
    ("LOBOT_LLAMA_QUANT", "f16"),
    ("LOBOT_LLAMAD_BIN", "third_party/lobot-llamad-build/lobot-llamad"),
    ("LOBOT_LLAMA_PREFIX_CACHE", "1"),
    ("LOBOT_CREATE_LANE", "torch"),
    ("LOBOT_CREATE_QUANT", "f16"),
    ("LOBOT_GPU_SLOTS", "1"),
    ("LOBOT_GPU_WAIT", "off"),
    ("LOBOT_GUARD_DEFAULT", "off"),
    ("LOBOT_GUARD_SCREENS", "refusal,honesty,privacy"),
    ("LOBOT_PRIVACY_RULE", "control"),
    ("META_API_KEY", "none"),
    ("LOBOT_MUSE_MODEL", "muse-spark-1.3"),
    ("LOBOT_JUDGE_MODEL", "muse-spark-1.3"),
    ("LOBOT_HOME", "~/.lobot"),
    ("LOBOT_AUDIT_LOG", "$LOBOT_HOME/audit.jsonl"),
    ("LOBOT_LLAMACPP_CLI", "llama-cli on PATH"),
    ("LOBOT_JOBS_DIR", "<repo>/jobs"),
    ("LOBOT_JUDGE", "local"),
    ("LOBOT_GUARD", "screens-v2"),
    ("LOBOT_DIRECTION", "diffmean"),
    ("LOBOT_DOSE_RULE", "fluency-first-v4"),
)


def _gpu_slots(raw: str | None) -> int:
    try:
        return max(0, int(raw if raw is not None else "1"))
    except ValueError:
        return 1


@dataclass(frozen=True)
class Settings:
    port: int = 7860
    host: str = "127.0.0.1"
    public_demo: bool = False
    allow_research: bool = False
    model: str = ""
    replay: bool = False
    replay_stage_s: float = 0.4
    no_model: bool = False
    integration: bool = False
    backend: str = "torch"
    llama_quant: str = "f16"
    llamad_bin: str | None = None
    llama_prefix_cache: bool = True
    create_lane: str = "torch"
    create_quant: str = "f16"
    gpu_slots: int = 1
    gpu_wait: bool = False
    guard_default: bool = False
    guard_screens: str | None = None
    privacy_rule: str | None = None
    meta_api_key: str = ""
    muse_model: str = ""
    judge_model: str = ""
    lobot_home: Path = Path.home() / ".lobot"
    audit_log: str | None = None
    llamacpp_cli: str | None = None
    jobs_dir: Path | None = None
    judge_kind: str = "local"
    guard_kind: str = "screens-v2"
    direction_kind: str = "diffmean"
    dose_rule_kind: str = "fluency-first-v4"


def load_settings(env=None) -> Settings:
    """Build one frozen Settings from `env` (default: os.environ)."""
    e = os.environ if env is None else env
    get = e.get
    home = Path(get("LOBOT_HOME", Path.home() / ".lobot"))
    return Settings(
        port=int(get("PORT", 7860)),
        host=get("LOBOT_HOST", "127.0.0.1"),
        public_demo=get("LOBOT_PUBLIC_DEMO") == "1",
        allow_research=get("LOBOT_ALLOW_RESEARCH") == "1",
        model=get("LOBOT_MODEL", ""),
        replay=bool(get("MODEL_LOBOTOMY_REPLAY")),
        replay_stage_s=float(get("MODEL_LOBOTOMY_REPLAY_STAGE_S", "0.4")),
        no_model=bool(get("MODEL_LOBOTOMY_NO_MODEL")),
        integration=get("MODEL_LOBOTOMY_INTEGRATION") == "1",
        backend=get("LOBOT_BACKEND", "torch"),
        llama_quant=get("LOBOT_LLAMA_QUANT", "f16"),
        llamad_bin=get("LOBOT_LLAMAD_BIN"),
        llama_prefix_cache=get("LOBOT_LLAMA_PREFIX_CACHE", "1") != "0",
        create_lane=get("LOBOT_CREATE_LANE", "torch"),
        create_quant=get("LOBOT_CREATE_QUANT", "f16"),
        gpu_slots=_gpu_slots(get("LOBOT_GPU_SLOTS")),
        gpu_wait=get("LOBOT_GPU_WAIT") == "1",
        guard_default=get("LOBOT_GUARD_DEFAULT") == "1",
        guard_screens=get("LOBOT_GUARD_SCREENS"),
        privacy_rule=get("LOBOT_PRIVACY_RULE"),
        meta_api_key=get("META_API_KEY", ""),
        muse_model=get("LOBOT_MUSE_MODEL", ""),
        judge_model=get("LOBOT_JUDGE_MODEL", ""),
        lobot_home=home,
        audit_log=get("LOBOT_AUDIT_LOG"),
        llamacpp_cli=get("LOBOT_LLAMACPP_CLI"),
        jobs_dir=Path(get("LOBOT_JOBS_DIR")) if get("LOBOT_JOBS_DIR") else None,
        judge_kind=get("LOBOT_JUDGE", "local"),
        guard_kind=get("LOBOT_GUARD", "screens-v2"),
        direction_kind=get("LOBOT_DIRECTION", "diffmean"),
        dose_rule_kind=get("LOBOT_DOSE_RULE", "fluency-first-v4"),
    )


def load_env_file(path: Path | None = None, env=None) -> list[str]:
    """Copy KEY=VALUE lines from `.env` into the environment.

    Only the server entry point calls this, before it reads any
    setting, so `.env` holds META_API_KEY without a shell export.
    A variable that is already set wins. Returns the names it set.
    `.env` is git-ignored; never commit it.
    """
    e = os.environ if env is None else env
    path = path or Path(__file__).resolve().parents[2] / ".env"
    if not path.is_file():
        return []
    done = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and key not in e:
            e[key] = value
            done.append(key)
    return done


def get_settings() -> Settings:
    """Current settings. Fresh on every call; see module docstring."""
    return load_settings()
