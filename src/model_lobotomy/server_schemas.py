"""Request schemas for the HTTP API. One place for every body shape."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, StrictBool

from model_lobotomy import app as engine


def _guard_default() -> bool:
    """Default for "guard" in run/update/create (S1 demo default)."""
    from model_lobotomy.config import get_settings
    return get_settings().guard_default


class Row(BaseModel):
    layer: int
    neuron: int
    dose: float = Field(ge=engine.DOSE_MIN, le=engine.DOSE_MAX)


class RunRequest(BaseModel):
    prompt: str
    rows: list[Row]
    guard: bool = Field(default_factory=_guard_default)
    research_mode: bool = False
    dial_trait: str | None = None


class ModelRequest(BaseModel):
    model_id: str


class CreateRequest(BaseModel):
    description: str
    seed: int = 0
    guard: bool = Field(default_factory=_guard_default)
    research_mode: bool = False
    schedule: Literal["fast", "full"] = "full"
    # None: the PP stage runs on the full schedule and not on fast.
    pareto: bool | None = None


class DraftRequest(BaseModel):
    description: str
    seed: int = 0
    # "muse": Muse Spark (Meta model API) writes the parts; "local":
    # the loaded model does (C3 default).
    drafter: Literal["local", "muse"] = "local"


class AgentRequest(BaseModel):
    description: str
    lane: Literal["torch", "llama"] = "torch"
    max_rounds: int = Field(3, ge=1, le=5)


class LabelRow(BaseModel):
    reply_id: str
    label: StrictBool


class LabelsRequest(BaseModel):
    labels: list[LabelRow] = Field(min_length=1)


class UpdateRequest(BaseModel):
    prompt: str
    trait: str | None = None
    layer: int | None = None
    direction: list[float] | None = None
    scale: float | None = None
    doses: list[float]
    max_new_tokens: int = 60
    guard: bool = Field(default_factory=_guard_default)
    research_mode: bool = False


class NeuronsRequest(BaseModel):
    prompt: str
    rows: list[Row] = []
    trait: str | None = None
    dose: float | None = None
    token_index: int | None = None
    reply_char: int | None = None
    top_k: int = 8
    window: int = 4
    guard: bool = False
    research_mode: bool = False
    reply: str | None = None


class LiftRequest(BaseModel):
    layer: int
    neuron: int
    dose: float = Field(ge=engine.DOSE_MIN, le=engine.DOSE_MAX)
    words: list[str] = Field(min_length=1)
    prompts: list[str] | None = None


class FeatureLiftRequest(BaseModel):
    layer: int
    index: int
    dose: float = Field(ge=engine.DOSE_MIN, le=engine.DOSE_MAX)
    words: list[str] | None = None
    prompts: list[str] | None = None


class DeleteRequest(BaseModel):
    prompt: str
    trait: str
    max_new_tokens: int = 150
    guard: bool = False
    research_mode: bool = False


class InstallRequest(BaseModel):
    trait_id: str = ""
    target: str = ""
    model_ref: str = ""


class CombineCheckRequest(BaseModel):
    patch_ids: list = []
    max_new_tokens: int = 24
    seed: int = 0


class VerifyRequest(BaseModel):
    trait_id: str = ""
    quick: bool = True


class InstallVerifyRequest(BaseModel):
    prompt: str = ""
    judge: str | None = None
    max_new_tokens: int = 64
    # > 0: judge the first N prompts of the patch's own final split
    # (where its shipped interval came from) and compare their mean
    # gain; needs judge "local". The prompt field is then unused.
    judged_n: int = 0


# A judged verify never needs more than the final split (30 prompts).
INSTALL_JUDGED_MAX = 30


class DeleteCheckRequest(BaseModel):
    trait: str
    prompts: list[str] | None = None


class ChatStartRequest(BaseModel):
    rows: list[Row] = []
    trait: str | None = None
    dose: float | None = None
    max_new_tokens: int = 150
    guard: bool = False
    research_mode: bool = False
    dial_traits: list[str] = []


class ReplayTurn(BaseModel):
    message: str
    rows: list[Row] = []


class ReplayRequest(BaseModel):
    transcript: dict


class ChatRequest(BaseModel):
    session_id: str
    message: str
    rows: list[Row] | None = None
    max_new_tokens: int | None = None



class FollowupRequest(BaseModel):
    method: Literal["gradient", "project"] = "gradient"


class TuneRequest(BaseModel):
    goal: str
    patch_id: str


