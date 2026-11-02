"""Create stage order: the one place that lists the stages.

Each stage is a function run(ctx) -> ctx in trait_eval. A stage
may set ctx["stop"] to end the run early (the no_knob exit).
`create_job.STAGES` and the UI stage list must match this order.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    name: str
    label: str


STAGES = (
    Stage("extract", "Contrast prompts and capture"),
    Stage("tune", "Tune layer and dose"),
    Stage("read_gate", "Read check"),
    Stage("final", "Final check"),
    Stage("baseline", "Prompt baseline"),
    Stage("pareto", "Compare with a prompt"),
    Stage("guard", "Guard"),
    Stage("verdict", "Report card"),
)

STAGE_NAMES = tuple(s.name for s in STAGES)
