"""Shared server state. AppState holds the state itself.

Routers and helpers read STATE. Tests patch STATE fields.
No module global in server.py holds this state.
"""
from __future__ import annotations

import threading
from pathlib import Path

from model_lobotomy.config import get_settings

ROOT = Path(__file__).resolve().parent.parent.parent
WEB = ROOT / "web"

try:
    from model_lobotomy import reader as _reader
except ImportError:
    _reader = None


class AppState:
    """The server's shared state, held as plain fields."""

    def __init__(self):
        settings = get_settings()
        self.settings = settings
        self.no_model = settings.no_model
        self.replay = settings.replay
        self.replay_stage_s = settings.replay_stage_s
        self.root = ROOT
        self.web = WEB
        self.reader = _reader
        self.cancel = threading.Event()
        self.runner = None
        self.followups = None
        self.agents = {}
        self.agent_lock = threading.Lock()
        self.chats = {}
        self.sae_cache = {}
        self.install_lock = threading.Lock()

    def get_runner(self):
        if self.runner is None:
            from model_lobotomy import create_job
            from model_lobotomy.engine import get_engine
            self.runner = create_job.CreateRunner(engine=get_engine())
        return self.runner

    def get_followups(self):
        if self.followups is None:
            from model_lobotomy import followups as followups_mod
            from model_lobotomy.engine import get_engine
            runner = self.get_runner()
            self.followups = followups_mod.FollowupRunner(
                runner._engine or get_engine(), lane_fn=runner._lane)
        return self.followups


STATE = AppState()
