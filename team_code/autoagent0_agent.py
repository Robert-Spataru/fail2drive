"""Thin shim so leaderboard can load AutoAgent0 from team_code/autoagent0_agent.py."""
import sys
from pathlib import Path

_AGENT_DIR = Path(__file__).resolve().parent / "autoagent0"
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from autoagent0_carla import AutoAgent0CarlaAgent, get_entry_point

__all__ = ["AutoAgent0CarlaAgent", "get_entry_point"]
