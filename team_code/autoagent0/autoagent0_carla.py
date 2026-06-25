# team_code/autoagent0/autoagent0_carla.py

from __future__ import annotations
import sys
import logging
from collections import deque
from pathlib import Path
from omegaconf import OmegaConf

# Add fail2drive root to path so sim/ and planners/ are importable
_F2D_ROOT = Path(__file__).parent.parent.parent
if str(_F2D_ROOT) not in sys.path:
    sys.path.insert(0, str(_F2D_ROOT))

# Add HUGSIM root for anything not copied over
_AUTOAGENT0_ROOT = Path("/data/robert/AutoAgent0")
if str(_AUTOAGENT0_ROOT) not in sys.path:
    sys.path.insert(0, str(_AUTOAGENT0_ROOT))

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from autoagent0_carla_helper import (
    EGO_HISTORY_FRAMES,
    brake_control,
    finalize_recording,
    find_hero,
    get_ego_state,
    maybe_record_frames,
    record_post_step_visualization,
    run_step_drivor,
    run_step_rap,
    run_step_rule_based,
    setup_cameras_and_recording,
    setup_drivor,
    setup_rap,
    setup_rule_based,
)

LOG = logging.getLogger(__name__)


def get_entry_point():
    return "AutoAgent0CarlaAgent"


class AutoAgent0CarlaAgent(AutonomousAgent):

    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS
        raw_cfg = OmegaConf.load(path_to_conf_file)
        self._cfg = OmegaConf.to_container(raw_cfg, resolve=True)
        self._carla_cfg = self._cfg.get("carla", {})

        if "rule_based" in self._cfg:
            self._planner_type = "rule_based"
        elif "rap" in self._cfg:
            self._planner_type = "rap"
        elif "drivor" in self._cfg:
            self._planner_type = "drivor"
        else:
            raise ValueError(
                f"Config must have a 'rule_based', 'rap', or "
                f"'drivor' top-level key"
            )
        LOG.info("AutoAgent0: planner_type=%s", self._planner_type)

        output_dir = Path(
            self._carla_cfg.get("output_dir", "/tmp/fail2drive_autoagent0")
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir = output_dir

        self._route_cursor = 0
        self._frame_index = 0
        self._last_steer = 0.0
        self._info_history = deque(maxlen=EGO_HISTORY_FRAMES)
        self._vlm_selector = None
        self._vlm_selector_cfg = None

        if self._planner_type == "rule_based":
            setup_rule_based(self)
        elif self._planner_type == "rap":
            setup_rap(self)
        elif self._planner_type == "drivor":
            setup_drivor(self)
        else:
            raise ValueError(
                f"Invalid planner type: {self._planner_type}"
            )

        setup_cameras_and_recording(self)

    def sensors(self):
        sensors = []
        for spec in getattr(self, "_camera_specs", []):
            sensors.append({
                "type": spec["type"],
                "x": spec["x"],
                "y": spec["y"],
                "z": spec["z"],
                "roll": spec["roll"],
                "pitch": spec["pitch"],
                "yaw": spec["yaw"],
                "width": spec["width"],
                "height": spec["height"],
                "fov": spec["fov"],
                "id": spec["id"],
            })

        sensors.append({
            "type": "sensor.speedometer",
            "reading_frequency": 25,
            "id": "speedometer",
        })

        return sensors

    def run_step(self, input_data, timestamp):
        hero = find_hero(self)
        if hero is None:
            return brake_control()

        ego_state = get_ego_state(self, hero, input_data, timestamp)
        self._info_history.append(ego_state["info"])
        while len(self._info_history) < EGO_HISTORY_FRAMES:
            self._info_history.appendleft(
                dict(self._info_history[0])
            )

        self._route_cursor = min(
            self._route_cursor + 1,
            max(0, len(self._global_plan_world_coord) - 1),
        )

        maybe_record_frames(self, ego_state)

        if self._planner_type == "rule_based":
            ctrl = run_step_rule_based(self, ego_state)
        elif self._planner_type == "rap":
            ctrl = run_step_rap(self, ego_state)
        elif self._planner_type == "drivor":
            ctrl = run_step_drivor(self, ego_state)
        else:
            ctrl = brake_control()

        record_post_step_visualization(self, ego_state)
        self._frame_index += 1
        return ctrl

    def destroy(self):
        try:
            finalize_recording(self)
        except Exception:
            LOG.exception("Failed to finalize recording")

        vlm_selector = getattr(self, "_vlm_selector", None)
        if vlm_selector is not None:
            try:
                vlm_selector.finalize()
            except Exception:
                pass
