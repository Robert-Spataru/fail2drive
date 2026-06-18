# team_code/autoagent0/autoagent0_carla.py

from __future__ import annotations
import sys
import os
import math
import logging
import numpy as np
import carla
import cv2
from pathlib import Path
from collections import deque
from typing import Any, Dict, List, Optional
from omegaconf import OmegaConf

# Add fail2drive root to path so sim/ and planners/ are importable
_F2D_ROOT = Path(__file__).parent.parent.parent
if str(_F2D_ROOT) not in sys.path:
    sys.path.insert(0, str(_F2D_ROOT))

# Add HUGSIM root for anything not copied over
_HUGSIM_ROOT = Path("/data/robert/HUGSIM")
if str(_HUGSIM_ROOT) not in sys.path:
    sys.path.insert(0, str(_HUGSIM_ROOT))

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

LOG = logging.getLogger(__name__)

EGO_HISTORY_FRAMES = 4
RULE_BASED_TOPK = 10
PRIVILEGED_AGENT_RADIUS_M = 50.0

# HUGSIM-compatible camera names and CARLA ego mounts (meters, degrees).
CAMERA_MOUNTS: Dict[str, Dict[str, float]] = {
    "CAM_FRONT": {"x": 0.7, "y": 0.0, "z": 1.6, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
    "CAM_FRONT_LEFT": {"x": 0.7, "y": -0.4, "z": 1.6, "roll": 0.0, "pitch": 0.0, "yaw": -55.0},
    "CAM_FRONT_RIGHT": {"x": 0.7, "y": 0.4, "z": 1.6, "roll": 0.0, "pitch": 0.0, "yaw": 55.0},
    "CAM_BACK": {"x": -1.0, "y": 0.0, "z": 1.6, "roll": 0.0, "pitch": 0.0, "yaw": 180.0},
    "CAM_BACK_LEFT": {"x": -0.7, "y": -0.4, "z": 1.6, "roll": 0.0, "pitch": 0.0, "yaw": -110.0},
    "CAM_BACK_RIGHT": {"x": -0.7, "y": 0.4, "z": 1.6, "roll": 0.0, "pitch": 0.0, "yaw": 110.0},
}

CAMERA_RIGS: Dict[str, List[str]] = {
    "front_only": ["CAM_FRONT"],
    "rap_4cam": [
        "CAM_BACK",
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
    ],
    "full_6cam": list(CAMERA_MOUNTS.keys()),
}

VIDEO_GRID_LAYOUT = [
    ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
    ["CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"],
]


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _camera_defaults(carla_cfg: Dict[str, Any]) -> Dict[str, float]:
    return {
        "width": float(carla_cfg.get("camera_width", 800)),
        "height": float(carla_cfg.get("camera_height", 600)),
        "fov": float(carla_cfg.get("camera_fov", 100.0)),
    }


def _build_camera_spec(cam_id: str, carla_cfg: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    defaults = _camera_defaults(carla_cfg)
    mount = dict(CAMERA_MOUNTS.get(cam_id, CAMERA_MOUNTS["CAM_FRONT"]))
    if overrides:
        mount.update({k: overrides[k] for k in overrides if overrides[k] is not None})
    return {
        "id": cam_id,
        "type": "sensor.camera.rgb",
        "x": float(mount.get("x", 0.7)),
        "y": float(mount.get("y", 0.0)),
        "z": float(mount.get("z", 1.6)),
        "roll": float(mount.get("roll", 0.0)),
        "pitch": float(mount.get("pitch", 0.0)),
        "yaw": float(mount.get("yaw", 0.0)),
        "width": int(overrides.get("width", defaults["width"]) if overrides else defaults["width"]),
        "height": int(overrides.get("height", defaults["height"]) if overrides else defaults["height"]),
        "fov": float(overrides.get("fov", defaults["fov"]) if overrides else defaults["fov"]),
    }


def resolve_camera_specs(carla_cfg: Dict[str, Any], *, attach_legacy: bool, planner_type: str, vlm_enabled: bool) -> List[Dict[str, Any]]:
    """Resolve which CARLA RGB cameras to attach from config."""
    explicit = carla_cfg.get("cameras")
    if explicit:
        specs = []
        for entry in explicit:
            if isinstance(entry, str):
                specs.append(_build_camera_spec(entry, carla_cfg))
            elif isinstance(entry, dict):
                cam_id = str(entry.get("id", entry.get("name", "CAM_FRONT")))
                specs.append(_build_camera_spec(cam_id, carla_cfg, entry))
        return specs

    rig_name = str(carla_cfg.get("camera_rig", "")).strip()
    if rig_name:
        if rig_name not in CAMERA_RIGS:
            raise ValueError(
                f"Unknown camera_rig={rig_name!r}. "
                f"Choose from {sorted(CAMERA_RIGS.keys())} or set carla.cameras explicitly."
            )
        return [_build_camera_spec(cam_id, carla_cfg) for cam_id in CAMERA_RIGS[rig_name]]

    recording_enabled = _coerce_bool((carla_cfg.get("recording") or {}).get("enabled", False))
    if attach_legacy or recording_enabled or vlm_enabled or planner_type in {"rap", "drivor"}:
        if planner_type in {"rap", "drivor"}:
            return [
                _build_camera_spec(cam_id, carla_cfg)
                for cam_id in CAMERA_RIGS["rap_4cam"]
            ]
        return [_build_camera_spec("CAM_FRONT", carla_cfg)]
    return []


def _rgb_from_input_data(input_data: Dict[str, Any], cam_id: str) -> Optional[np.ndarray]:
    if cam_id not in input_data:
        return None
    raw = np.array(input_data[cam_id][1])
    if raw.ndim != 3:
        return None
    if raw.shape[2] == 4:
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
    if raw.shape[2] == 3:
        return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    return raw


def _resize_for_grid(image: np.ndarray, target_height: int) -> np.ndarray:
    if image.shape[0] == target_height:
        return image
    width = max(1, int(round(image.shape[1] * (target_height / image.shape[0]))))
    return cv2.resize(image, (width, target_height), interpolation=cv2.INTER_LINEAR)


def _compose_grid_frame(obs_rgb: Dict[str, np.ndarray], layout: List[List[str]]) -> Optional[np.ndarray]:
    present = {name for name in obs_rgb if obs_rgb[name] is not None}
    needed = [name for row in layout for name in row if name in present]
    if not needed:
        return None
    target_height = max(obs_rgb[name].shape[0] for name in needed)
    rows = []
    for row_names in layout:
        tiles = []
        for name in row_names:
            if name not in present:
                continue
            tiles.append(_resize_for_grid(obs_rgb[name], target_height))
        if tiles:
            rows.append(np.concatenate(tiles, axis=1))
    if not rows:
        return None
    target_width = max(row.shape[1] for row in rows)
    padded_rows = []
    for row in rows:
        if row.shape[1] < target_width:
            pad = target_width - row.shape[1]
            row = np.pad(row, ((0, 0), (0, pad), (0, 0)), mode="constant")
        padded_rows.append(row)
    return np.concatenate(padded_rows, axis=0)





_PLAN2CONTROL = None


def _get_plan2control():
    global _PLAN2CONTROL
    if _PLAN2CONTROL is not None:
        return _PLAN2CONTROL

    import sys as _sys

    saved_path = list(_sys.path)
    saved_modules = {
        key: _sys.modules.pop(key)
        for key in list(_sys.modules)
        if key == "sim" or key.startswith("sim.")
    }
    try:
        _sys.path = [str(_HUGSIM_ROOT)] + [
            entry
            for entry in saved_path
            if entry not in ("", str(_F2D_ROOT))
        ]
        from sim.ilqr.lqr import plan2control

        _PLAN2CONTROL = plan2control
        return _PLAN2CONTROL
    finally:
        _sys.path[:] = saved_path
        _sys.modules.update(saved_modules)


def _traj_to_control(plan_traj: np.ndarray, info: Dict[str, Any]):
    """Convert HUGSIM-format plan [x_right, y_forward] to CARLA controls."""
    plan2control = _get_plan2control()
    plan_traj_stats = np.zeros((plan_traj.shape[0] + 1, 5))
    plan_traj_stats[1:, :2] = plan_traj[:, [1, 0]]
    prev_a, prev_b = 0.0, 0.0
    for i, (a, b) in enumerate(plan_traj):
        rot = np.arctan2(a - prev_a, b - prev_b)
        rot = np.where(rot > np.pi / 2, rot - np.pi, rot)
        rot = np.where(rot < -np.pi / 2, rot + np.pi, rot)
        plan_traj_stats[i + 1, 2] = rot
        prev_a, prev_b = a, b
    curr_stat = np.array(
        [0.0, 0.0, 0.0, info["ego_velo"], info["ego_steer"]]
    )
    return plan2control(plan_traj_stats, curr_stat)



def get_entry_point():
    return "AutoAgent0CarlaAgent"


class AutoAgent0CarlaAgent(AutonomousAgent):

    def setup(self, path_to_conf_file):
        self.track = Track.SENSORS
        raw_cfg = OmegaConf.load(path_to_conf_file)
        self._cfg = OmegaConf.to_container(raw_cfg, resolve=True)
        self._carla_cfg = self._cfg.get("carla", {})

        # Detect planner type from config keys
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

        # Route tracking
        self._route_cursor = 0
        self._frame_index = 0
        self._last_steer = 0.0
        self._info_history = deque(maxlen=EGO_HISTORY_FRAMES)
        self._vlm_selector = None
        self._vlm_selector_cfg = None

        # Planner-specific setup
        if self._planner_type == "rule_based":
            self._setup_rule_based()
        elif self._planner_type == "rap":
            self._setup_rap()
        elif self._planner_type == "drivor":
            self._setup_drivor()

        self._setup_cameras_and_recording()

    # ------------------------------------------------------------------
    # Rule-based setup
    # ------------------------------------------------------------------
    # In team_code/autoagent0_agent.py

    def _setup_rule_based(self):
        rb_cfg = self._cfg["rule_based"]
        repo_root = str(rb_cfg["repo_root"])

        # Set env vars exactly as launch.sh does
        os.environ["RULE_BASED_REPO_ROOT"] = repo_root
        os.environ["RULE_BASED_CONFIG"] = str(rb_cfg.get("config", ""))
        os.environ["RULE_BASED_DEVICE"] = str(rb_cfg.get("device", "cpu"))
        os.environ["RULE_BASED_PYTHON_BIN"] = str(
            rb_cfg.get("python_bin", sys.executable)
        )
        os.environ["PLANNER_CONFIG"] = str(rb_cfg.get("config", ""))

        # Add Rule-Planner repo to path (same as client.py does at module level)
        repo_root_path = Path(repo_root).resolve()
        if str(repo_root_path) not in sys.path:
            sys.path.insert(0, str(repo_root_path))

        # Import PrivilegedPlannerService exactly as client.py does
        try:
            from privileged_planner.service import PrivilegedPlannerService
        except ImportError as e:
            raise RuntimeError(
                f"PrivilegedPlannerService not found in {repo_root}. "
                f"Error: {e}"
            )

        # Load planner config yaml if specified (same logic as client.py main())
        planner_config = None
        config_path = rb_cfg.get("config", "").strip()
        if config_path:
            try:
                import yaml
                with open(config_path, "r") as f:
                    planner_config = yaml.safe_load(f)
                LOG.info("Loaded rule-based planner config from %s", config_path)
            except Exception as exc:
                LOG.warning(
                    "Failed to load rule-based config %s: %s; using None",
                    config_path, exc
                )

        # Initialize planner (same as client.py)
        self._rule_planner = PrivilegedPlannerService(config=planner_config)
        LOG.info("PrivilegedPlannerService initialized OK")

        # Determine output_num_poses from config (same as client.py)
        try:
            self._rule_based_output_num_poses = int(
                planner_config.get("horizon", 8)
                if planner_config and isinstance(planner_config, dict)
                else 8
            )
        except Exception:
            self._rule_based_output_num_poses = 8
        LOG.info(
            "Rule-based output_num_poses=%d", self._rule_based_output_num_poses
        )

        from planners.rule_based import client as rb_client
        from autoagent0.adapters.hugsim.geometry import info_to_pose

        self._rb_client = rb_client
        self._info_to_pose = info_to_pose

        # VLM selector (disabled by default for rule_based config)
        vlm_dict = rb_cfg.get("vlm", {})
        self._vlm_selector_cfg = self._build_vlm_selector_config_from_dict(
            vlm_dict
        )
        if self._vlm_selector_cfg.enabled:
            # Set VLM env vars so the subprocess worker can find the model
            os.environ["RULE_BASED_VLM_MODEL_ID"] = str(
                vlm_dict.get("model_id", "")
            )
            os.environ["RULE_BASED_VLM_DEVICE"] = str(
                vlm_dict.get("device", "auto")
            )
            os.environ["RULE_BASED_VLM_PYTHON_BIN"] = str(
                vlm_dict.get("python_bin", sys.executable)
            )
            os.environ["PLANNER_VLM_MODEL_ID"] = os.environ[
                "RULE_BASED_VLM_MODEL_ID"
            ]
            os.environ["PLANNER_VLM_DEVICE"] = os.environ[
                "RULE_BASED_VLM_DEVICE"
            ]
            os.environ["PLANNER_VLM_PYTHON_BIN"] = os.environ[
                "RULE_BASED_VLM_PYTHON_BIN"
            ]
            from planners.common.vlm_selector import VLMPlanSelector
            self._vlm_selector = VLMPlanSelector(
                self._vlm_selector_cfg, self._output_dir
            )
            self._vlm_selector.preload()
        else:
            self._vlm_selector = None
            LOG.info("VLM disabled for rule-based planner")

        self._init_selection_state()

    def _init_selection_state(self) -> None:
        self._previous_selected_plan = None
        self._previous_selected_pose = None
        self._previous_selected_score = None
        self._previous_selected_timestamp = None
        self._previous_selected_source = None

    def _apply_rule_based_merge_env(
        self,
        rb_merge_dict: Optional[Dict[str, Any]],
        python_bin: str,
        prefixes: tuple[str, ...],
    ) -> None:
        from planners.common.rule_based_env import build_prefixed_rule_based_env

        env_values = build_prefixed_rule_based_env(
            rb_merge_dict or {},
            planner_python_bin=str(python_bin),
            prefixes=prefixes,
        )
        for key, value in env_values.items():
            os.environ[str(key)] = str(value)

    def _setup_vlm_selector(self, vlm_dict: Optional[Dict[str, Any]]) -> None:
        self._vlm_selector_cfg = self._build_vlm_selector_config_from_dict(
            vlm_dict or {}
        )
        if self._vlm_selector_cfg.enabled:
            from planners.common.vlm_selector import VLMPlanSelector

            self._vlm_selector = VLMPlanSelector(
                self._vlm_selector_cfg, self._output_dir
            )
            self._vlm_selector.preload()
        else:
            self._vlm_selector = None
            LOG.info("VLM disabled")

    def _apply_plan_control(
        self,
        selected_plan: Optional[np.ndarray],
        info: Dict[str, Any],
        selected_score_raw: Optional[float],
        selected_source: str,
    ) -> carla.VehicleControl:
        if selected_plan is None or len(selected_plan) == 0:
            return self._brake_control()
        try:
            acc_cmd, steer_rate = _traj_to_control(selected_plan, info)
        except Exception:
            LOG.exception("traj2control failed")
            return self._brake_control()

        self._previous_selected_plan = np.asarray(
            selected_plan, dtype=np.float32
        ).copy()
        self._previous_selected_pose = self._info_to_pose(info)
        self._previous_selected_score = selected_score_raw
        self._previous_selected_timestamp = float(info.get("timestamp", 0.0))
        self._previous_selected_source = selected_source

        ctrl = carla.VehicleControl()
        ctrl.steer = float(np.clip(steer_rate, -1.0, 1.0))
        ctrl.throttle = float(np.clip(acc_cmd, 0.0, 1.0))
        ctrl.brake = float(np.clip(-acc_cmd, 0.0, 1.0))
        ctrl.hand_brake = False
        ctrl.manual_gear_shift = False
        self._last_steer = ctrl.steer
        return ctrl

    def _export_rap_env(self, rap_cfg: Dict[str, Any]) -> None:
        os.environ["RAP_REPO_ROOT"] = str(rap_cfg["repo_root"])
        os.environ["RAP_CHECKPOINT"] = str(rap_cfg["checkpoint"])
        os.environ["RAP_DEVICE"] = str(rap_cfg.get("device", "cuda"))
        os.environ["RAP_IMAGE_SCALE"] = str(rap_cfg.get("image_scale", 0.4))
        os.environ["RAP_PYTHON_BIN"] = str(
            rap_cfg.get("python_bin", sys.executable)
        )
        hf_home = Path(str(rap_cfg.get("hf_home", "/data/robert/models/hf")))
        hf_hub_cache = Path(str(rap_cfg.get("hf_hub_cache", hf_home / "hub")))
        hf_home.mkdir(parents=True, exist_ok=True)
        hf_hub_cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_home)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_hub_cache)
        os.environ["TRANSFORMERS_CACHE"] = str(
            rap_cfg.get("transformers_cache", hf_hub_cache)
        )
        if rap_cfg.get("hf_hub_offline") is not None:
            os.environ["HF_HUB_OFFLINE"] = "1" if _coerce_bool(
                rap_cfg.get("hf_hub_offline"), default=True
            ) else "0"
        if rap_cfg.get("transformers_offline") is not None:
            os.environ["TRANSFORMERS_OFFLINE"] = "1" if _coerce_bool(
                rap_cfg.get("transformers_offline"), default=True
            ) else "0"
        nuplan_dir = str(rap_cfg.get("nuplan_devkit_dir", "")).strip()
        if nuplan_dir and Path(nuplan_dir).exists():
            if nuplan_dir not in sys.path:
                sys.path.insert(0, nuplan_dir)

    def _export_drivor_env(self, drivor_cfg: Dict[str, Any]) -> None:
        os.environ["DRIVOR_REPO_ROOT"] = str(drivor_cfg["repo_root"])
        os.environ["DRIVOR_CHECKPOINT"] = str(drivor_cfg["checkpoint"])
        os.environ["DRIVOR_DEVICE"] = str(drivor_cfg.get("device", "cuda"))
        os.environ["DRIVOR_PYTHON_BIN"] = str(
            drivor_cfg.get("python_bin", sys.executable)
        )
        if drivor_cfg.get("dino"):
            os.environ["DRIVOR_DINO"] = str(drivor_cfg["dino"])
        if drivor_cfg.get("config"):
            os.environ["DRIVOR_CONFIG"] = str(drivor_cfg["config"])

    def _resolve_rule_based_merge(
        self, planner_cfg: Dict[str, Any], prefixes: tuple[str, ...]
    ):
        from planners.common.rule_based_provider import (
            resolve_rule_based_merge_config,
        )

        return resolve_rule_based_merge_config(
            planner_python_bin=str(
                planner_cfg.get("python_bin", sys.executable)
            ),
            prefixes=prefixes,
        )

    def _select_learned_plan(
        self,
        *,
        obs: Dict[str, Any],
        info: Dict[str, Any],
        scores: np.ndarray,
        proposals: np.ndarray,
        output_num_poses: int,
        build_candidate_rows_fn,
        adapter_cfg: Any,
        rule_based_merge_cfg: Any,
        privileged_agents: Optional[List[Dict[str, Any]]],
        learned_source_name: str,
        learned_default_source: str,
        score_fallback_key: str,
        planner_log_name: str,
        strict_learned_argmax_lookup: bool,
        q_key_prefix: bool,
        plain_result_fn,
    ) -> tuple[np.ndarray, Optional[float], str]:
        if not self._vlm_selector_cfg.enabled:
            plain_result = plain_result_fn(proposals, scores, output_num_poses)
            return (
                np.asarray(plain_result["selected_plan"], dtype=np.float32),
                float(plain_result.get("selected_score_raw", plain_result["selected_score"])),
                str(plain_result["selected_row"].get("source", learned_default_source)),
            )

        reserved_candidate_slots = (
            max(0, int(rule_based_merge_cfg.topk))
            if rule_based_merge_cfg.enabled
            and not self._vlm_selector_cfg.planner_gate_enabled
            else 0
        )
        learned_candidate_rows, allow_carry_previous = build_candidate_rows_fn(
            proposals=proposals,
            scores=scores,
            cfg=adapter_cfg,
            current_info=info,
            previous_selected_plan=self._previous_selected_plan,
            previous_selected_pose=self._previous_selected_pose,
            previous_selected_score=self._previous_selected_score,
            previous_selected_timestamp=self._previous_selected_timestamp,
            previous_selected_source=self._previous_selected_source,
            reserved_candidate_slots=reserved_candidate_slots,
        )
        rule_based_candidate_rows: List[Dict[str, Any]] = []
        if rule_based_merge_cfg.enabled:
            try:
                from autoagent0.experts.rule_based import (
                    build_rule_based_candidate_rows,
                    get_rule_based_proposals_and_scores,
                )

                rb_proposals, rb_scores, _ = get_rule_based_proposals_and_scores(
                    rule_based_merge_cfg,
                    obs=obs,
                    info=info,
                    info_history=self._info_history,
                    privileged_agents=privileged_agents,
                    output_num_poses=output_num_poses,
                    topk=rule_based_merge_cfg.topk,
                )
                rule_based_candidate_rows = build_rule_based_candidate_rows(
                    rb_proposals,
                    rb_scores,
                    output_num_poses=output_num_poses,
                    source_name=rule_based_merge_cfg.source_name,
                    topk=rule_based_merge_cfg.topk,
                )
            except Exception:
                LOG.exception(
                    "Failed to append rule-based merge candidates for %s",
                    planner_log_name,
                )

        from autoagent0.core.config import resolve_autoagent0_config

        autoagent0_cfg = resolve_autoagent0_config()
        if autoagent0_cfg.enabled:
            selection = self._autoagent0_runtime.select_final_actions_recovery_loop(
                frame_index=self._frame_index,
                camera_images=obs.get("rgb", {}),
                info=info,
                vlm_selector=self._vlm_selector,
                scores=scores,
                learned_candidate_rows=learned_candidate_rows,
                rule_based_candidate_rows=rule_based_candidate_rows,
                redesign_candidate_budget=autoagent0_cfg.redesign_candidate_budget,
                learned_source_name=learned_source_name,
                learned_default_source=learned_default_source,
                score_fallback_key=score_fallback_key,
                planner_log_name=planner_log_name,
                logger=LOG,
                strict_learned_argmax_lookup=strict_learned_argmax_lookup,
                fallback_mode=autoagent0_cfg.fallback_mode,
                max_redesign_attempts=autoagent0_cfg.max_redesign_attempts,
            )
        else:
            selection = self._autoagent0_runtime.select_final_actions(
                frame_index=self._frame_index,
                camera_images=obs.get("rgb", {}),
                info=info,
                vlm_selector=self._vlm_selector,
                scores=scores,
                learned_candidate_rows=learned_candidate_rows,
                rule_based_candidate_rows=rule_based_candidate_rows,
                rule_based_merge_enabled=rule_based_merge_cfg.enabled,
                planner_gate_enabled=self._vlm_selector_cfg.planner_gate_enabled,
                vlm_enabled=self._vlm_selector_cfg.enabled,
                display_default_trajectories=self._vlm_selector_cfg.display_default_trajectories,
                include_default_candidates=self._vlm_selector_cfg.include_default_candidates,
                allow_carry_previous=allow_carry_previous,
                previous_selected_source=self._previous_selected_source,
                learned_source_name=learned_source_name,
                learned_default_source=learned_default_source,
                score_fallback_key=score_fallback_key,
                planner_log_name=planner_log_name,
                logger=LOG,
                strict_learned_argmax_lookup=strict_learned_argmax_lookup,
                q_key_prefix=q_key_prefix,
            )

        selected_row = selection.selected_row
        selected_plan = np.asarray(selection.selected_plan, dtype=np.float32)
        selected_score_raw = float(selection.selected_score_raw)
        selected_source = str(selection.selected_source)
        if selected_row.get("source") == "carry_prev":
            selected_source = "carry_prev"
        return selected_plan, selected_score_raw, selected_source

    def _setup_rap(self):
        import torch

        rap_cfg = self._cfg["rap"]
        repo_root = Path(str(rap_cfg["repo_root"])).expanduser().resolve()
        checkpoint_path = Path(str(rap_cfg["checkpoint"])).expanduser().resolve()
        if not repo_root.exists():
            raise RuntimeError(f"RAP repo_root does not exist: {repo_root}")
        self._export_rap_env(rap_cfg)

        if not checkpoint_path.exists():
            raise RuntimeError(
                f"RAP checkpoint does not exist: {checkpoint_path}. "
                "Download RAP_DINO_navsimv2.ckpt from "
                "https://huggingface.co/Lanl11/RAP_ckpts"
            )
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from planners.rap import client as rap_client
        from autoagent0.adapters.hugsim.geometry import info_to_pose
        from autoagent0.core.runtime import AutoAgent0Runtime

        self._rap_client = rap_client
        self._info_to_pose = info_to_pose
        self._autoagent0_runtime = AutoAgent0Runtime(runtime_name="rap", logger=LOG)

        vlm_dict = rap_cfg.get("vlm", {})
        self._setup_vlm_selector(vlm_dict)
        self._apply_rule_based_merge_env(
            rap_cfg.get("rule_based_merge", {}),
            str(rap_cfg.get("python_bin", sys.executable)),
            ("PLANNER_RULE_BASED_", "RAP_RULE_BASED_"),
        )
        rule_based_merge = self._resolve_rule_based_merge(
            rap_cfg, ("PLANNER_RULE_BASED_", "RAP_RULE_BASED_")
        )

        device_name = str(rap_cfg.get("device", "cuda"))
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
            LOG.warning("CUDA unavailable for RAP; falling back to CPU")

        camera_order = rap_cfg.get("camera_order", rap_client.DEFAULT_CAM_ORDER)
        self._rap_adapter_cfg = rap_client.AdapterConfig(
            output_dir=self._output_dir,
            rap_repo_root=repo_root,
            checkpoint_path=checkpoint_path,
            camera_order=list(camera_order),
            image_scale=float(rap_cfg.get("image_scale", 0.4)),
            device=torch.device(device_name),
            debug_diagnostics=_coerce_bool(
                rap_cfg.get("debug_diagnostics", False)
            ),
            use_scene_rig_lidar2img=_coerce_bool(
                rap_cfg.get("use_scene_rig_lidar2img", True)
            ),
            output_num_poses=int(
                rap_cfg.get("output_num_poses", rap_client.DEFAULT_OUTPUT_POSES)
            ),
            vlm=self._vlm_selector_cfg,
            rule_based_merge=rule_based_merge,
        )

        LOG.info(
            "Loading RAP model repo=%s checkpoint=%s device=%s lidar2img=%s",
            repo_root,
            checkpoint_path,
            device_name,
            "scene_rig"
            if self._rap_adapter_cfg.use_scene_rig_lidar2img
            else "static_l2c",
        )
        self._rap_model = rap_client.load_rap_model(self._rap_adapter_cfg)
        LOG.info("RAP model loaded OK (output_num_poses=%d)", self._rap_adapter_cfg.output_num_poses)
        self._init_selection_state()

    def _setup_drivor(self):
        import torch
        from omegaconf import OmegaConf

        drivor_cfg = self._cfg["drivor"]
        repo_root = Path(str(drivor_cfg["repo_root"])).expanduser().resolve()
        checkpoint_path = Path(str(drivor_cfg["checkpoint"])).expanduser().resolve()
        if not repo_root.exists():
            raise RuntimeError(f"DrivoR repo_root does not exist: {repo_root}")
        self._export_drivor_env(drivor_cfg)

        if not checkpoint_path.exists():
            raise RuntimeError(
                f"DrivoR checkpoint does not exist: {checkpoint_path}. "
                "Download drivor_Nav1_25epochs.pth from "
                "https://github.com/valeoai/DrivoR/releases/tag/model_weights"
            )
        dino_weights = Path(str(drivor_cfg.get("dino", ""))).expanduser() / "model.safetensors"
        if drivor_cfg.get("dino") and not dino_weights.exists():
            raise RuntimeError(
                f"DrivoR DINO weights missing: {dino_weights}. "
                "Download timm/vit_small_patch14_reg4_dinov2.lvd142m from Hugging Face."
            )
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from navsim.agents.drivoR.drivor_agent import DrivoRAgent
        from planners.drivor import client as drivor_client
        from autoagent0.adapters.hugsim.geometry import info_to_pose
        from autoagent0.core.runtime import AutoAgent0Runtime

        self._drivor_client = drivor_client
        self._info_to_pose = info_to_pose
        self._autoagent0_runtime = AutoAgent0Runtime(
            runtime_name="drivor", logger=LOG
        )

        vlm_dict = drivor_cfg.get("vlm", {})
        self._setup_vlm_selector(vlm_dict)
        self._apply_rule_based_merge_env(
            drivor_cfg.get("rule_based_merge", {}),
            str(drivor_cfg.get("python_bin", sys.executable)),
            ("PLANNER_RULE_BASED_", "DRIVOR_RULE_BASED_"),
        )
        self._drivor_rule_based_merge = self._resolve_rule_based_merge(
            drivor_cfg, ("PLANNER_RULE_BASED_", "DRIVOR_RULE_BASED_")
        )

        device_name = str(drivor_cfg.get("device", "cuda"))
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
            LOG.warning("CUDA unavailable for DrivoR; falling back to CPU")
        self._drivor_device = torch.device(device_name)

        config_path = str(drivor_cfg.get("config", "")).strip()
        drivo_config: Any = {}
        if config_path:
            loaded = OmegaConf.load(config_path)
            if "num_poses" in loaded:
                drivo_config = loaded
            elif "config" in loaded:
                drivo_config = loaded.config
            else:
                drivo_config = loaded

        dino_dir = Path(str(drivor_cfg.get("dino", ""))).expanduser()
        if dino_dir.exists() and hasattr(drivo_config, "image_backbone"):
            safetensors = dino_dir / "model.safetensors"
            if safetensors.exists():
                drivo_config.image_backbone.model_weights = str(safetensors)

        lr_args = {"name": "AdamW", "base_lr": 5e-4, "base_batch_size": 64}
        self._drivor_agent = DrivoRAgent(
            config=drivo_config,
            lr_args=lr_args,
            checkpoint_path=str(checkpoint_path),
            progress_bar=False,
        )
        original_cwd = os.getcwd()
        try:
            os.chdir(repo_root)
            self._drivor_agent.initialize()
        finally:
            os.chdir(original_cwd)
        try:
            self._drivor_agent._drivor_model.to(self._drivor_device)
            self._drivor_agent._drivor_model.eval()
        except Exception:
            LOG.warning("Could not move DrivoR model to %s", self._drivor_device)

        agent_config = getattr(self._drivor_agent, "_config", None)
        if agent_config is not None:
            self._drivor_output_num_poses = int(
                getattr(agent_config, "num_poses", drivor_cfg.get("output_num_poses", 8))
            )
        else:
            self._drivor_output_num_poses = int(drivor_cfg.get("output_num_poses", 8))

        LOG.info(
            "DrivoR agent initialized OK (output_num_poses=%d)",
            self._drivor_output_num_poses,
        )
        self._init_selection_state()

    def _run_step_rap(self, ego_state):
        import torch

        obs = ego_state["obs"]
        info = ego_state["info"]
        cfg = self._rap_adapter_cfg
        rap = self._rap_client

        for cam_name in cfg.camera_order:
            if cam_name not in obs.get("rgb", {}):
                LOG.warning("Missing camera %s for RAP inference", cam_name)
                return self._brake_control()

        privileged_agents = None
        if cfg.rule_based_merge.enabled and cfg.rule_based_merge.include_privileged_info:
            privileged_agents = self._get_privileged_info(info)

        try:
            features = rap.build_features(obs, list(self._info_history), cfg)
            with torch.no_grad():
                predictions = self._rap_model(
                    features, targets=None, return_score=True
                )
                scores = predictions["score"][0].detach().cpu().numpy()
                proposals = predictions["trajectory"][0].detach().cpu().numpy()
        except Exception:
            LOG.exception("RAP inference failed")
            return self._brake_control()

        try:
            selected_plan, selected_score_raw, selected_source = (
                self._select_learned_plan(
                    obs=obs,
                    info=info,
                    scores=scores,
                    proposals=proposals,
                    output_num_poses=cfg.output_num_poses,
                    build_candidate_rows_fn=rap.build_vlm_candidate_rows,
                    adapter_cfg=cfg,
                    rule_based_merge_cfg=cfg.rule_based_merge,
                    privileged_agents=privileged_agents,
                    learned_source_name="current_rap",
                    learned_default_source="fallback_rap_argmax",
                    score_fallback_key="rap_score",
                    planner_log_name="RAP",
                    strict_learned_argmax_lookup=True,
                    q_key_prefix=True,
                    plain_result_fn=lambda p, s, n: rap.build_plain_rap_plan_result(
                        p, s, cfg
                    ),
                )
            )
        except Exception:
            LOG.exception("RAP plan selection failed")
            return self._brake_control()

        return self._apply_plan_control(
            selected_plan, info, selected_score_raw, selected_source
        )

    def _run_step_drivor(self, ego_state):
        import torch

        obs = ego_state["obs"]
        info = ego_state["info"]
        drivor = self._drivor_client
        output_num_poses = self._drivor_output_num_poses

        for cam_name in drivor.MAP_HUGSIM_TO_DRIVOR:
            if cam_name not in obs.get("rgb", {}):
                LOG.warning("Missing camera %s for DrivoR inference", cam_name)
                return self._brake_control()

        privileged_agents = None
        if (
            self._drivor_rule_based_merge.enabled
            and self._drivor_rule_based_merge.include_privileged_info
        ):
            privileged_agents = self._get_privileged_info(info)

        try:
            agent_input = drivor.build_agent_input_from_hugsim(
                obs, list(self._info_history), num_history=EGO_HISTORY_FRAMES
            )
            features: Dict[str, Any] = {}
            for builder in self._drivor_agent.get_feature_builders():
                features.update(builder.compute_features(agent_input))

            features_batched: Dict[str, Any] = {}
            for key, value in features.items():
                if isinstance(value, torch.Tensor):
                    features_batched[key] = value.unsqueeze(0).to(self._drivor_device)
                else:
                    try:
                        tensor = torch.from_numpy(np.array(value))
                        features_batched[key] = tensor.unsqueeze(0).to(self._drivor_device)
                    except Exception:
                        features_batched[key] = value

            with torch.no_grad():
                try:
                    predictions = self._drivor_agent.forward(features_batched)
                except Exception:
                    LOG.exception("DrivoR agent.forward failed; trying internal model")
                    predictions = self._drivor_agent._drivor_model(features_batched)

            proposals, scores = drivor.extract_proposals_and_scores_from_predictions(
                predictions, output_num_poses=output_num_poses
            )
        except Exception:
            LOG.exception("DrivoR inference failed")
            return self._brake_control()

        class _DrivorAdapterCfg:
            def __init__(self, vlm_cfg, output_num_poses):
                self.vlm = vlm_cfg
                self.output_num_poses = output_num_poses

        adapter_cfg = _DrivorAdapterCfg(self._vlm_selector_cfg, output_num_poses)

        def _build_rows(proposals, scores, cfg, **kwargs):
            return drivor.build_drivor_candidate_rows(
                proposals=proposals,
                scores=scores,
                output_num_poses=output_num_poses,
                vlm_cfg=cfg.vlm,
                current_info=kwargs["current_info"],
                previous_selected_plan=kwargs.get("previous_selected_plan"),
                previous_selected_pose=kwargs.get("previous_selected_pose"),
                previous_selected_score=kwargs.get("previous_selected_score"),
                previous_selected_timestamp=kwargs.get("previous_selected_timestamp"),
                previous_selected_source=kwargs.get("previous_selected_source"),
                reserved_candidate_slots=kwargs.get("reserved_candidate_slots", 0),
            )

        try:
            selected_plan, selected_score_raw, selected_source = (
                self._select_learned_plan(
                    obs=obs,
                    info=info,
                    scores=scores,
                    proposals=proposals,
                    output_num_poses=output_num_poses,
                    build_candidate_rows_fn=_build_rows,
                    adapter_cfg=adapter_cfg,
                    rule_based_merge_cfg=self._drivor_rule_based_merge,
                    privileged_agents=privileged_agents,
                    learned_source_name="current_drivor",
                    learned_default_source="drivor_argmax",
                    score_fallback_key="proposal_score",
                    planner_log_name="DrivoR",
                    strict_learned_argmax_lookup=False,
                    q_key_prefix=False,
                    plain_result_fn=drivor.build_plain_drivor_plan_result,
                )
            )
        except Exception:
            LOG.exception("DrivoR plan selection failed")
            return self._brake_control()

        return self._apply_plan_control(
            selected_plan, info, selected_score_raw, selected_source
        )

    def _setup_cameras_and_recording(self):
        carla_cfg = self._carla_cfg
        attach_legacy = _coerce_bool(carla_cfg.get("attach_camera", False))
        vlm_enabled = bool(
            getattr(getattr(self, "_vlm_selector_cfg", None), "enabled", False)
        )
        self._camera_specs = resolve_camera_specs(
            carla_cfg,
            attach_legacy=attach_legacy,
            planner_type=self._planner_type,
            vlm_enabled=vlm_enabled,
        )

        rec_cfg = carla_cfg.get("recording") or {}
        self._recording_enabled = _coerce_bool(rec_cfg.get("enabled", False))
        self._recording_save_frames = _coerce_bool(
            rec_cfg.get("save_frames", True), default=True
        )
        self._recording_save_video = _coerce_bool(
            rec_cfg.get("save_video", True), default=True
        )
        self._recording_save_front_video = _coerce_bool(
            rec_cfg.get("save_front_video", True), default=True
        )
        self._recording_save_grid_video = _coerce_bool(
            rec_cfg.get("save_grid_video", True), default=True
        )
        self._recording_fps = float(rec_cfg.get("fps", 20.0))
        self._recording_frame_ext = str(
            rec_cfg.get("frame_format", "jpg")
        ).lstrip(".")

        dir_name = str(rec_cfg.get("dir_name", "recordings"))
        self._recording_dir = self._output_dir / dir_name
        self._video_buffers: Dict[str, List[np.ndarray]] = {}
        self._grid_video_buffer: List[np.ndarray] = []
        self._recording_finalized = False

        if self._recording_enabled and not self._camera_specs:
            self._camera_specs = [_build_camera_spec("CAM_FRONT", carla_cfg)]
            LOG.warning(
                "Recording enabled but no cameras configured; attaching CAM_FRONT"
            )

        if self._recording_enabled:
            self._recording_dir.mkdir(parents=True, exist_ok=True)
            if self._recording_save_frames:
                for spec in self._camera_specs:
                    (self._recording_dir / spec["id"]).mkdir(exist_ok=True)
            LOG.info(
                "Recording enabled -> %s (cameras=%s)",
                self._recording_dir,
                [spec["id"] for spec in self._camera_specs],
            )
        elif self._camera_specs:
            LOG.info(
                "Cameras attached (no recording): %s",
                [spec["id"] for spec in self._camera_specs],
            )

    def _maybe_record_frames(self, ego_state: Dict[str, Any]) -> None:
        if not getattr(self, "_recording_enabled", False):
            return

        obs_rgb = ego_state.get("obs", {}).get("rgb", {})
        if not obs_rgb:
            return

        frame_idx = self._frame_index
        for cam_id, rgb in obs_rgb.items():
            if self._recording_save_frames:
                frame_path = (
                    self._recording_dir
                    / cam_id
                    / f"{frame_idx:05d}.{self._recording_frame_ext}"
                )
                cv2.imwrite(
                    str(frame_path),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                )
            if self._recording_save_video:
                self._video_buffers.setdefault(cam_id, []).append(rgb.copy())

        if self._recording_save_grid_video and len(obs_rgb) > 1:
            grid = _compose_grid_frame(obs_rgb, VIDEO_GRID_LAYOUT)
            if grid is not None:
                self._grid_video_buffer.append(grid)
                if self._recording_save_frames:
                    grid_dir = self._recording_dir / "grid"
                    grid_dir.mkdir(exist_ok=True)
                    grid_path = (
                        grid_dir
                        / f"{frame_idx:05d}.{self._recording_frame_ext}"
                    )
                    cv2.imwrite(
                        str(grid_path),
                        cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
                    )

    def _write_rgb_video(
        self, out_path: Path, frames: List[np.ndarray], fps: float
    ) -> None:
        if not frames:
            return
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            LOG.warning("VideoWriter failed for %s", out_path)
            return
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

    def _finalize_recording(self) -> None:
        if getattr(self, "_recording_finalized", False):
            return
        self._recording_finalized = True
        if not getattr(self, "_recording_enabled", False):
            return
        if not getattr(self, "_recording_save_video", False):
            return

        fps = max(1.0, float(getattr(self, "_recording_fps", 20.0)))
        for cam_id, frames in self._video_buffers.items():
            if not frames:
                continue
            if cam_id == "CAM_FRONT" and self._recording_save_front_video:
                out_path = self._recording_dir / "front.mp4"
            else:
                out_path = self._recording_dir / f"{cam_id.lower()}.mp4"
            self._write_rgb_video(out_path, frames, fps)
            LOG.info("Wrote camera video: %s (%d frames)", out_path, len(frames))

        if self._grid_video_buffer:
            grid_path = self._recording_dir / "grid.mp4"
            self._write_rgb_video(grid_path, self._grid_video_buffer, fps)
            LOG.info(
                "Wrote grid video: %s (%d frames)",
                grid_path,
                len(self._grid_video_buffer),
            )

    # ------------------------------------------------------------------
    # sensors() — what CARLA should attach to the ego vehicle
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # run_step() — called every tick by the leaderboard evaluator
    # ------------------------------------------------------------------

    def run_step(self, input_data, timestamp):
        hero = self._find_hero()
        if hero is None:
            return self._brake_control()

        # Build shared ego state regardless of planner type
        ego_state = self._get_ego_state(hero, input_data, timestamp)
        self._info_history.append(ego_state["info"])
        while len(self._info_history) < EGO_HISTORY_FRAMES:
            self._info_history.appendleft(
                dict(self._info_history[0])
            )

        # Route cursor advance
        self._route_cursor = min(
            self._route_cursor + 1,
            max(0, len(self._global_plan_world_coord) - 1),
        )

        self._maybe_record_frames(ego_state)

        # Dispatch to planner-specific run
        if self._planner_type == "rule_based":
            ctrl = self._run_step_rule_based(ego_state)
        elif self._planner_type == "rap":
            ctrl = self._run_step_rap(ego_state)
        elif self._planner_type == "drivor":
            ctrl = self._run_step_drivor(ego_state)
        else:
            ctrl = self._brake_control()

        self._frame_index += 1
        return ctrl

    # ------------------------------------------------------------------
    # Rule-based run_step
    # ------------------------------------------------------------------

    def _run_step_rule_based(self, ego_state):

        obs = ego_state["obs"]
        info = ego_state["info"]
        privileged_agents = self._get_privileged_info(info)

        try:
            selected, planner_debug = self._rule_planner.process(
                obs=obs,
                info=info,
                info_history=self._info_history,
                privileged_agents=privileged_agents,
                k=RULE_BASED_TOPK,
            )
        except Exception:
            LOG.exception("Rule-based planner.process() failed")
            return self._brake_control()

        if not selected:
            LOG.warning("Rule-based planner returned no trajectories")
            return self._brake_control()

        rb = self._rb_client
        output_num_poses = self._rule_based_output_num_poses

        try:
            proposals = rb.trajectory_to_proposals(selected, output_num_poses)
            scores = rb.trajectory_to_scores(selected, planner_debug)
        except Exception:
            LOG.exception("Failed to convert rule-based trajectories")
            return self._brake_control()

        vlm_cfg = self._vlm_selector_cfg
        selected_plan = None
        selected_score_raw = None
        selected_source = "rule_based_argmax"

        if not vlm_cfg.enabled:
            plain_result = rb.build_plain_rule_based_plan_result(
                proposals, scores, output_num_poses
            )
            selected_plan = np.asarray(
                plain_result["selected_plan"], dtype=np.float32
            )
            selected_score_raw = float(
                plain_result.get("selected_score_raw", plain_result["selected_score"])
            )
            selected_source = "rule_based_argmax"
        else:
            try:
                candidate_rows, _allow_carry_prev = rb.build_rule_based_candidate_rows(
                    proposals=proposals,
                    scores=scores,
                    output_num_poses=output_num_poses,
                    vlm_cfg=vlm_cfg,
                    current_info=info,
                    previous_selected_plan=self._previous_selected_plan,
                    previous_selected_pose=self._previous_selected_pose,
                    previous_selected_score=self._previous_selected_score,
                    previous_selected_timestamp=self._previous_selected_timestamp,
                    previous_selected_source=self._previous_selected_source,
                )
                scores_arr = np.asarray(scores, dtype=np.float32)
                best_idx = int(np.argmax(scores_arr))
                default_selected_index = 0
                for idx, row in enumerate(candidate_rows):
                    proposal_index = row.get("proposal_index")
                    if proposal_index is not None and int(proposal_index) == best_idx:
                        default_selected_index = idx
                        break

                camera_images = obs.get("rgb", {}) if isinstance(obs, dict) else {}
                selection_result = self._vlm_selector.maybe_select(
                    frame_index=self._frame_index,
                    camera_images=camera_images,
                    info=info,
                    candidate_rows=candidate_rows,
                    default_selected_index=default_selected_index,
                    default_selected_source="rule_based_argmax",
                )
                selected_row = selection_result["selected_candidate_row"]
                selected_plan = np.asarray(
                    selected_row.get(
                        "execution_plan", selected_row["local_plan"]
                    ),
                    dtype=np.float32,
                )
                selected_score_raw = (
                    float(selected_row.get("origin_selected_score_raw"))
                    if selected_row.get("origin_selected_score_raw") is not None
                    else float(selected_row.get("proposal_score", 0.0))
                )
                selected_source = str(
                    selection_result.get("selected_source", "rule_based_vlm")
                )
            except Exception:
                LOG.exception("VLM selection failed; falling back to argmax")
                plain_result = rb.build_plain_rule_based_plan_result(
                    proposals, scores, output_num_poses
                )
                selected_plan = np.asarray(
                    plain_result["selected_plan"], dtype=np.float32
                )
                selected_score_raw = float(
                    plain_result.get("selected_score_raw", plain_result["selected_score"])
                )
                selected_source = "rule_based_argmax_fallback"

        if selected_plan is None or len(selected_plan) == 0:
            return self._brake_control()

        try:
            acc_cmd, steer_rate = _traj_to_control(selected_plan, info)
        except Exception:
            LOG.exception("traj2control failed")
            return self._brake_control()

        self._previous_selected_plan = selected_plan.copy()
        self._previous_selected_pose = self._info_to_pose(info)
        self._previous_selected_score = selected_score_raw
        self._previous_selected_timestamp = float(info.get("timestamp", 0.0))
        self._previous_selected_source = selected_source

        ctrl = carla.VehicleControl()
        ctrl.steer = float(np.clip(steer_rate, -1.0, 1.0))
        ctrl.throttle = float(np.clip(acc_cmd, 0.0, 1.0))
        ctrl.brake = float(np.clip(-acc_cmd, 0.0, 1.0))
        ctrl.hand_brake = False
        ctrl.manual_gear_shift = False
        self._last_steer = ctrl.steer
        return ctrl

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _brake_control(self) -> carla.VehicleControl:
        ctrl = carla.VehicleControl()
        ctrl.brake = 1.0
        return ctrl

    def _get_ego_state(self, hero, input_data, timestamp):
        transform = hero.get_transform()
        velocity = hero.get_velocity()
        speed_mps = math.sqrt(
            velocity.x**2 + velocity.y**2 + velocity.z**2
        )
        accel = hero.get_acceleration()
        forward = np.array([
            math.cos(math.radians(transform.rotation.pitch))
            * math.cos(math.radians(transform.rotation.yaw)),
            math.cos(math.radians(transform.rotation.pitch))
            * math.sin(math.radians(transform.rotation.yaw)),
            math.sin(math.radians(transform.rotation.pitch)),
        ])
        accel_fwd = float(
            np.dot([accel.x, accel.y, accel.z], forward)
        )
        command = self._get_route_command()

        obs: Dict[str, Any] = {"rgb": {}}
        front_rgb = None
        for spec in getattr(self, "_camera_specs", []):
            cam_id = spec["id"]
            rgb = _rgb_from_input_data(input_data, cam_id)
            if rgb is not None:
                obs["rgb"][cam_id] = rgb
                if cam_id == "CAM_FRONT":
                    front_rgb = rgb

        bbox = hero.bounding_box
        ego_box = np.array([
            transform.location.x,
            transform.location.y,
            transform.location.z,
            2.0 * bbox.extent.x,
            2.0 * bbox.extent.y,
            2.0 * bbox.extent.z,
            math.radians(transform.rotation.yaw),
        ], dtype=np.float32)

        info = {
            "timestamp": float(timestamp),
            "ego_pos": np.array([
                transform.location.x,
                transform.location.y,
                transform.location.z,
            ], dtype=np.float32),
            "ego_rot": np.radians(np.array([
                transform.rotation.roll,
                transform.rotation.pitch,
                transform.rotation.yaw,
            ], dtype=np.float32)),
            "ego_velo": float(speed_mps),
            "accelerate": float(accel_fwd),
            "ego_steer": float(
                getattr(hero.get_control(), "steer", self._last_steer)
            ),
            "command": command,
            "ego_box": ego_box,
            "task_instruction": {
                0: "right", 1: "left", 2: "straight"
            }.get(command, "straight"),
            "cam_params": self._build_cam_params(),
        }
        

        return {
            "info": info,
            "obs": obs,
            "front_rgb": front_rgb,
            "speed_mps": speed_mps,
            "transform": transform,
        }

    def _get_privileged_info(self, info: Dict[str, Any]) -> List[Dict[str, Any]]:
        world = CarlaDataProvider.get_world()
        if world is None:
            return []

        ego = self._find_hero()
        ego_loc = ego.get_location() if ego else None
        timestamp = float(info.get("timestamp", 0.0))
        nearby: List[Dict[str, Any]] = []

        for actor in world.get_actors():
            type_id = actor.type_id
            if "vehicle" not in type_id and "walker" not in type_id:
                continue
            if ego is not None and actor.id == ego.id:
                continue

            loc = actor.get_location()
            if ego_loc is not None:
                dist = math.sqrt(
                    (loc.x - ego_loc.x) ** 2 + (loc.y - ego_loc.y) ** 2
                )
                if dist > PRIVILEGED_AGENT_RADIUS_M:
                    continue

            vel = actor.get_velocity()
            actor_tf = actor.get_transform()
            bbox = actor.bounding_box
            object_type = "pedestrian" if "walker" in type_id else "vehicle"

            nearby.append({
                "agent_id": str(actor.id),
                "timestamp": timestamp,
                "agent_pos_world": [
                    float(loc.x),
                    float(loc.y),
                    float(loc.z),
                ],
                "agent_heading": math.radians(actor_tf.rotation.yaw),
                "agent_velo": math.sqrt(
                    vel.x ** 2 + vel.y ** 2 + vel.z ** 2
                ),
                "agent_vel_vec": [float(vel.x), float(vel.y)],
                "agent_extent": [
                    float(2.0 * bbox.extent.x),
                    float(2.0 * bbox.extent.y),
                ],
                "object_type": object_type,
            })

        return nearby

    def _build_cam_params(self):
        cam_params: Dict[str, Dict[str, Any]] = {}
        specs = getattr(self, "_camera_specs", None)
        if not specs:
            specs = [_build_camera_spec("CAM_FRONT", self._carla_cfg)]

        for spec in specs:
            w = int(spec["width"])
            h = int(spec["height"])
            fov_rad = math.radians(float(spec["fov"]))
            front2cam = np.eye(4, dtype=np.float32)
            front2cam[0, 3] = float(spec["x"])
            front2cam[1, 3] = float(spec["y"])
            front2cam[2, 3] = float(spec["z"])
            intrinsic = {
                "H": h,
                "W": w,
                "cx": w / 2.0,
                "cy": h / 2.0,
                "fovx": fov_rad,
                "fovy": fov_rad,
            }
            single_cam = {
                "intrinsic": intrinsic,
                "front2cam": front2cam,
                "v2c": front2cam.copy(),
                "l2c": front2cam.copy(),
            }
            cam_params[spec["id"]] = single_cam
        return cam_params

    def _get_route_command(self) -> int:
        if not self._global_plan_world_coord:
            return 2
        start = max(0, self._route_cursor)
        end = min(len(self._global_plan_world_coord), start + 20)
        for _, option in self._global_plan_world_coord[start:end]:
            name = str(getattr(option, "name", option)).upper()
            if "LEFT" in name and "CHANGE" not in name:
                return 1
            if "RIGHT" in name and "CHANGE" not in name:
                return 0
        return 2

    def _find_hero(self):
        world = CarlaDataProvider.get_world()
        if world is None:
            return None
        for actor in world.get_actors():
            if actor.attributes.get("role_name") == "hero":
                return actor
        return None

    def destroy(self):
        try:
            self._finalize_recording()
        except Exception:
            LOG.exception("Failed to finalize recording")

        vlm_selector = getattr(self, "_vlm_selector", None)
        if vlm_selector is not None:
            try:
                vlm_selector.finalize()
            except Exception:
                pass

    def _build_vlm_selector_config_from_dict(
        self, vlm_dict: dict
    ) -> "VLMSelectorConfig":
        from planners.common.vlm_selector import VLMSelectorConfig
        return VLMSelectorConfig(
            enabled=bool(vlm_dict.get("enabled", False)),
            intervention_enabled=bool(
                vlm_dict.get("intervention_enabled", False)
            ),
            camera_mode=str(vlm_dict.get("camera_mode", "front_only")),
            intervention_camera_mode=str(
                vlm_dict.get("intervention_camera_mode", "front_only")
            ),
            scoring_camera_mode=str(
                vlm_dict.get("scoring_camera_mode", "front_only")
            ),
            backend=str(
                vlm_dict.get("backend", "local_transformers_subprocess")
            ),
            model_id=str(
                vlm_dict.get("model_id", "Qwen/Qwen3-VL-8B-Instruct")
            ),
            device=str(vlm_dict.get("device", "auto")),
            python_bin=str(vlm_dict.get("python_bin", sys.executable)),
            max_new_tokens=int(vlm_dict.get("max_new_tokens", 300)),
            intervention_max_new_tokens=int(
                vlm_dict.get("intervention_max_new_tokens", 120)
            ),
            candidate_limit=int(vlm_dict.get("candidate_limit", 10)),
            timeout_sec=float(vlm_dict.get("timeout_sec", 180.0)),
            intervention_timeout_sec=float(
                vlm_dict.get("intervention_timeout_sec", 180.0)
            ),
            save_debug_artifacts=bool(
                vlm_dict.get("save_debug_artifacts", True)
            ),
            debug_dir_name=str(vlm_dict.get("debug_dir_name", "vlm_debug")),
            carry_previous_enabled=bool(
                vlm_dict.get("carry_previous_enabled", True)
            ),
            carry_previous_min_path_m=float(
                vlm_dict.get("carry_previous_min_path_m", 0.5)
            ),
            carry_previous_min_points=int(
                vlm_dict.get("carry_previous_min_points", 2)
            ),
            q_enabled=bool(vlm_dict.get("q_enabled", True)),
            q_switch_margin=float(vlm_dict.get("q_switch_margin", 0.05)),
            q_weight_rap_score=float(
                vlm_dict.get("q_weight_rap_score", 0.55)
            ),
            q_weight_progress=float(
                vlm_dict.get("q_weight_progress", 0.30)
            ),
            q_weight_offcenter=float(
                vlm_dict.get("q_weight_offcenter", 0.10)
            ),
            q_weight_curvature=float(
                vlm_dict.get("q_weight_curvature", 0.08)
            ),
            q_weight_shortplan=float(
                vlm_dict.get("q_weight_shortplan", 0.18)
            ),
            q_carry_score_decay=float(
                vlm_dict.get("q_carry_score_decay", 0.0)
            ),
            display_default_trajectories=bool(
                vlm_dict.get("display_default_trajectories", False)
            ),
            include_default_candidates=bool(
                vlm_dict.get("include_default_candidates", False)
            ),
            planner_gate_enabled=bool(
                vlm_dict.get("planner_gate_enabled", False)
            ),
            planner_gate_camera_mode=str(
                vlm_dict.get("planner_gate_camera_mode", "")
            ),
            planner_gate_max_new_tokens=int(
                vlm_dict.get("planner_gate_max_new_tokens", 120)
            ),
            planner_gate_timeout_sec=float(
                vlm_dict.get("planner_gate_timeout_sec", 180.0)
            ),
            planner_gate_default_planner=str(
                vlm_dict.get("planner_gate_default_planner", "learned")
            ),
            planner_gate_save_debug_artifacts=bool(
                vlm_dict.get("planner_gate_save_debug_artifacts", True)
            ),
        )