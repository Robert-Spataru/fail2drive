# team_code/autoagent0/autoagent0_carla_helper.py

from __future__ import annotations

import logging
import math
import os
import sys
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import carla
import cv2
import numpy as np

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

LOG = logging.getLogger(__name__)

_F2D_ROOT = Path(__file__).parent.parent.parent
_AUTOAGENT0_ROOT = Path("/data/robert/AutoAgent0")

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
    "4cam": [
        "CAM_BACK",
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
    ],
    "full_6cam": list(CAMERA_MOUNTS.keys()),
    "rap_4cam": [
        "CAM_BACK",
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
    ],
}

VIDEO_GRID_LAYOUT = [
    ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
    ["CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"],
]

TOPDOWN_CAMERA_ID = "CAM_TOPDOWN"
TOPDOWN_VIZ_TOPK = 5
TOPDOWN_SELECTED_COLOR_BGR = (80, 255, 80)
TOPDOWN_CANDIDATE_COLOR_BGR = (200, 200, 200)
TOPDOWN_EGO_COLOR_BGR = (255, 255, 255)


def navsim_proposal_to_hugsim_plan(trajectory: np.ndarray) -> np.ndarray:
    traj = np.asarray(trajectory, dtype=np.float32)
    if traj.ndim == 1:
        traj = traj.reshape(1, -1)
    right = -traj[:, 1]
    forward = traj[:, 0]
    return np.stack([right, forward], axis=-1).astype(np.float32)


def build_step_viz_payload(
    *,
    selected_plan: np.ndarray,
    selected_source: str,
    selected_score: Optional[float],
    proposals: np.ndarray,
    scores: np.ndarray,
    output_num_poses: int,
    topk: int = TOPDOWN_VIZ_TOPK,
    proposals_already_hugsim: bool = False,
) -> Dict[str, Any]:
    proposals = np.asarray(proposals, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if proposals.ndim == 2:
        proposals = proposals[np.newaxis, ...]
        scores = scores.reshape(1)

    topk = max(1, min(int(topk), int(len(scores))))
    top_indices = np.argsort(scores)[-topk:][::-1]
    candidate_plans: List[np.ndarray] = []
    candidate_scores: List[float] = []
    for idx in top_indices:
        if proposals_already_hugsim:
            plan = np.asarray(proposals[int(idx)], dtype=np.float32)
        else:
            plan = navsim_proposal_to_hugsim_plan(
                proposals[int(idx), :output_num_poses]
            )
        candidate_plans.append(plan)
        candidate_scores.append(float(scores[int(idx)]))

    return {
        "selected_plan": np.asarray(selected_plan, dtype=np.float32),
        "selected_source": str(selected_source),
        "selected_score": (
            None if selected_score is None else float(selected_score)
        ),
        "candidate_plans": candidate_plans,
        "candidate_scores": candidate_scores,
    }


def resolve_predictions_dir(agent) -> Optional[Path]:
    raw = agent._carla_cfg.get("predictions_dir")
    if not raw:
        return None
    path = Path(str(raw)).expanduser()
    auto_run_id = coerce_bool(agent._carla_cfg.get("auto_run_id", True), default=True)
    if auto_run_id:
        rel = path.name if path.is_absolute() else str(path)
        if rel in {".", ""}:
            rel = "predictions"
        return agent._output_dir / rel
    if not path.is_absolute():
        path = agent._output_dir / path
    return path


def build_topdown_camera_spec(carla_cfg: Dict[str, Any]) -> Dict[str, Any]:
    td_cfg = carla_cfg.get("topdown_camera") or {}
    width = int(td_cfg.get("width", 800))
    height = int(td_cfg.get("height", width))
    return {
        "id": TOPDOWN_CAMERA_ID,
        "type": "sensor.camera.rgb",
        "x": float(td_cfg.get("x", 0.0)),
        "y": float(td_cfg.get("y", 0.0)),
        "z": float(td_cfg.get("z", 45.0)),
        "roll": float(td_cfg.get("roll", 0.0)),
        "pitch": float(td_cfg.get("pitch", -90.0)),
        "yaw": float(td_cfg.get("yaw", 0.0)),
        "width": width,
        "height": height,
        "fov": float(td_cfg.get("fov", 90.0)),
    }


def topdown_pixels_per_meter(width: int, fov_deg: float, altitude_m: float) -> float:
    ground_span_m = 2.0 * altitude_m * math.tan(math.radians(fov_deg * 0.5))
    return float(width) / max(ground_span_m, 1e-3)


def _plan_to_pixel_points(
    plan: np.ndarray,
    *,
    center_xy: Tuple[int, int],
    pixels_per_meter: float,
) -> np.ndarray:
    plan = np.asarray(plan, dtype=np.float32)
    if len(plan) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    cx, cy = center_xy
    xs = cx + plan[:, 0] * pixels_per_meter
    ys = cy - plan[:, 1] * pixels_per_meter
    return np.round(np.stack([xs, ys], axis=1)).astype(np.int32)


def overlay_trajectories_on_topdown(
    image_rgb: np.ndarray,
    viz_payload: Dict[str, Any],
    *,
    pixels_per_meter: float,
    ego_center_xy: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    height, width = img.shape[:2]
    center = ego_center_xy or (width // 2, height // 2)

    for plan in viz_payload.get("candidate_plans", []):
        pts = _plan_to_pixel_points(
            plan, center_xy=center, pixels_per_meter=pixels_per_meter
        )
        if len(pts) < 2:
            continue
        cv2.polylines(
            img,
            [pts],
            isClosed=False,
            color=TOPDOWN_CANDIDATE_COLOR_BGR,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    selected_plan = viz_payload.get("selected_plan")
    if selected_plan is not None and len(selected_plan) > 0:
        pts = _plan_to_pixel_points(
            selected_plan,
            center_xy=center,
            pixels_per_meter=pixels_per_meter,
        )
        if len(pts) >= 2:
            cv2.polylines(
                img,
                [pts],
                isClosed=False,
                color=TOPDOWN_SELECTED_COLOR_BGR,
                thickness=3,
                lineType=cv2.LINE_AA,
            )

    cv2.circle(
        img, center, 6, TOPDOWN_EGO_COLOR_BGR, thickness=-1, lineType=cv2.LINE_AA
    )

    label = str(viz_payload.get("selected_source", ""))
    score = viz_payload.get("selected_score")
    if score is not None:
        label = f"{label} ({float(score):.2f})"
    if label:
        cv2.putText(
            img,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            TOPDOWN_SELECTED_COLOR_BGR,
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_predictions_frame(
    agent,
    frame_idx: int,
    viz_payload: Dict[str, Any],
    info: Dict[str, Any],
) -> None:
    predictions_dir = getattr(agent, "_predictions_dir", None)
    if predictions_dir is None:
        return
    predictions_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "frame_index": int(frame_idx),
        "timestamp": float(info.get("timestamp", 0.0)),
        "selected_source": str(viz_payload.get("selected_source", "")),
        "selected_score": viz_payload.get("selected_score"),
        "selected_plan": np.asarray(
            viz_payload.get("selected_plan", []), dtype=np.float32
        ).tolist(),
        "candidates": [
            {
                "score": float(score),
                "plan": np.asarray(plan, dtype=np.float32).tolist(),
            }
            for plan, score in zip(
                viz_payload.get("candidate_plans", []),
                viz_payload.get("candidate_scores", []),
            )
        ],
    }
    out_path = predictions_dir / f"{frame_idx:05d}.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def record_post_step_visualization(agent, ego_state: Dict[str, Any]) -> None:
    viz_payload = getattr(agent, "_step_viz", None)
    agent._step_viz = None
    if viz_payload is None:
        return

    info = ego_state.get("info", {})
    frame_idx = agent._frame_index
    save_predictions_frame(agent, frame_idx, viz_payload, info)

    if not getattr(agent, "_recording_save_topdown_video", False):
        return

    topdown_rgb = ego_state.get("topdown_rgb")
    if topdown_rgb is None:
        spec = getattr(agent, "_topdown_camera_spec", None) or {}
        width = int(spec.get("width", 800))
        height = int(spec.get("height", width))
        topdown_rgb = np.full((height, width, 3), 32, dtype=np.uint8)

    overlay = overlay_trajectories_on_topdown(
        topdown_rgb,
        viz_payload,
        pixels_per_meter=float(
            getattr(agent, "_topdown_pixels_per_meter", 10.0)
        ),
    )
    agent._topdown_video_buffer.append(overlay)



def _sanitize_run_id(run_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(run_id).strip())
    return cleaned or "run"


def resolve_run_output_dir(
    carla_cfg: Dict[str, Any],
    *,
    base_output_dir: Path,
    planner_type: str,
    config_path: Optional[str] = None,
) -> Path:
    """Create a unique per-run output directory under base_output_dir/runs/."""
    auto_run_id = coerce_bool(carla_cfg.get("auto_run_id", True), default=True)
    if not auto_run_id:
        base_output_dir.mkdir(parents=True, exist_ok=True)
        return base_output_dir

    run_id = carla_cfg.get("run_id")
    if not run_id:
        run_id = f"{planner_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_id = _sanitize_run_id(str(run_id))

    runs_parent = str(carla_cfg.get("run_dir_name", "runs")).strip() or "runs"
    run_output_dir = base_output_dir / runs_parent / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "planner_type": planner_type,
        "base_output_dir": str(base_output_dir),
        "run_output_dir": str(run_output_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": config_path,
    }
    manifest_path = run_output_dir / "run_info.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    LOG.info("Run output directory: %s (run_id=%s)", run_output_dir, run_id)
    return run_output_dir


def coerce_bool(value: Any, default: bool = False) -> bool:
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

    recording_enabled = coerce_bool((carla_cfg.get("recording") or {}).get("enabled", False))
    if attach_legacy or recording_enabled or vlm_enabled or planner_type in {"rap", "drivor"}:
        if planner_type in {"rap", "drivor"}:
            return [
                _build_camera_spec(cam_id, carla_cfg)
                for cam_id in CAMERA_RIGS["4cam"]
            ]
        return [_build_camera_spec("CAM_FRONT", carla_cfg)]
    return []


def rgb_from_input_data(input_data: Dict[str, Any], cam_id: str) -> Optional[np.ndarray]:
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


def compose_grid_frame(obs_rgb: Dict[str, np.ndarray], layout: List[List[str]]) -> Optional[np.ndarray]:
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


def get_plan2control():
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
        _sys.path = [str(_AUTOAGENT0_ROOT)] + [
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


def traj_to_control(plan_traj: np.ndarray, info: Dict[str, Any]):
    """Convert HUGSIM-format plan [x_right, y_forward] to CARLA controls."""
    plan2control = get_plan2control()
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



def build_vlm_selector_config_from_dict(vlm_dict: dict) -> "VLMSelectorConfig":
    # importing from AutoAgent0 repo
    from autoagent0.decision.vlm_selector import VLMSelectorConfig
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

def setup_rule_based(agent):
    rb_cfg = agent._cfg["rule_based"]
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
    agent._rule_planner = PrivilegedPlannerService(config=planner_config)
    LOG.info("PrivilegedPlannerService initialized OK")

    # Determine output_num_poses from config (same as client.py)
    try:
        agent._rule_based_output_num_poses = int(
            planner_config.get("horizon", 8)
            if planner_config and isinstance(planner_config, dict)
            else 8
        )
    except Exception:
        agent._rule_based_output_num_poses = 8
    LOG.info(
        "Rule-based output_num_poses=%d", agent._rule_based_output_num_poses
    )

    from autoagent0.planners.rule_based import planner as rb_client
    from autoagent0.adapters.hugsim.geometry import info_to_pose

    agent._rb_client = rb_client
    agent._info_to_pose = info_to_pose

    # VLM selector (disabled by default for rule_based config)
    vlm_dict = rb_cfg.get("vlm", {})
    agent._vlm_selector_cfg = build_vlm_selector_config_from_dict(
        vlm_dict
    )
    if agent._vlm_selector_cfg.enabled:
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
        from autoagent0.decision.vlm_selector import VLMPlanSelector
        agent._vlm_selector = VLMPlanSelector(
            agent._vlm_selector_cfg, agent._output_dir
        )
        agent._vlm_selector.preload()
    else:
        agent._vlm_selector = None
        LOG.info("VLM disabled for rule-based planner")

    init_selection_state(agent)

def init_selection_state(agent) -> None:
    agent._previous_selected_plan = None
    agent._previous_selected_pose = None
    agent._previous_selected_score = None
    agent._previous_selected_timestamp = None
    agent._previous_selected_source = None

def apply_rule_based_merge_env(
    rb_merge_dict: Optional[Dict[str, Any]],
    python_bin: str,
    prefixes: tuple[str, ...],
) -> None:
    from autoagent0.experts.rule_based_env import build_prefixed_rule_based_env

    env_values = build_prefixed_rule_based_env(
        rb_merge_dict or {},
        planner_python_bin=str(python_bin),
        prefixes=prefixes,
    )
    for key, value in env_values.items():
        os.environ[str(key)] = str(value)

def export_hf_env(
    planner_cfg: Dict[str, Any],
    *,
    default_hf_home: str = "/data/robert/models/hf",
) -> None:
    """Set Hugging Face cache env vars for VLM / transformer model downloads."""
    hf_home = Path(str(planner_cfg.get("hf_home", default_hf_home))).expanduser()
    hf_hub_cache = Path(
        str(planner_cfg.get("hf_hub_cache", hf_home / "hub"))
    ).expanduser()
    hf_home.mkdir(parents=True, exist_ok=True)
    hf_hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(
        planner_cfg.get("transformers_cache", hf_hub_cache)
    )
    if planner_cfg.get("hf_hub_offline") is not None:
        os.environ["HF_HUB_OFFLINE"] = (
            "1" if coerce_bool(planner_cfg.get("hf_hub_offline"), default=True) else "0"
        )
    if planner_cfg.get("transformers_offline") is not None:
        os.environ["TRANSFORMERS_OFFLINE"] = (
            "1"
            if coerce_bool(planner_cfg.get("transformers_offline"), default=True)
            else "0"
        )


def resolve_vlm_subprocess_gpu_index(vlm_dict: Optional[Dict[str, Any]]) -> Optional[str]:
    if not vlm_dict:
        return None
    cuda_device = str(vlm_dict.get("cuda_device", "")).strip()
    if cuda_device:
        return cuda_device
    device = str(vlm_dict.get("device", "")).strip()
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    return None


def setup_vlm_selector(
    agent,
    vlm_dict: Optional[Dict[str, Any]],
    *,
    planner_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    agent._vlm_selector_cfg = build_vlm_selector_config_from_dict(
        vlm_dict or {}
    )
    if agent._vlm_selector_cfg.enabled:
        if planner_cfg is not None:
            export_hf_env(planner_cfg)
        if coerce_bool((vlm_dict or {}).get("force_cpu_offload", False), default=False):
            os.environ["PLANNER_VLM_FORCE_CPU_OFFLOAD"] = "1"
        else:
            os.environ.pop("PLANNER_VLM_FORCE_CPU_OFFLOAD", None)
        from autoagent0.vlm.backends import set_vlm_subprocess_cuda_visible_devices
        from autoagent0.decision.vlm_selector import VLMPlanSelector

        set_vlm_subprocess_cuda_visible_devices(
            resolve_vlm_subprocess_gpu_index(vlm_dict)
        )
        agent._vlm_selector = VLMPlanSelector(
            agent._vlm_selector_cfg, agent._output_dir
        )
        agent._vlm_selector.preload()
    else:
        agent._vlm_selector = None
        LOG.info("VLM disabled")

def apply_plan_control(
    agent,
    selected_plan: Optional[np.ndarray],
    info: Dict[str, Any],
    selected_score_raw: Optional[float],
    selected_source: str,
) -> carla.VehicleControl:
    if selected_plan is None or len(selected_plan) == 0:
        return brake_control()
    try:
        acc_cmd, steer_rate = traj_to_control(selected_plan, info)
    except Exception:
        LOG.exception("traj2control failed")
        return brake_control()

    agent._previous_selected_plan = np.asarray(
        selected_plan, dtype=np.float32
    ).copy()
    agent._previous_selected_pose = agent._info_to_pose(info)
    agent._previous_selected_score = selected_score_raw
    agent._previous_selected_timestamp = float(info.get("timestamp", 0.0))
    agent._previous_selected_source = selected_source

    ctrl = carla.VehicleControl()
    ctrl.steer = float(np.clip(steer_rate, -1.0, 1.0))
    ctrl.throttle = float(np.clip(acc_cmd, 0.0, 1.0))
    ctrl.brake = float(np.clip(-acc_cmd, 0.0, 1.0))
    ctrl.hand_brake = False
    ctrl.manual_gear_shift = False
    agent._last_steer = ctrl.steer
    return ctrl

def export_rap_env(rap_cfg: Dict[str, Any]) -> None:
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
        os.environ["HF_HUB_OFFLINE"] = "1" if coerce_bool(
            rap_cfg.get("hf_hub_offline"), default=True
        ) else "0"
    if rap_cfg.get("transformers_offline") is not None:
        os.environ["TRANSFORMERS_OFFLINE"] = "1" if coerce_bool(
            rap_cfg.get("transformers_offline"), default=True
        ) else "0"
    nuplan_dir = str(rap_cfg.get("nuplan_devkit_dir", "")).strip()
    if nuplan_dir and Path(nuplan_dir).exists():
        if nuplan_dir not in sys.path:
            sys.path.insert(0, nuplan_dir)

def export_drivor_env(drivor_cfg: Dict[str, Any]) -> None:
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

def resolve_rule_based_merge(
    planner_cfg: Dict[str, Any], prefixes: tuple[str, ...]
):
    from autoagent0.experts.rule_based_provider import (
        resolve_rule_based_merge_config,
    )

    return resolve_rule_based_merge_config(
        planner_python_bin=str(
            planner_cfg.get("python_bin", sys.executable)
        ),
        prefixes=prefixes,
    )

def create_learned_planner_selector(
    agent,
    *,
    rule_based_merge_cfg: Any,
    current_source_name: str,
    learned_default_source: str,
    plain_source: str,
    score_fallback_key: str,
    planner_log_name: str,
    strict_learned_argmax_lookup: bool,
    q_key_prefix: bool,
) -> Any:
    from autoagent0.agent.runtime import AutoAgent0Runtime
    from autoagent0.config import resolve_autoagent0_config
    from autoagent0.decision.planner_selection import LearnedPlannerSelector

    if not hasattr(agent, "_autoagent0_runtime") or agent._autoagent0_runtime is None:
        agent._autoagent0_runtime = AutoAgent0Runtime(
            runtime_name=planner_log_name.lower(),
            logger=LOG,
        )

    return LearnedPlannerSelector(
        vlm_selector=agent._vlm_selector,
        autoagent0_runtime=agent._autoagent0_runtime,
        autoagent0_cfg=resolve_autoagent0_config(),
        vlm_cfg=agent._vlm_selector_cfg,
        rule_based_merge_cfg=rule_based_merge_cfg,
        current_source_name=current_source_name,
        learned_default_source=learned_default_source,
        plain_source=plain_source,
        score_fallback_key=score_fallback_key,
        planner_log_name=planner_log_name,
        strict_learned_argmax_lookup=strict_learned_argmax_lookup,
        q_key_prefix=q_key_prefix,
        logger=LOG,
    )


def run_learned_selection(
    agent,
    *,
    proposals_hugsim: np.ndarray,
    scores: np.ndarray,
    obs: Dict[str, Any],
    info: Dict[str, Any],
    privileged_agents: Optional[List[Dict[str, Any]]],
) -> tuple[np.ndarray, Optional[float], str, Dict[str, Any]]:
    agent._learned_selector.frame_index = agent._frame_index
    plan_payload = agent._learned_selector.select(
        proposals=proposals_hugsim,
        scores=scores,
        obs=obs,
        info=info,
        info_history=list(agent._info_history),
        privileged_info=privileged_agents,
    )
    selected_plan = np.asarray(plan_payload["selected_plan"], dtype=np.float32)
    selected_score = plan_payload.get("selected_score")
    selected_score_raw = (
        None if selected_score is None else float(selected_score)
    )
    selected_source = str(plan_payload.get("selected_source", ""))
    return selected_plan, selected_score_raw, selected_source, plan_payload

def setup_rap(agent):
    import torch

    rap_cfg = agent._cfg["rap"]
    repo_root = Path(str(rap_cfg["repo_root"])).expanduser().resolve()
    checkpoint_path = Path(str(rap_cfg["checkpoint"])).expanduser().resolve()
    if not repo_root.exists():
        raise RuntimeError(f"RAP repo_root does not exist: {repo_root}")
    export_rap_env(rap_cfg)

    if not checkpoint_path.exists():
        raise RuntimeError(
            f"RAP checkpoint does not exist: {checkpoint_path}. "
            "Download RAP_DINO_navsimv2.ckpt from "
            "https://huggingface.co/Lanl11/RAP_ckpts"
        )
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from autoagent0.planners.rap import planner as rap_client
    from autoagent0.adapters.hugsim.geometry import info_to_pose

    agent._rap_client = rap_client
    agent._info_to_pose = info_to_pose

    apply_rule_based_merge_env(
        rap_cfg.get("rule_based_merge", {}),
        str(rap_cfg.get("python_bin", sys.executable)),
        ("PLANNER_RULE_BASED_", "RAP_RULE_BASED_"),
    )
    agent._rap_rule_based_merge = resolve_rule_based_merge(
        rap_cfg, ("PLANNER_RULE_BASED_", "RAP_RULE_BASED_")
    )

    device_name = str(rap_cfg.get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
        LOG.warning("CUDA unavailable for RAP; falling back to CPU")

    camera_order = rap_cfg.get("camera_order", rap_client.DEFAULT_CAM_ORDER)
    agent._rap_adapter_cfg = rap_client.AdapterConfig(
        output_dir=agent._output_dir,
        rap_repo_root=repo_root,
        checkpoint_path=checkpoint_path,
        camera_order=list(camera_order),
        image_scale=float(rap_cfg.get("image_scale", 0.4)),
        device=torch.device(device_name),
        debug_diagnostics=coerce_bool(
            rap_cfg.get("debug_diagnostics", False)
        ),
        use_scene_rig_lidar2img=coerce_bool(
            rap_cfg.get("use_scene_rig_lidar2img", True)
        ),
        output_num_poses=int(
            rap_cfg.get("output_num_poses", rap_client.DEFAULT_OUTPUT_POSES)
        ),
    )

    LOG.info(
        "Loading RAP model repo=%s checkpoint=%s device=%s lidar2img=%s",
        repo_root,
        checkpoint_path,
        device_name,
        "scene_rig"
        if agent._rap_adapter_cfg.use_scene_rig_lidar2img
        else "static_l2c",
    )
    agent._rap_model = rap_client.load_rap_model(agent._rap_adapter_cfg)
    LOG.info("RAP model loaded OK (output_num_poses=%d)", agent._rap_adapter_cfg.output_num_poses)

    vlm_dict = rap_cfg.get("vlm", {})
    setup_vlm_selector(agent, vlm_dict, planner_cfg=rap_cfg)

    agent._learned_selector = create_learned_planner_selector(
        agent,
        rule_based_merge_cfg=agent._rap_rule_based_merge,
        current_source_name="current_rap",
        learned_default_source="fallback_rap_argmax",
        plain_source="rap_argmax",
        score_fallback_key="rap_score",
        planner_log_name="RAP",
        strict_learned_argmax_lookup=True,
        q_key_prefix=True,
    )

def setup_drivor(agent):
    import torch
    from omegaconf import OmegaConf

    drivor_cfg = agent._cfg["drivor"]
    repo_root = Path(str(drivor_cfg["repo_root"])).expanduser().resolve()
    checkpoint_path = Path(str(drivor_cfg["checkpoint"])).expanduser().resolve()
    if not repo_root.exists():
        raise RuntimeError(f"DrivoR repo_root does not exist: {repo_root}")
    export_drivor_env(drivor_cfg)

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
    from autoagent0.planners.drivor import planner as drivor_client
    from autoagent0.adapters.hugsim.geometry import info_to_pose

    agent._drivor_client = drivor_client
    agent._info_to_pose = info_to_pose

    apply_rule_based_merge_env(
        drivor_cfg.get("rule_based_merge", {}),
        str(drivor_cfg.get("python_bin", sys.executable)),
        ("PLANNER_RULE_BASED_", "DRIVOR_RULE_BASED_"),
    )
    agent._drivor_rule_based_merge = resolve_rule_based_merge(
        drivor_cfg, ("PLANNER_RULE_BASED_", "DRIVOR_RULE_BASED_")
    )

    device_name = str(drivor_cfg.get("device", "cuda"))
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
        LOG.warning("CUDA unavailable for DrivoR; falling back to CPU")
    agent._drivor_device = torch.device(device_name)

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
    agent._drivor_agent = DrivoRAgent(
        config=drivo_config,
        lr_args=lr_args,
        checkpoint_path=str(checkpoint_path),
        progress_bar=False,
    )
    original_cwd = os.getcwd()
    try:
        os.chdir(repo_root)
        agent._drivor_agent.initialize()
    finally:
        os.chdir(original_cwd)
    try:
        agent._drivor_agent._drivor_model.to(agent._drivor_device)
        agent._drivor_agent._drivor_model.eval()
    except Exception:
        LOG.warning("Could not move DrivoR model to %s", agent._drivor_device)

    agent_config = getattr(agent._drivor_agent, "_config", None)
    if agent_config is not None:
        agent._drivor_output_num_poses = int(
            getattr(agent_config, "num_poses", drivor_cfg.get("output_num_poses", 8))
        )
    else:
        agent._drivor_output_num_poses = int(drivor_cfg.get("output_num_poses", 8))

    LOG.info(
        "DrivoR agent initialized OK (output_num_poses=%d)",
        agent._drivor_output_num_poses,
    )

    vlm_dict = drivor_cfg.get("vlm", {})
    setup_vlm_selector(agent, vlm_dict, planner_cfg=drivor_cfg)

    agent._learned_selector = create_learned_planner_selector(
        agent,
        rule_based_merge_cfg=agent._drivor_rule_based_merge,
        current_source_name="current_drivor",
        learned_default_source="drivor_argmax",
        plain_source="drivor_argmax",
        score_fallback_key="proposal_score",
        planner_log_name="DrivoR",
        strict_learned_argmax_lookup=False,
        q_key_prefix=False,
    )


def run_step_rule_based(agent, ego_state):

    obs = ego_state["obs"]
    info = ego_state["info"]
    privileged_agents = get_privileged_info(agent, info)

    try:
        selected, planner_debug = agent._rule_planner.process(
            obs=obs,
            info=info,
            info_history=agent._info_history,
            privileged_agents=privileged_agents,
            k=RULE_BASED_TOPK,
        )
    except Exception:
        LOG.exception("Rule-based planner.process() failed")
        return brake_control()

    if not selected:
        LOG.warning("Rule-based planner returned no trajectories")
        return brake_control()

    rb = agent._rb_client
    output_num_poses = agent._rule_based_output_num_poses

    try:
        proposals = rb.trajectory_to_proposals(selected, output_num_poses)
        scores = rb.trajectory_to_scores(selected, planner_debug)
    except Exception:
        LOG.exception("Failed to convert rule-based trajectories")
        return brake_control()

    vlm_cfg = agent._vlm_selector_cfg
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
                previous_selected_plan=agent._previous_selected_plan,
                previous_selected_pose=agent._previous_selected_pose,
                previous_selected_score=agent._previous_selected_score,
                previous_selected_timestamp=agent._previous_selected_timestamp,
                previous_selected_source=agent._previous_selected_source,
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
            selection_result = agent._vlm_selector.maybe_select(
                frame_index=agent._frame_index,
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
        return brake_control()

    agent._step_viz = build_step_viz_payload(
        selected_plan=selected_plan,
        selected_source=selected_source,
        selected_score=selected_score_raw,
        proposals=proposals,
        scores=scores,
        output_num_poses=output_num_poses,
    )

    try:
        acc_cmd, steer_rate = traj_to_control(selected_plan, info)
    except Exception:
        LOG.exception("traj2control failed")
        return brake_control()

    agent._previous_selected_plan = selected_plan.copy()
    agent._previous_selected_pose = agent._info_to_pose(info)
    agent._previous_selected_score = selected_score_raw
    agent._previous_selected_timestamp = float(info.get("timestamp", 0.0))
    agent._previous_selected_source = selected_source

    ctrl = carla.VehicleControl()
    ctrl.steer = float(np.clip(steer_rate, -1.0, 1.0))
    ctrl.throttle = float(np.clip(acc_cmd, 0.0, 1.0))
    ctrl.brake = float(np.clip(-acc_cmd, 0.0, 1.0))
    ctrl.hand_brake = False
    ctrl.manual_gear_shift = False
    agent._last_steer = ctrl.steer
    return ctrl

def run_step_rap(agent, ego_state):
    import torch

    obs = ego_state["obs"]
    info = ego_state["info"]
    cfg = agent._rap_adapter_cfg
    rap = agent._rap_client

    for cam_name in cfg.camera_order:
        if cam_name not in obs.get("rgb", {}):
            LOG.warning("Missing camera %s for RAP inference", cam_name)
            return brake_control()

    privileged_agents = None
    if (
        agent._rap_rule_based_merge.enabled
        and agent._rap_rule_based_merge.include_privileged_info
    ):
        privileged_agents = get_privileged_info(agent, info)

    try:
        features = rap.build_features(obs, list(agent._info_history), cfg)
        with torch.no_grad():
            predictions = agent._rap_model(
                features, targets=None, return_score=True
            )
            scores = predictions["score"][0].detach().cpu().numpy()
            proposals = predictions["trajectory"][0].detach().cpu().numpy()
    except Exception:
        LOG.exception("RAP inference failed")
        return brake_control()

    proposals_hugsim = np.stack(
        [
            rap.rap_to_hugsim_plan(proposals[i, : cfg.output_num_poses])
            for i in range(proposals.shape[0])
        ],
        axis=0,
    ).astype(np.float32)

    try:
        selected_plan, selected_score_raw, selected_source, _plan_payload = (
            run_learned_selection(
                agent,
                proposals_hugsim=proposals_hugsim,
                scores=scores,
                obs=obs,
                info=info,
                privileged_agents=privileged_agents,
            )
        )
    except Exception:
        LOG.exception("RAP plan selection failed")
        return brake_control()

    agent._step_viz = build_step_viz_payload(
        selected_plan=selected_plan,
        selected_source=selected_source,
        selected_score=selected_score_raw,
        proposals=proposals_hugsim,
        scores=scores,
        output_num_poses=cfg.output_num_poses,
        proposals_already_hugsim=True,
    )
    return apply_plan_control(agent, 
        selected_plan, info, selected_score_raw, selected_source
    )

def run_step_drivor(agent, ego_state):
    import torch

    obs = ego_state["obs"]
    info = ego_state["info"]
    drivor = agent._drivor_client
    output_num_poses = agent._drivor_output_num_poses

    for cam_name in drivor.MAP_HUGSIM_TO_DRIVOR:
        if cam_name not in obs.get("rgb", {}):
            LOG.warning("Missing camera %s for DrivoR inference", cam_name)
            return brake_control()

    privileged_agents = None
    if (
        agent._drivor_rule_based_merge.enabled
        and agent._drivor_rule_based_merge.include_privileged_info
    ):
        privileged_agents = get_privileged_info(agent, info)

    try:
        agent_input = drivor.build_agent_input_from_hugsim(
            obs, list(agent._info_history), num_history=EGO_HISTORY_FRAMES
        )
        features: Dict[str, Any] = {}
        for builder in agent._drivor_agent.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        features_batched: Dict[str, Any] = {}
        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                features_batched[key] = value.unsqueeze(0).to(agent._drivor_device)
            else:
                try:
                    tensor = torch.from_numpy(np.array(value))
                    features_batched[key] = tensor.unsqueeze(0).to(agent._drivor_device)
                except Exception:
                    features_batched[key] = value

        with torch.no_grad():
            try:
                predictions = agent._drivor_agent.forward(features_batched)
            except Exception:
                LOG.exception("DrivoR agent.forward failed; trying internal model")
                predictions = agent._drivor_agent._drivor_model(features_batched)

        proposals_raw, scores = drivor.extract_proposals_and_scores_from_predictions(
            predictions, output_num_poses=output_num_poses
        )
    except Exception:
        LOG.exception("DrivoR inference failed")
        return brake_control()

    proposals_hugsim = np.stack(
        [
            drivor.drivor_to_hugsim_plan(proposals_raw[i, :output_num_poses])
            for i in range(proposals_raw.shape[0])
        ],
        axis=0,
    ).astype(np.float32)

    try:
        selected_plan, selected_score_raw, selected_source, _plan_payload = (
            run_learned_selection(
                agent,
                proposals_hugsim=proposals_hugsim,
                scores=scores,
                obs=obs,
                info=info,
                privileged_agents=privileged_agents,
            )
        )
    except Exception:
        LOG.exception("DrivoR plan selection failed")
        return brake_control()

    agent._step_viz = build_step_viz_payload(
        selected_plan=selected_plan,
        selected_source=selected_source,
        selected_score=selected_score_raw,
        proposals=proposals_hugsim,
        scores=scores,
        output_num_poses=output_num_poses,
        proposals_already_hugsim=True,
    )
    return apply_plan_control(agent, 
        selected_plan, info, selected_score_raw, selected_source
    )

def setup_cameras_and_recording(agent):
    carla_cfg = agent._carla_cfg
    attach_legacy = coerce_bool(carla_cfg.get("attach_camera", False))
    vlm_enabled = bool(
        getattr(getattr(agent, "_vlm_selector_cfg", None), "enabled", False)
    )
    agent._camera_specs = resolve_camera_specs(
        carla_cfg,
        attach_legacy=attach_legacy,
        planner_type=agent._planner_type,
        vlm_enabled=vlm_enabled,
    )

    rec_cfg = carla_cfg.get("recording") or {}
    agent._recording_enabled = coerce_bool(rec_cfg.get("enabled", False))
    agent._recording_save_frames = coerce_bool(
        rec_cfg.get("save_frames", True), default=True
    )
    agent._recording_save_video = coerce_bool(
        rec_cfg.get("save_video", True), default=True
    )
    agent._recording_save_front_video = coerce_bool(
        rec_cfg.get("save_front_video", True), default=True
    )
    agent._recording_save_grid_video = coerce_bool(
        rec_cfg.get("save_grid_video", True), default=True
    )
    agent._recording_save_topdown_video = coerce_bool(
        rec_cfg.get("save_topdown_video", False), default=False
    )
    agent._recording_fps = float(rec_cfg.get("fps", 20.0))
    agent._recording_frame_ext = str(
        rec_cfg.get("frame_format", "jpg")
    ).lstrip(".")

    dir_name = str(rec_cfg.get("dir_name", "recordings"))
    agent._recording_dir = agent._output_dir / dir_name
    agent._grid_video_buffer: List[np.ndarray] = []
    agent._topdown_video_buffer: List[np.ndarray] = []
    agent._recording_finalized = False
    agent._predictions_dir = resolve_predictions_dir(agent)
    agent._topdown_camera_spec = None
    agent._topdown_pixels_per_meter = 10.0

    wants_topdown = (
        agent._recording_save_topdown_video
        or agent._predictions_dir is not None
    )
    if wants_topdown:
        agent._topdown_camera_spec = build_topdown_camera_spec(carla_cfg)
        agent._topdown_pixels_per_meter = topdown_pixels_per_meter(
            int(agent._topdown_camera_spec["width"]),
            float(agent._topdown_camera_spec["fov"]),
            float(agent._topdown_camera_spec["z"]),
        )
        if not any(
            spec["id"] == TOPDOWN_CAMERA_ID for spec in agent._camera_specs
        ):
            agent._camera_specs.append(agent._topdown_camera_spec)

    if agent._recording_enabled and not agent._camera_specs:
        agent._camera_specs = [_build_camera_spec("CAM_FRONT", carla_cfg)]
        LOG.warning(
            "Recording enabled but no cameras configured; attaching CAM_FRONT"
        )

    if agent._recording_enabled:
        agent._recording_dir.mkdir(parents=True, exist_ok=True)
        if agent._recording_save_frames:
            for spec in agent._camera_specs:
                (agent._recording_dir / spec["id"]).mkdir(exist_ok=True)
        LOG.info(
            "Recording enabled -> %s (cameras=%s, topdown=%s, predictions=%s)",
            agent._recording_dir,
            [spec["id"] for spec in agent._camera_specs],
            agent._recording_save_topdown_video,
            agent._predictions_dir,
        )
    elif agent._camera_specs:
        LOG.info(
            "Cameras attached (no recording): %s",
            [spec["id"] for spec in agent._camera_specs],
        )

def maybe_record_frames(agent, ego_state: Dict[str, Any]) -> None:
    if not getattr(agent, "_recording_enabled", False):
        return

    obs_rgb = ego_state.get("obs", {}).get("rgb", {})
    if not obs_rgb:
        return

    frame_idx = agent._frame_index
    if agent._recording_save_frames:
        for cam_id, rgb in obs_rgb.items():
            if cam_id == TOPDOWN_CAMERA_ID:
                continue
            frame_path = (
                agent._recording_dir
                / cam_id
                / f"{frame_idx:05d}.{agent._recording_frame_ext}"
            )
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(frame_path),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )

    if agent._recording_save_grid_video and len(obs_rgb) > 1:
        grid_rgb = {
            cam_id: rgb
            for cam_id, rgb in obs_rgb.items()
            if cam_id != TOPDOWN_CAMERA_ID
        }
        if len(grid_rgb) > 1:
            grid = compose_grid_frame(grid_rgb, VIDEO_GRID_LAYOUT)
            if grid is not None:
                agent._grid_video_buffer.append(grid)
                if agent._recording_save_frames:
                    grid_dir = agent._recording_dir / "grid"
                    grid_dir.mkdir(exist_ok=True)
                    grid_path = (
                        grid_dir
                        / f"{frame_idx:05d}.{agent._recording_frame_ext}"
                    )
                    cv2.imwrite(
                        str(grid_path),
                        cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
                    )

def write_rgb_video(
    out_path: Path, frames: List[np.ndarray], fps: float
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

def finalize_recording(agent) -> None:
    if getattr(agent, "_recording_finalized", False):
        return
    agent._recording_finalized = True
    if not getattr(agent, "_recording_enabled", False):
        return

    fps = max(1.0, float(getattr(agent, "_recording_fps", 20.0)))

    if getattr(agent, "_recording_save_grid_video", False) and agent._grid_video_buffer:
        grid_path = agent._recording_dir / "grid.mp4"
        write_rgb_video(grid_path, agent._grid_video_buffer, fps)
        LOG.info(
            "Wrote grid video: %s (%d frames)",
            grid_path,
            len(agent._grid_video_buffer),
        )

    topdown_buffer = getattr(agent, "_topdown_video_buffer", [])
    if getattr(agent, "_recording_save_topdown_video", False) and topdown_buffer:
        topdown_path = agent._recording_dir / "topdown.mp4"
        write_rgb_video(topdown_path, topdown_buffer, fps)
        LOG.info(
            "Wrote topdown video: %s (%d frames)",
            topdown_path,
            len(topdown_buffer),
        )

# ------------------------------------------------------------------
# sensors() — what CARLA should attach to the ego vehicle
# ------------------------------------------------------------------

def brake_control() -> carla.VehicleControl:
    ctrl = carla.VehicleControl()
    ctrl.brake = 1.0
    return ctrl

def get_ego_state(agent, hero, input_data, timestamp):
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
    command = get_route_command(agent)

    obs: Dict[str, Any] = {"rgb": {}}
    front_rgb = None
    topdown_rgb = None
    for spec in getattr(agent, "_camera_specs", []):
        cam_id = spec["id"]
        rgb = rgb_from_input_data(input_data, cam_id)
        if rgb is not None:
            obs["rgb"][cam_id] = rgb
            if cam_id == "CAM_FRONT":
                front_rgb = rgb
            if cam_id == TOPDOWN_CAMERA_ID:
                topdown_rgb = rgb

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
            getattr(hero.get_control(), "steer", agent._last_steer)
        ),
        "command": command,
        "ego_box": ego_box,
        "task_instruction": {
            0: "right", 1: "left", 2: "straight"
        }.get(command, "straight"),
        "cam_params": build_cam_params(agent),
    }
    

    return {
        "info": info,
        "obs": obs,
        "front_rgb": front_rgb,
        "topdown_rgb": topdown_rgb,
        "speed_mps": speed_mps,
        "transform": transform,
    }

def get_privileged_info(agent, info: Dict[str, Any]) -> List[Dict[str, Any]]:
    world = CarlaDataProvider.get_world()
    if world is None:
        return []

    ego = find_hero(agent)
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

def build_cam_params(agent):
    cam_params: Dict[str, Dict[str, Any]] = {}
    specs = getattr(agent, "_camera_specs", None)
    if not specs:
        specs = [_build_camera_spec("CAM_FRONT", agent._carla_cfg)]

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

def get_route_command(agent) -> int:
    if not agent._global_plan_world_coord:
        return 2
    start = max(0, agent._route_cursor)
    end = min(len(agent._global_plan_world_coord), start + 20)
    for _, option in agent._global_plan_world_coord[start:end]:
        name = str(getattr(option, "name", option)).upper()
        if "LEFT" in name and "CHANGE" not in name:
            return 1
        if "RIGHT" in name and "CHANGE" not in name:
            return 0
    return 2

def find_hero(agent):
    world = CarlaDataProvider.get_world()
    if world is None:
        return None
    for actor in world.get_actors():
        if actor.attributes.get("role_name") == "hero":
            return actor
    return None

