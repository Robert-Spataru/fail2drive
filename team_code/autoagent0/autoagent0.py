# team_code/autoagent0_agent.py

from __future__ import annotations
import sys
import os
import math
import logging
import numpy as np
import torch
import carla
import cv2
from pathlib import Path
from collections import deque
from omegaconf import OmegaConf

# Add fail2drive root to path so sim/ and planners/ are importable
_F2D_ROOT = Path(__file__).parent.parent
if str(_F2D_ROOT) not in sys.path:
    sys.path.insert(0, str(_F2D_ROOT))

# Add HUGSIM root for anything not copied over
_HUGSIM_ROOT = Path("/data/robert/HUGSIM")
if str(_HUGSIM_ROOT) not in sys.path:
    sys.path.insert(0, str(_HUGSIM_ROOT))

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

LOG = logging.getLogger(__name__)

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
        self._info_history = deque(maxlen=4)

        # Planner-specific setup
        if self._planner_type == "rule_based":
            self._setup_rule_based()
        elif self._planner_type == "rap":
            self._setup_rap()
        elif self._planner_type == "drivor":
            self._setup_drivor()

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

    # State for carry-prev logic (same fields as client.py)
    self._previous_selected_plan = None
    self._previous_selected_pose = None
    self._previous_selected_score = None
    self._previous_selected_timestamp = None
    self._previous_selected_source = None
    
    def _setup_rule_based(self):
        rb_cfg = self._cfg["rule_based"]

        # Add Rule-Planner repo to path
        repo_root = str(rb_cfg["repo_root"])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        # Set env vars the rule-based adapter reads
        os.environ["RULE_BASED_REPO_ROOT"] = repo_root
        os.environ["RULE_BASED_CONFIG"] = str(rb_cfg.get("config", ""))
        os.environ["RULE_BASED_DEVICE"] = str(rb_cfg.get("device", "cpu"))
        os.environ["RULE_BASED_PYTHON_BIN"] = str(
            rb_cfg.get("python_bin", sys.executable)
        )

        # Import rule-based planner adapter
        # This is the same adapter used in HUGSIM via the pipe,
        # but here we call its functions directly
        from planners.rule_based.adapter import RuleBasedPlanner
        self._rule_planner = RuleBasedPlanner(
            config_path=rb_cfg["config"],
            device=rb_cfg.get("device", "cpu"),
            output_dir=self._output_dir,
        )

        # Rule-based does not use VLM by default
        # (vlm.enabled: false in the config)
        self._vlm_selector = None
        self._vlm_selector_cfg = None

        LOG.info(
            "Rule-based planner initialized repo=%s", repo_root
        )

    # ------------------------------------------------------------------
    # RAP setup (to be filled in at Stage 2)
    # ------------------------------------------------------------------

    def _setup_rap(self):
        raise NotImplementedError(
            "RAP setup not yet implemented — start with rule_based"
        )

    # ------------------------------------------------------------------
    # DrivoR setup (to be filled in at Stage 3)
    # ------------------------------------------------------------------

    def _setup_drivor(self):
        raise NotImplementedError(
            "DrivoR setup not yet implemented — start with rule_based"
        )

    # ------------------------------------------------------------------
    # sensors() — what CARLA should attach to the ego vehicle
    # ------------------------------------------------------------------

    def sensors(self):
        carla_cfg = self._carla_cfg

        sensors = []

        # Rule-based doesn't need cameras for its core logic
        # but we add front camera anyway for VLM if it gets enabled later
        # and for debug visualization
        if self._planner_type != "rule_based" or carla_cfg.get(
            "attach_camera", False
        ):
            sensors.append({
                "type": "sensor.camera.rgb",
                "x": float(carla_cfg.get("camera_x", 0.7)),
                "y": float(carla_cfg.get("camera_y", 0.0)),
                "z": float(carla_cfg.get("camera_z", 1.6)),
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                "width": int(carla_cfg.get("camera_width", 800)),
                "height": int(carla_cfg.get("camera_height", 600)),
                "fov": float(carla_cfg.get("camera_fov", 100.0)),
                "id": "CAM_FRONT",
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
            ctrl = carla.VehicleControl()
            ctrl.brake = 1.0
            return ctrl

        # Build shared ego state regardless of planner type
        ego_state = self._get_ego_state(hero, input_data, timestamp)
        self._info_history.append(ego_state["info"])
        while len(self._info_history) < 4:
            self._info_history.appendleft(
                dict(self._info_history[0])
            )

        # Route cursor advance
        self._route_cursor = min(
            self._route_cursor + 1,
            max(0, len(self._global_plan_world_coord) - 1),
        )

        # Dispatch to planner-specific run
        if self._planner_type == "rule_based":
            return self._run_step_rule_based(ego_state)
        elif self._planner_type == "rap":
            return self._run_step_rap(ego_state)
        elif self._planner_type == "drivor":
            return self._run_step_drivor(ego_state)

    # ------------------------------------------------------------------
    # Rule-based run_step
    # ------------------------------------------------------------------

    def _run_step_rule_based(self, ego_state):
        from sim.utils.sim_utils import traj2control

        # Get privileged info from CARLA world
        # Rule-based needs nearby agent positions
        privileged_info = self._get_privileged_info()

        # Call rule planner directly (no pipe needed)
        plan = self._rule_planner.plan(
            info=ego_state["info"],
            privileged_agents=privileged_info,
        )

        if plan is None or len(plan) == 0:
            ctrl = carla.VehicleControl()
            ctrl.brake = 1.0
            return ctrl

        acc_cmd, steer_rate = traj2control(
            np.asarray(plan, dtype=np.float32),
            ego_state["info"],
        )

        ctrl = carla.VehicleControl()
        ctrl.steer = float(np.clip(steer_rate, -1.0, 1.0))
        ctrl.throttle = float(np.clip(acc_cmd, 0.0, 1.0))
        ctrl.brake = float(np.clip(-acc_cmd, 0.0, 1.0))
        ctrl.hand_brake = False
        ctrl.manual_gear_shift = False
        self._last_steer = ctrl.steer
        self._frame_index += 1
        return ctrl

    # ------------------------------------------------------------------
    # RAP run_step (to be filled in at Stage 2)
    # ------------------------------------------------------------------

    def _run_step_rap(self, ego_state):
        raise NotImplementedError("RAP run_step not yet implemented")

    # ------------------------------------------------------------------
    # DrivoR run_step (to be filled in at Stage 3)
    # ------------------------------------------------------------------

    def _run_step_drivor(self, ego_state):
        raise NotImplementedError("DrivoR run_step not yet implemented")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

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

        # Extract front camera if available
        front_rgb = None
        if "CAM_FRONT" in input_data:
            raw = np.array(input_data["CAM_FRONT"][1])
            front_rgb = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)

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

    def _get_privileged_info(self):
        # In HUGSIM this came through a special pipe
        # In CARLA we get it directly from the world
        # Format needs to match what Rule-Planner expects —
        # check planners/rule_based/ to confirm the exact schema
        world = CarlaDataProvider.get_world()
        if world is None:
            return []

        ego = self._find_hero()
        ego_loc = ego.get_location() if ego else None
        nearby = []

        for actor in world.get_actors():
            type_id = actor.type_id
            if not ("vehicle" in type_id or "walker" in type_id):
                continue
            if ego is not None and actor.id == ego.id:
                continue
            loc = actor.get_location()
            # Only include actors within 50m
            if ego_loc is not None:
                dist = math.sqrt(
                    (loc.x - ego_loc.x)**2
                    + (loc.y - ego_loc.y)**2
                )
                if dist > 50.0:
                    continue
            vel = actor.get_velocity()
            t = actor.get_transform()
            nearby.append({
                "id": actor.id,
                "x": loc.x,
                "y": loc.y,
                "z": loc.z,
                "yaw": t.rotation.yaw,
                "vx": vel.x,
                "vy": vel.y,
                "type": type_id,
            })

        return nearby

    def _build_cam_params(self):
        carla_cfg = self._carla_cfg
        w = int(carla_cfg.get("camera_width", 800))
        h = int(carla_cfg.get("camera_height", 600))
        fov = float(carla_cfg.get("camera_fov", 100.0))
        cx = float(carla_cfg.get("camera_x", 0.7))
        cy = float(carla_cfg.get("camera_y", 0.0))
        cz = float(carla_cfg.get("camera_z", 1.6))
        fov_rad = math.radians(fov)

        front2cam = np.eye(4, dtype=np.float32)
        front2cam[0, 3] = cx
        front2cam[1, 3] = cy
        front2cam[2, 3] = cz

        intrinsic = {
            "H": h, "W": w,
            "cx": w / 2.0, "cy": h / 2.0,
            "fovx": fov_rad, "fovy": fov_rad,
        }
        single_cam = {
            "intrinsic": intrinsic,
            "front2cam": front2cam,
            "v2c": front2cam.copy(),
            "l2c": front2cam.copy(),
        }
        return {
            "CAM_FRONT": single_cam,
            "CAM_BACK": single_cam.copy(),
            "CAM_FRONT_LEFT": single_cam.copy(),
            "CAM_FRONT_RIGHT": single_cam.copy(),
        }

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
        if self._vlm_selector is not None:
            try:
                self._vlm_selector.finalize()
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
        )