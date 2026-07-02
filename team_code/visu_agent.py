"""
Simplified visualization agent:
- Drives like AutoPilot (PDM-Lite expert)
- Optionally displays front RGB live in a window
- Optionally records front-camera video via configs/pdm_lite.yaml
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

try:
  import pygame
except ImportError as exc:
  raise RuntimeError("cannot import pygame, make sure pygame package is installed") from exc

try:
  from omegaconf import OmegaConf
except ImportError:
  OmegaConf = None  # type: ignore

from autopilot import AutoPilot


def get_entry_point():
  return "VisuAgent"


def _coerce_bool(value: Any, default: bool = False) -> bool:
  if isinstance(value, bool):
    return value
  if value is None:
    return default
  return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _sanitize_run_id(value: str) -> str:
  return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _resolve_scenario_output_dir(
    carla_cfg: Dict[str, Any],
    *,
    base_output_dir: Path,
    scenario_name: Optional[str],
    repetition_index: int,
) -> Path:
  auto_run_id = _coerce_bool(carla_cfg.get("auto_run_id", True), default=True)
  if not auto_run_id:
    base_output_dir.mkdir(parents=True, exist_ok=True)
    return base_output_dir

  scenario_id = carla_cfg.get("scenario_id") or carla_cfg.get("run_id")
  if not scenario_id:
    if scenario_name:
      scenario_id = str(scenario_name)
      if int(repetition_index) > 0:
        scenario_id = f"{scenario_id}_rep{int(repetition_index)}"
    else:
      scenario_id = "pdm_lite"
  scenario_id = _sanitize_run_id(str(scenario_id))

  scenarios_parent = (
      str(carla_cfg.get("scenarios_dir_name", carla_cfg.get("run_dir_name", "scenarios"))).strip()
      or "scenarios"
  )
  scenario_output_dir = base_output_dir / scenarios_parent / scenario_id

  overwrite = _coerce_bool(carla_cfg.get("overwrite_scenario_output", True), default=True)
  if scenario_output_dir.exists() and overwrite:
    shutil.rmtree(scenario_output_dir)
  scenario_output_dir.mkdir(parents=True, exist_ok=True)
  return scenario_output_dir


def _write_rgb_video(out_path: Path, frames: List[Any], fps: float) -> None:
  if not frames:
    return
  height, width = frames[0].shape[:2]
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
  if not writer.isOpened():
    print(f"Warning: VideoWriter failed for {out_path}")
    return
  for frame in frames:
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
  writer.release()


class VisuAgent(AutoPilot):
  """AutoPilot variant with optional live window and front-camera video recording."""

  def set_route_context(self, route_name: str, repetition_index: int = 0) -> None:
    self._route_name = str(route_name)
    self._repetition_index = int(repetition_index)

  def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
    super().setup(path_to_conf_file, route_index, traffic_manager=None)
    self._interface = None
    self._quit_requested = False
    self._route_name = getattr(self, "_route_name", None)
    self._repetition_index = int(getattr(self, "_repetition_index", 0))
    self._front_video_buffer: List[Any] = []
    self._recording_enabled = False
    self._recording_save_front_video = False
    self._recording_dir: Optional[Path] = None
    self._recording_fps = 20.0
    self._recording_finalized = False
    self._live_window = True

    self._load_recording_config(path_to_conf_file)
    self.visualize = self._live_window

  def _load_recording_config(self, path_to_conf_file) -> None:
    if not path_to_conf_file or not str(path_to_conf_file).strip():
      return
    config_path = Path(str(path_to_conf_file)).expanduser()
    if not config_path.exists():
      print(f"Warning: agent config not found: {config_path}")
      return
    if OmegaConf is None:
      print("Warning: omegaconf is not installed; skipping video recording config")
      return

    raw_cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw_cfg, dict):
      return

    pdm_cfg = raw_cfg.get("pdm_lite", {})
    if isinstance(pdm_cfg, dict):
      self._live_window = _coerce_bool(pdm_cfg.get("live_window", True), default=True)

    carla_cfg = raw_cfg.get("carla", {})
    if not isinstance(carla_cfg, dict):
      return

    rec_cfg = carla_cfg.get("recording", {})
    if not isinstance(rec_cfg, dict):
      return
    if not _coerce_bool(rec_cfg.get("enabled", False), default=False):
      return

    base_output_dir = Path(str(carla_cfg.get("output_dir", "outputs/pdm_lite"))).expanduser()
    scenario_output_dir = _resolve_scenario_output_dir(
        carla_cfg,
        base_output_dir=base_output_dir,
        scenario_name=self._route_name,
        repetition_index=self._repetition_index,
    )
    dir_name = str(rec_cfg.get("dir_name", "recordings")).strip() or "recordings"
    self._recording_dir = scenario_output_dir / dir_name
    self._recording_dir.mkdir(parents=True, exist_ok=True)

    self._recording_enabled = True
    self._recording_save_front_video = _coerce_bool(
        rec_cfg.get("save_front_video", True), default=True
    )
    self._recording_fps = max(1.0, float(rec_cfg.get("fps", 20.0)))
    print(f"PDM-Lite recording enabled: {self._recording_dir}")

  def sensors(self):
    result = super().sensors()

    result += [{
        "type": "sensor.camera.rgb",
        "x": self.config.camera_pos[0]-4,
        "y": self.config.camera_pos[1],
        "z": self.config.camera_pos[2]+1.5,
        "roll": self.config.camera_rot_0[0],
        "pitch": self.config.camera_rot_0[1]-8,
        "yaw": self.config.camera_rot_0[2],
        "width": 1920,
        "height": 1080,
        "fov": 110,
        "id": "rgb",
    }]

    return result

  def run_step(self, input_data, timestamp, sensors=None, plant=False):
    control = super().run_step(input_data, timestamp, sensors=sensors, plant=plant)

    rgb_bgr = input_data["rgb"][1][:, :, :3]
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    if self._recording_enabled and self._recording_save_front_video:
      self._front_video_buffer.append(rgb.copy())

    if self._live_window and not self._quit_requested:
      self._visualize(rgb)

    return control

  def _visualize(self, rgb_img):
    if self._interface is None:
      self._interface = _VisuInterface(rgb_img.shape[1], rgb_img.shape[0])

    self._interface.run_interface(rgb_img)
    if self._interface.quit_requested:
      self._quit_requested = True

  def _finalize_recording(self) -> None:
    if self._recording_finalized or not self._recording_enabled or self._recording_dir is None:
      return
    self._recording_finalized = True

    if self._recording_save_front_video and self._front_video_buffer:
      front_path = self._recording_dir / "front.mp4"
      _write_rgb_video(front_path, self._front_video_buffer, self._recording_fps)
      print(f"Wrote front video: {front_path} ({len(self._front_video_buffer)} frames)")

  def destroy(self, results=None):
    self._finalize_recording()
    if self._interface is not None:
      self._interface.close()
    super().destroy(results)


class _VisuInterface:
  """Minimal pygame interface that displays one RGB image per step."""

  def __init__(self, width, height):
    self._width = width
    self._height = height
    self.quit_requested = False

    pygame.init()
    self._display = pygame.display.set_mode((self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Visu Agent")

  def run_interface(self, image):
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        self.quit_requested = True
      if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        self.quit_requested = True

    surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
    self._display.blit(surface, (0, 0))
    pygame.display.flip()

  def close(self):
    pygame.quit()
