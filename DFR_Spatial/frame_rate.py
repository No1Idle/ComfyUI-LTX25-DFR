"""Model conditioning FPS is independent of playback/audio timing."""

import math


def official_conditioning_fps(playback_fps: float) -> float:
    """Match upstream DFR: preserve <=30 FPS, condition higher rates at 60."""
    playback_fps = float(playback_fps)
    if not math.isfinite(playback_fps) or playback_fps <= 0:
        raise ValueError(f"playback_fps must be finite and positive, got {playback_fps}.")
    return 60.0 if playback_fps > 30.0 else playback_fps


def model_video_positions(positions, state_fps: float):
    """Convert stored time coordinates without mutating playback-based state.

    Spatial states use playback FPS; temporal tile states already use conditioning
    FPS. Recover pixel time before dividing by the model rate in either case.
    Spatial coordinates and all audio coordinates retain their original basis.
    """
    target_fps = official_conditioning_fps(state_fps)
    result = positions.clone()
    if target_fps != float(state_fps):
        result[:, 0, ...] *= float(state_fps)
        result[:, 0, ...] /= target_fps
    return result
