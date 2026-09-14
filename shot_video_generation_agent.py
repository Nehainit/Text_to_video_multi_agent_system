import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from artifact_store import public_artifact_url
from editor_agent import _clip, _resolution
from image_agent import _generate_scene_video_file, _out_dir
from prompts import SHOT_VIDEO_GENERATION_CONFIG
from schema import AgentState


def _failure_type(error: str) -> str:
    lowered = error.casefold()
    return "invalid_generation_output" if any(term in lowered for term in ("did not return", "empty", "missing", "corrupt", "unreadable")) else "video_generation_error"


def _record(state: AgentState, shot: dict, image_file: str | None, requested: float, generated: float | None, attempt: int, video_file: str | None, error: str | None) -> dict:
    return {
        "shot_id": shot["shot_id"],
        "scene_id": shot["scene_id"],
        "generation_status": "success" if video_file else "failed",
        "source_image_file": image_file,
        "video_file": video_file,
        "requested_duration_seconds": requested,
        "generated_duration_seconds": generated if video_file else None,
        "model_used": SHOT_VIDEO_GENERATION_CONFIG["model_used"],
        "aspect_ratio": state.get("aspect_ratio", SHOT_VIDEO_GENERATION_CONFIG["default_aspect_ratio"]),
        "video_quality": state.get("video_quality", "standard"),
        "generation_attempt": attempt,
        "failure_type": None if video_file else _failure_type(error or ""),
        "error": error,
    }


def _worker_count(job_count: int) -> int:
    try:
        configured = int(os.getenv("VIDEO_GENERATION_WORKERS", "3"))
    except ValueError:
        configured = 3
    return min(job_count, max(1, min(4, configured)))


def _fallback_motion(plan: dict) -> str:
    camera = plan.get("camera_motion") or {}
    return {
        "subtle_push_in": "zoom_in",
        "subtle_pull_back": "zoom_out",
        "slow_pan": "pan_right",
        "gentle_tilt": "tilt_up",
        "small_lateral_tracking": "pan_right",
    }.get(camera.get("type"), "static")


def _generate_job(args: tuple) -> tuple[dict, dict | None, str | None]:
    state, shot, image_file, plan, out, index, round_number, provider_duration, start_image_url, provider_aspect_ratio, max_attempts = args
    shot_id = shot["shot_id"]
    candidate_id = f"{shot_id}-c{round_number + 1}"
    seed = int(SHOT_VIDEO_GENERATION_CONFIG["seed_base"]) * (round_number + 1) + index
    video_file = out / f"{candidate_id}.mp4"
    started = time.monotonic()
    worker = threading.current_thread().name
    prefix = f"[pipeline:{state.get('thread_id', '-')}] [video-worker:{worker}]"
    print(f"{prefix} START {shot_id} candidate={candidate_id}", flush=True)
    error = ""
    result = None
    attempt = 0
    scene = {**shot, "duration_seconds": provider_duration}
    for attempt in range(1, max_attempts + 1):
        try:
            result = _generate_scene_video_file(
                state,
                scene,
                image_file,
                video_file,
                plan["video_prompt"],
                start_image_url,
                plan["negative_prompt"],
                float(SHOT_VIDEO_GENERATION_CONFIG["cfg_scale"]),
                provider_aspect_ratio,
                bool(SHOT_VIDEO_GENERATION_CONFIG["generate_audio"]),
            )
            if not Path(result).is_file() or Path(result).stat().st_size == 0:
                raise RuntimeError("Provider returned an empty video asset.")
            break
        except Exception as exc:
            result, error = None, str(exc)
    requested = float(plan["duration_seconds"])
    warning = None
    generated_duration = float(provider_duration)
    if not result:
        try:
            width, height = _resolution(state)
            _clip(image_file, requested, _fallback_motion(plan), video_file, width, height)
            result, generated_duration = str(video_file), requested
            warning = f"{shot_id} image-to-video failed; used FFmpeg movement instead: {error}"
        except Exception as fallback_exc:
            error = f"{error}; FFmpeg fallback failed: {fallback_exc}"
    item = _record(state, shot, image_file, requested, generated_duration, attempt, result, None if result else error)
    if result and warning:
        item["model_used"] = "ffmpeg"
    candidate = None
    if state.get("quality_mode") == "refine":
        candidate = {
            "shot_id": shot_id, "candidate_id": candidate_id, "round": round_number,
            "prompt": plan["video_prompt"], "negative_prompt": plan["negative_prompt"],
            "seed": seed, "file": result, "generation_status": item["generation_status"],
            "generation_attempt": attempt, "error": item["error"],
        }
    warning = warning or (None if result else f"{shot_id} video generation failed after {attempt} attempts: {error}")
    print(
        f"{prefix} {'DONE' if result else 'ERROR'} {shot_id} candidate={candidate_id} "
        f"elapsed={time.monotonic() - started:.1f}s",
        flush=True,
    )
    return item, candidate, warning


def _generate(state: AgentState) -> dict:
    shots = state.get("shot_plan") or []
    images = state.get("image_files") or []
    plans = state.get("motion_plans") or []
    if not shots or [item.get("shot_id") for item in plans] != [item.get("shot_id") for item in shots] or len(images) != len(shots):
        raise RuntimeError("Shot Video Generation needs one ordered image and motion plan per approved shot.")

    out = _out_dir(state, "scene_videos")
    out.mkdir(parents=True, exist_ok=True)
    previous = {item["shot_id"]: item for item in state.get("generated_videos", [])}
    retry_targets = set(state.get("video_retry_shots", []))
    retrying = bool(previous and retry_targets)
    targets = retry_targets if retrying else {shot["shot_id"] for shot in shots}
    required_durations = {
        item["shot_id"]: float(item["required_duration_seconds"])
        for item in state.get("timeline_issues", [])
        if item.get("type") == "insufficient_clip_duration"
        and item.get("shot_id") in targets
        and item.get("required_duration_seconds") is not None
    }
    round_number = int(state.get("video_refinement_round", 0)) + int(retrying)
    generated_by_shot = {item["shot_id"]: item for item in previous.values()}
    new_candidates = []
    jobs = []
    warnings = list(state.get("warnings", []))
    allowed_durations = [int(value) for value in SHOT_VIDEO_GENERATION_CONFIG["allowed_duration_seconds"]]
    max_attempts = int(SHOT_VIDEO_GENERATION_CONFIG["max_retries"]) + 1
    requested_aspect_ratio = str(state.get("aspect_ratio", SHOT_VIDEO_GENERATION_CONFIG["default_aspect_ratio"]))
    provider_aspect_ratio = str(SHOT_VIDEO_GENERATION_CONFIG["aspect_ratios"][requested_aspect_ratio])

    for index, (shot, image_file, plan) in enumerate(zip(shots, images, plans), start=1):
        if shot["shot_id"] not in targets and shot["shot_id"] in previous:
            continue
        requested = max(float(plan["duration_seconds"]), required_durations.get(shot["shot_id"], 0))
        plan = {**plan, "duration_seconds": requested}
        provider_duration = next((value for value in sorted(allowed_durations) if value >= requested), max(allowed_durations))
        candidate_id = f"{shot['shot_id']}-c{round_number + 1}"
        seed = int(SHOT_VIDEO_GENERATION_CONFIG["seed_base"]) * (round_number + 1) + index
        source = Path(image_file) if image_file else None
        if not source or not source.is_file() or source.stat().st_size == 0:
            item = _record(state, shot, image_file, requested, None, 0, None, "The approved source image is missing or invalid.")
            item["failure_type"] = "missing_source_image"
            generated_by_shot[shot["shot_id"]] = item
            if state.get("quality_mode") == "refine":
                new_candidates.append({
                    "shot_id": shot["shot_id"], "candidate_id": candidate_id, "round": round_number,
                    "prompt": plan["video_prompt"], "negative_prompt": plan["negative_prompt"],
                    "seed": seed, "file": None, "generation_status": "failed",
                    "generation_attempt": 0, "error": item["error"],
                })
            continue
        start_image_url = public_artifact_url(state, image_file) if os.getenv("MINIO_PUBLIC_ENDPOINT") and requested <= max(allowed_durations) else None
        if not start_image_url:
            error = (
                f"The requested {requested:g}s duration exceeds the provider maximum of {max(allowed_durations)}s."
                if requested > max(allowed_durations)
                else "The approved source image has no provider-accessible public URL."
            )
            video_file = out / f"{candidate_id}.mp4"
            try:
                width, height = _resolution(state)
                _clip(str(source), requested, _fallback_motion(plan), video_file, width, height)
                item = _record(state, shot, image_file, requested, requested, 1, str(video_file), None)
                item["model_used"] = "ffmpeg"
            except Exception as exc:
                item = _record(state, shot, image_file, requested, None, 1, None, f"{error}; FFmpeg fallback failed: {exc}")
            generated_by_shot[shot["shot_id"]] = item
            if state.get("quality_mode") == "refine":
                new_candidates.append({
                    "shot_id": shot["shot_id"], "candidate_id": candidate_id, "round": round_number,
                    "prompt": plan["video_prompt"], "negative_prompt": plan["negative_prompt"],
                    "seed": seed, "file": item["video_file"], "generation_status": item["generation_status"],
                    "generation_attempt": item["generation_attempt"], "error": item["error"],
                })
            warnings.append(
                f"{shot['shot_id']} image-to-video unavailable; used FFmpeg movement instead."
                if item["video_file"] else f"{shot['shot_id']} video generation failed: {item['error']}"
            )
            continue
        jobs.append((
            state, shot, image_file, plan, out, index, round_number, provider_duration,
            start_image_url, provider_aspect_ratio, max_attempts,
        ))

    if jobs:
        workers = _worker_count(len(jobs))
        print(f"[pipeline:{state.get('thread_id', '-')}] VIDEO WORKERS count={workers} jobs={len(jobs)}", flush=True)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="shot-video") as pool:
            for item, candidate, warning in pool.map(_generate_job, jobs):
                generated_by_shot[item["shot_id"]] = item
                if candidate:
                    new_candidates.append(candidate)
                if warning:
                    warnings.append(warning)

    generated_videos = [generated_by_shot[shot["shot_id"]] for shot in shots]

    result = {
        "generated_videos": generated_videos,
        "scene_video_files": [item["video_file"] for item in generated_videos],
        "warnings": list(dict.fromkeys(warnings)),
        "video_refinement_round": round_number,
    }
    if state.get("quality_mode") == "refine":
        result.update(
            video_candidates=[*state.get("video_candidates", []), *new_candidates],
        )
    return result


def create_shot_videos(state: AgentState) -> dict:
    if not SHOT_VIDEO_GENERATION_CONFIG["enabled"]:
        raise RuntimeError("Shot Video Generation Agent is disabled in config/shot_video_generation.yml.")
    return _generate(state)
