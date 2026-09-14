import json
import logging
import subprocess
from pathlib import Path

import yaml

from video_automation.schema import AgentState
from video_automation.agents.video_validator import _decodes, _fps, _number, _probe


logger = logging.getLogger(__name__)
with (Path(__file__).parents[2] / "config" / "video_compilation.yml").open(encoding="utf-8") as config_file:
    VIDEO_COMPILATION_CONFIG = yaml.safe_load(config_file)["video_compilation"]


def _issue(issue_type: str, description: str, *, shot_id=None, expected=None, actual=None) -> dict:
    issue = {"type": issue_type, "description": description}
    if shot_id is not None:
        issue["shot_id"] = shot_id
    if expected is not None:
        issue["expected"] = expected
    if actual is not None:
        issue["actual"] = actual
    return issue


def _shot_timestamps(timeline: list[dict]) -> list[dict]:
    return [
        {
            "timeline_item_id": item.get("timeline_item_id"),
            "shot_id": item.get("shot_id"),
            "scene_id": item.get("scene_id"),
            "start_sec": item.get("timeline_start"),
            "end_sec": item.get("timeline_end"),
            "narration_segment_ids": list(item.get("narration_segment_ids", [])),
        }
        for item in timeline
    ]


def _failure(state: AgentState, issues: list[dict], source: str, error: str, timeline: list[dict]) -> dict:
    retry_count = int(state.get("rough_cut_retry_count", 0))
    can_retry = retry_count < int(VIDEO_COMPILATION_CONFIG["max_retries"])
    failed_shots = list(dict.fromkeys(
        issue["shot_id"] for issue in issues if issue.get("shot_id") and source == "source_video"
    ))
    result = {
        "compilation_status": "failed",
        "rough_cut_file": None,
        "narration_file": state.get("narration_file"),
        "duration_seconds": None,
        "width": None,
        "height": None,
        "fps": None,
        "shot_timestamps": _shot_timestamps(timeline),
        "sync_valid": False,
        "compilation_issues": issues,
        "compilation_error": error,
        "rough_cut_failure_source": source,
        "rough_cut_retry_count": retry_count + int(can_retry),
        "rough_cut_retry_exhausted": not can_retry,
        "video_retry_shots": failed_shots if can_retry else [],
        "pipeline_status": "rough_cut_failed" if not can_retry else "rough_cut_retry",
    }
    logger.info("rough_cut_compilation %s", json.dumps({
        "status": "failed", "failure_source": source,
        "issue_types": [issue["type"] for issue in issues],
        "retry_count": result["rough_cut_retry_count"],
    }, sort_keys=True))
    return result


def _duration(probe: dict) -> float | None:
    format_data = probe.get("format") if isinstance(probe.get("format"), dict) else {}
    value = _number(format_data.get("duration"))
    if value:
        return value
    return next((_number(stream.get("duration")) for stream in probe.get("streams", []) if _number(stream.get("duration"))), None)


def _number_or_none(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _preconditions(state: AgentState) -> tuple[list[dict], list[dict], float | None]:
    if "edit_timeline" not in state:
        return [], [_issue("timeline_missing", "The approved edit timeline is missing.")], None
    timeline = state.get("edit_timeline") or []
    issues = []
    if not state.get("timeline_valid"):
        issues.append(_issue("timeline_invalid", "The edit timeline is not marked valid."))
    if not timeline:
        issues.append(_issue("timeline_empty", "The edit timeline contains no items."))

    narration = Path(state["narration_file"]) if state.get("narration_file") else None
    narration_duration = None
    if not narration or not narration.is_file() or narration.stat().st_size == 0:
        issues.append(_issue("narration_file_missing", "The narration audio file is missing or empty."))
    else:
        try:
            narration_probe = _probe(narration)
            if not any(stream.get("codec_type") == "audio" for stream in narration_probe.get("streams", [])):
                issues.append(_issue("narration_stream_missing", "The narration file contains no audio stream."))
            narration_duration = _duration(narration_probe)
            if narration_duration is None:
                issues.append(_issue("narration_stream_missing", "The narration duration could not be determined."))
        except RuntimeError as exc:
            issues.append(_issue("narration_stream_missing", f"The narration file could not be parsed: {exc}"))

    validation_by_shot = {
        item.get("shot_id"): item for item in state.get("video_validation_results", []) if isinstance(item, dict)
    }
    epsilon = float(VIDEO_COMPILATION_CONFIG["timeline_tolerance_seconds"])
    duration_tolerance = float(VIDEO_COMPILATION_CONFIG["duration_tolerance_seconds"])
    previous = None
    for item in timeline:
        shot_id = item.get("shot_id")
        video = Path(item["video_file"]) if item.get("video_file") else None
        source_in = _number_or_none(item.get("source_in"))
        source_out = _number_or_none(item.get("source_out"))
        start = _number_or_none(item.get("timeline_start"))
        end = _number_or_none(item.get("timeline_end"))
        if not video or not video.is_file() or video.stat().st_size == 0:
            issues.append(_issue("video_file_missing", "A timeline source video is missing or empty.", shot_id=shot_id))
        if source_in is None or source_out is None or source_in < 0 or source_out <= source_in:
            issues.append(_issue("invalid_source_range", "source_in/source_out define an invalid range.", shot_id=shot_id))
        if start is None or end is None or start < 0 or end <= start:
            issues.append(_issue("invalid_timeline_range", "timeline_start/timeline_end define an invalid range.", shot_id=shot_id))
        if None not in (source_in, source_out, start, end) and source_in >= 0 and source_out > source_in and end > start:
            source_duration = source_out - source_in
            timeline_duration = end - start
            if abs(source_duration - timeline_duration) > duration_tolerance:
                issues.append(_issue(
                    "invalid_source_range", "The requested source range does not match its timeline allocation.",
                    shot_id=shot_id, expected=timeline_duration, actual=source_duration,
                ))
        validation = validation_by_shot.get(shot_id, {})
        actual_duration = _number_or_none((validation.get("metadata") or {}).get("duration_seconds"))
        if source_out is not None and actual_duration is not None and source_out - actual_duration > duration_tolerance:
            issues.append(_issue(
                "source_range_exceeds_clip", "source_out exceeds the validated clip duration.",
                shot_id=shot_id, expected=actual_duration, actual=source_out,
            ))
        if previous and start is not None:
            prior_end = _number_or_none(previous.get("timeline_end"))
            if prior_end is not None:
                delta = start - prior_end
                transition = str(previous.get("transition_out") or VIDEO_COMPILATION_CONFIG["default_transition"])
                if delta > epsilon:
                    issues.append(_issue("timeline_gap", "An unintended gap exists between timeline items.", shot_id=shot_id, actual=delta))
                elif delta < -epsilon:
                    allowed = transition == "dissolve" and abs(abs(delta) - float(VIDEO_COMPILATION_CONFIG["dissolve_duration_seconds"])) <= duration_tolerance
                    if not allowed:
                        issues.append(_issue("timeline_overlap", "An unintended overlap exists between timeline items.", shot_id=shot_id, actual=abs(delta)))
        transition = str(item.get("transition_out") or VIDEO_COMPILATION_CONFIG["default_transition"])
        if transition not in {"cut", "dissolve"}:
            issues.append(_issue("timeline_invalid", f"Unsupported transition: {transition}.", shot_id=shot_id))
        previous = item

    timeline_duration = _number_or_none(state.get("timeline_duration_seconds"))
    if timeline and timeline_duration is None:
        timeline_duration = _number_or_none(timeline[-1].get("timeline_end"))
    if timeline and _number_or_none(timeline[0].get("timeline_start")) is not None and abs(float(timeline[0]["timeline_start"])) > epsilon:
        issues.append(_issue("timeline_gap", "The visual timeline must start at zero.", actual=timeline[0]["timeline_start"]))
    if narration_duration is not None and timeline_duration is not None and abs(narration_duration - timeline_duration) > duration_tolerance:
        issues.append(_issue(
            "timeline_narration_duration_mismatch", "The edit timeline duration differs materially from the narration duration.",
            expected=narration_duration, actual=timeline_duration,
        ))
    return timeline, issues, timeline_duration


def build_ffmpeg_command(timeline: list[dict], narration_file: str, output_file: str, *, width: int, height: int, fps: float, duration: float) -> list[str]:
    inputs = [part for item in timeline for part in ("-i", str(item["video_file"]))]
    inputs.extend(["-i", narration_file])
    filters = []
    for index, item in enumerate(timeline):
        filters.append(
            f"[{index}:v]trim=start={float(item['source_in']):.6f}:end={float(item['source_out']):.6f},"
            f"setpts=PTS-STARTPTS,scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},fps={fps},format={VIDEO_COMPILATION_CONFIG['pixel_format']}[v{index}]"
        )
    current = "[v0]"
    for index in range(1, len(timeline)):
        output = f"[joined{index}]"
        transition = str(timeline[index - 1].get("transition_out") or VIDEO_COMPILATION_CONFIG["default_transition"])
        if transition == "dissolve":
            filters.append(
                f"{current}[v{index}]xfade=transition=fade:duration={VIDEO_COMPILATION_CONFIG['dissolve_duration_seconds']}:"
                f"offset={float(timeline[index]['timeline_start']):.6f}{output}"
            )
        else:
            filters.append(f"{current}[v{index}]concat=n=2:v=1:a=0{output}")
        current = output
    if len(timeline) == 1:
        filters.append("[v0]null[vout]")
        current = "[vout]"
    audio_index = len(timeline)
    filters.append(
        f"[{audio_index}:a:0]aresample={VIDEO_COMPILATION_CONFIG['audio_sample_rate']},"
        f"asetpts=PTS-STARTPTS,apad=pad_dur={duration:.6f},atrim=duration={duration:.6f}[aout]"
    )
    return [
        "ffmpeg", "-y", *inputs, "-filter_complex", ";".join(filters),
        "-map", current, "-map", "[aout]", "-t", f"{duration:.6f}",
        "-c:v", VIDEO_COMPILATION_CONFIG["video_codec"], "-pix_fmt", VIDEO_COMPILATION_CONFIG["pixel_format"],
        "-r", str(fps), "-c:a", VIDEO_COMPILATION_CONFIG["audio_codec"],
        "-ar", str(VIDEO_COMPILATION_CONFIG["audio_sample_rate"]), "-movflags", "+faststart", output_file,
    ]


def _run_ffmpeg(command: list[str]) -> None:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True,
            timeout=float(VIDEO_COMPILATION_CONFIG["ffmpeg_timeout_seconds"]),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(str(exc)) from exc
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "FFmpeg rough-cut compilation failed.")


def compile_rough_video(state: AgentState) -> dict:
    if not VIDEO_COMPILATION_CONFIG["enabled"]:
        return _failure(state, [_issue("ffmpeg_failed", "Rough-cut compilation is disabled.")], "runtime", "Compilation aborted before FFmpeg execution.", [])
    timeline, issues, timeline_duration = _preconditions(state)
    if issues:
        source = "source_video" if any(issue["type"] == "video_file_missing" for issue in issues) else "timeline" if any(issue["type"].startswith("timeline_") or issue["type"] in {"invalid_source_range", "source_range_exceeds_clip", "invalid_timeline_range"} for issue in issues) else "runtime"
        return _failure(state, issues, source, "Compilation aborted before FFmpeg execution.", timeline)

    aspect_ratio = str(state.get("aspect_ratio", "9:16"))
    resolution = VIDEO_COMPILATION_CONFIG["resolutions"].get(aspect_ratio, VIDEO_COMPILATION_CONFIG["resolutions"]["9:16"])
    width, height = int(resolution["width"]), int(resolution["height"])
    fps = float(VIDEO_COMPILATION_CONFIG["fps"])
    out = Path(state.get("output_dir", "outputs")) / "rough_cut"
    out.mkdir(parents=True, exist_ok=True)
    rough_cut = out / "rough_cut_v001.mp4"
    command = build_ffmpeg_command(
        timeline, state["narration_file"], str(rough_cut),
        width=width, height=height, fps=fps, duration=float(timeline_duration),
    )
    try:
        _run_ffmpeg(command)
    except RuntimeError as exc:
        return _failure(state, [_issue("ffmpeg_failed", "FFmpeg could not compile the approved edit timeline.")], "runtime", str(exc), timeline)

    if not rough_cut.is_file() or rough_cut.stat().st_size == 0:
        return _failure(state, [_issue("output_file_missing", "FFmpeg did not create a nonempty rough-cut file.")], "runtime", "Compilation output validation failed.", timeline)
    try:
        probe = _probe(rough_cut)
    except RuntimeError as exc:
        return _failure(state, [_issue("output_decode_failed", f"The rough cut could not be probed: {exc}")], "runtime", "Compilation output validation failed.", timeline)
    video_stream = next((stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"), None)
    audio_stream = next((stream for stream in probe.get("streams", []) if stream.get("codec_type") == "audio"), None)
    actual_duration = _duration(probe)
    actual_fps = _fps(video_stream or {})
    post_issues = []
    if not video_stream or not _decodes(rough_cut):
        post_issues.append(_issue("output_decode_failed", "The rough-cut video stream is missing or cannot be decoded."))
    if not audio_stream:
        post_issues.append(_issue("narration_stream_missing", "The rough cut does not contain the narration audio stream."))
    if video_stream and (video_stream.get("width") != width or video_stream.get("height") != height or actual_fps is None or abs(actual_fps - fps) > 0.01):
        post_issues.append(_issue("normalization_failed", "The rough cut does not match configured width, height, or FPS."))
    if actual_duration is None or abs(actual_duration - float(timeline_duration)) > float(VIDEO_COMPILATION_CONFIG["duration_tolerance_seconds"]):
        post_issues.append(_issue("final_duration_mismatch", "The rough-cut duration differs from the approved timeline.", expected=timeline_duration, actual=actual_duration))
    if post_issues:
        return _failure(state, post_issues, "runtime", "Compilation output validation failed.", timeline)

    result = {
        "compilation_status": "success",
        "rough_cut_file": str(rough_cut),
        "narration_file": state["narration_file"],
        "duration_seconds": actual_duration,
        "width": width,
        "height": height,
        "fps": actual_fps,
        "shot_timestamps": _shot_timestamps(timeline),
        "sync_valid": True,
        "compilation_issues": [],
        "compilation_error": None,
        "rough_cut_failure_source": None,
        "rough_cut_retry_count": 0,
        "rough_cut_retry_exhausted": False,
        "video_retry_shots": [],
        "pipeline_status": "rough_cut_compiled",
    }
    logger.info("rough_cut_compilation %s", json.dumps({
        "status": "success", "file": str(rough_cut), "duration_seconds": actual_duration,
        "resolution": [width, height], "fps": actual_fps, "shots": len(timeline),
    }, sort_keys=True))
    return result
