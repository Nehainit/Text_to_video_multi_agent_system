import json
import logging
import math
import subprocess
from fractions import Fraction
from pathlib import Path

from prompts import VIDEO_VALIDATION_CONFIG
from schema import AgentState


logger = logging.getLogger(__name__)


def _metadata(file_size: int = 0) -> dict:
    return {
        "duration_seconds": None,
        "width": None,
        "height": None,
        "aspect_ratio": None,
        "fps": None,
        "frame_count": None,
        "video_codec": None,
        "has_video_stream": False,
        "has_audio_stream": False,
        "file_size_bytes": file_size,
    }


def _issue(issue_type: str, description: str, *, expected=None, actual=None) -> dict:
    issue = {"type": issue_type, "description": description}
    if expected is not None:
        issue["expected"] = expected
    if actual is not None:
        issue["actual"] = actual
    return issue


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _fps(stream: dict) -> float | None:
    for value in (stream.get("avg_frame_rate"), stream.get("r_frame_rate")):
        try:
            number = float(Fraction(str(value)))
        except (ValueError, ZeroDivisionError):
            continue
        if math.isfinite(number) and number > 0:
            return number
    return None


def _probe(video: Path) -> dict:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)],
            capture_output=True,
            text=True,
            timeout=float(VIDEO_VALIDATION_CONFIG["probe_timeout_seconds"]),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(str(exc)) from exc
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "ffprobe could not parse the video container.")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe returned invalid JSON.") from exc
    if not isinstance(data, dict) or not isinstance(data.get("streams"), list):
        raise RuntimeError("ffprobe returned incomplete metadata.")
    return data


def _decodes(video: Path) -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:v:0", "-f", "null", "-"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=float(VIDEO_VALIDATION_CONFIG["decode_timeout_seconds"]),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def validate_video(
    shot_id: str,
    video_file: str | None,
    *,
    requested_duration_seconds: float | None = None,
    expected_width: int | None = None,
    expected_height: int | None = None,
    expected_aspect_ratio: float | None = None,
) -> dict:
    video = Path(video_file) if video_file else None
    if not video or not video.is_file():
        return {"shot_id": shot_id, "valid": False, "issues": [_issue("video_file_missing", "Generated video file does not exist.")], "metadata": _metadata()}
    size = video.stat().st_size
    if size == 0:
        return {"shot_id": shot_id, "valid": False, "issues": [_issue("video_file_empty", "Generated video file is empty.")], "metadata": _metadata()}

    metadata = _metadata(size)
    try:
        probe = _probe(video)
    except RuntimeError as exc:
        return {"shot_id": shot_id, "valid": False, "issues": [_issue("video_probe_failed", f"Video container could not be parsed: {exc}")], "metadata": metadata}

    streams = probe["streams"]
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    metadata["has_audio_stream"] = any(stream.get("codec_type") == "audio" for stream in streams)
    issues = []
    if not video_stream:
        issues.append(_issue("video_stream_missing", "No video stream was found in the generated file."))
    else:
        metadata["has_video_stream"] = True
        metadata["video_codec"] = video_stream.get("codec_name")
        try:
            metadata["width"] = int(video_stream.get("width") or 0)
            metadata["height"] = int(video_stream.get("height") or 0)
        except (TypeError, ValueError):
            metadata["width"], metadata["height"] = 0, 0
        metadata["aspect_ratio"] = metadata["width"] / metadata["height"] if metadata["width"] > 0 and metadata["height"] > 0 else None
        metadata["fps"] = _fps(video_stream)
        raw_frame_count = video_stream.get("nb_frames")
        if raw_frame_count not in (None, "N/A"):
            try:
                metadata["frame_count"] = int(raw_frame_count)
            except (TypeError, ValueError):
                issues.append(_issue("invalid_frame_count", "Video frame count metadata is invalid.", actual=raw_frame_count))
        if metadata["aspect_ratio"] is None:
            issues.append(_issue("invalid_resolution", "Video width and height must both be greater than zero.", actual={"width": metadata["width"], "height": metadata["height"]}))
        if expected_width is not None and metadata["width"] != expected_width:
            issues.append(_issue("width_mismatch", "Video width does not match the expected width.", expected=expected_width, actual=metadata["width"]))
        if expected_height is not None and metadata["height"] != expected_height:
            issues.append(_issue("height_mismatch", "Video height does not match the expected height.", expected=expected_height, actual=metadata["height"]))
        if expected_aspect_ratio is not None and metadata["aspect_ratio"] is not None and abs(metadata["aspect_ratio"] - expected_aspect_ratio) > float(VIDEO_VALIDATION_CONFIG["aspect_ratio_tolerance"]):
            issues.append(_issue("aspect_ratio_mismatch", "Video aspect ratio exceeds the configured tolerance.", expected=expected_aspect_ratio, actual=metadata["aspect_ratio"]))
        if metadata["fps"] is None:
            issues.append(_issue("invalid_fps", "Video FPS is missing or not greater than zero."))
        if metadata["frame_count"] is not None and metadata["frame_count"] <= 0:
            issues.append(_issue("invalid_frame_count", "Video frame count is not greater than zero.", actual=metadata["frame_count"]))

    format_data = probe.get("format") if isinstance(probe.get("format"), dict) else {}
    metadata["duration_seconds"] = _number(video_stream.get("duration")) if video_stream else None
    metadata["duration_seconds"] = metadata["duration_seconds"] or _number(format_data.get("duration"))
    if metadata["duration_seconds"] is None:
        issues.append(_issue("invalid_duration", "Video duration is missing or not greater than zero."))
    elif requested_duration_seconds is not None and abs(metadata["duration_seconds"] - requested_duration_seconds) > float(VIDEO_VALIDATION_CONFIG["duration_tolerance_seconds"]):
        issues.append(_issue("duration_mismatch", "Generated video duration exceeds the configured tolerance.", expected=requested_duration_seconds, actual=metadata["duration_seconds"]))
    if VIDEO_VALIDATION_CONFIG["require_audio"] and not metadata["has_audio_stream"]:
        issues.append(_issue("audio_stream_missing", "An audio stream is required by configuration but was not found."))
    if video_stream and not _decodes(video):
        issues.append(_issue("video_decode_failed", "The video stream could not be decoded completely."))
    return {"shot_id": shot_id, "valid": not issues, "issues": issues, "metadata": metadata}


def _ratio(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) and value > 0 else None
    try:
        left, right = str(value).split(":", 1)
        return float(left) / float(right)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def validate_generated_videos(state: AgentState) -> dict:
    if not VIDEO_VALIDATION_CONFIG["enabled"]:
        raise RuntimeError("Video validation is disabled in config/video_validation.yml.")
    generated = state.get("generated_videos") or []
    shots = state.get("shot_plan") or []
    if not shots or [item.get("shot_id") for item in generated] != [item.get("shot_id") for item in shots]:
        raise RuntimeError("Video validation needs one ordered generated-video record per shot.")

    previous = {item["shot_id"]: item for item in state.get("video_validation_results", [])}
    targets = set(state.get("video_retry_shots", [])) if previous else {shot["shot_id"] for shot in shots}
    retry_counts = dict(state.get("video_validation_retry_counts", {}))
    max_retries = int(VIDEO_VALIDATION_CONFIG["max_retries"])
    results, retry_shots, exhausted_shots = [], [], []

    for item in generated:
        shot_id = item["shot_id"]
        if shot_id not in targets and shot_id in previous:
            result = previous[shot_id]
        else:
            result = validate_video(
                shot_id,
                item.get("video_file"),
                requested_duration_seconds=_number(item.get("generated_duration_seconds")),
                expected_width=item.get("expected_width"),
                expected_height=item.get("expected_height"),
                expected_aspect_ratio=_ratio(item.get("expected_aspect_ratio") or item.get("aspect_ratio") or state.get("aspect_ratio")),
            )
            if not result["valid"]:
                if retry_counts.get(shot_id, 0) < max_retries:
                    retry_counts[shot_id] = retry_counts.get(shot_id, 0) + 1
                    retry_shots.append(shot_id)
                else:
                    exhausted_shots.append(shot_id)
            metadata = result["metadata"]
            logger.info("video_validation %s", json.dumps({
                "shot_id": shot_id,
                "valid": result["valid"],
                "issue_types": [issue["type"] for issue in result["issues"]],
                "requested_duration_seconds": item.get("requested_duration_seconds"),
                "actual_duration_seconds": metadata["duration_seconds"],
                "resolution": [metadata["width"], metadata["height"]],
                "fps": metadata["fps"],
                "retry_count": retry_counts.get(shot_id, 0),
            }, sort_keys=True))
        results.append(result)

    return {
        "video_validation_results": results,
        "video_validation_retry_counts": retry_counts,
        "video_validation_exhausted_shots": exhausted_shots,
        "video_retry_shots": retry_shots,
        "scene_video_files": [item.get("video_file") if result["valid"] else None for item, result in zip(generated, results)],
        "pipeline_status": "video_validation_failed" if exhausted_shots else "video_validation_retry" if retry_shots else "video_validation_passed",
    }
