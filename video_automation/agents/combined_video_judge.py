import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, model_validator

from video_automation.models import invoke_with_images_and_evaluation, model_name_for
from video_automation.schema import AgentState
from video_automation.agents.story_agent import _parse_json


logger = logging.getLogger(__name__)
with (Path(__file__).parents[2] / "config" / "combined_video_judge.yml").open(encoding="utf-8") as config_file:
    COMBINED_VIDEO_JUDGE_CONFIG = yaml.safe_load(config_file)["combined_video_judge"]

Severity = Literal["minor", "major", "critical"]
FailureSource = Literal["source_image", "motion_plan", "video_generation", "edit_timeline", "none"]
RetryTarget = Literal["image_generation", "motion_planner", "video_generation", "plan_edit_timeline", "none"]
NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
ROUTES = {
    "source_image": "image_generation",
    "motion_plan": "motion_planner",
    "video_generation": "video_generation",
    "edit_timeline": "plan_edit_timeline",
    "none": "none",
}
CRITICAL_METRICS = (
    "character_consistency", "temporal_consistency", "narration_alignment", "story_faithfulness",
)


class CombinedScores(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    visual_fidelity: Annotated[int, Field(ge=1, le=10)]
    character_consistency: Annotated[int, Field(ge=1, le=10)]
    motion_quality: Annotated[int, Field(ge=1, le=10)]
    temporal_consistency: Annotated[int, Field(ge=1, le=10)]
    shot_alignment: Annotated[int, Field(ge=1, le=10)]
    narration_alignment: Annotated[int, Field(ge=1, le=10)]
    story_faithfulness: Annotated[int, Field(ge=1, le=10)]
    emotional_intent: Annotated[int, Field(ge=1, le=10)]
    pacing_and_editing: Annotated[int, Field(ge=1, le=10)]
    overall_coherence: Annotated[int, Field(ge=1, le=10)]


class RawCombinedIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: NonEmpty
    severity: Severity
    start_sec: Annotated[float, Field(ge=0)] | None
    end_sec: Annotated[float, Field(ge=0)] | None
    description: NonEmpty
    failure_source: FailureSource
    retry_target: RetryTarget
    correction: NonEmpty

    @model_validator(mode="after")
    def validate_range_and_route(self):
        if (self.start_sec is None) != (self.end_sec is None):
            raise ValueError("start_sec and end_sec must both be null or both be numbers")
        if self.start_sec is not None and self.start_sec > self.end_sec:
            raise ValueError("start_sec must not exceed end_sec")
        if ROUTES[self.failure_source] != self.retry_target:
            raise ValueError("failure_source and retry_target disagree")
        return self


class RawCombinedReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    approved: bool
    overall_score: Annotated[float, Field(ge=1, le=10)]
    scores: CombinedScores
    issues: list[RawCombinedIssue]
    primary_failure_source: FailureSource
    retry_target: RetryTarget
    refinement_needed: bool
    summary: NonEmpty


COMBINED_RESPONSE_SCHEMA = RawCombinedReport.model_json_schema()


def map_timestamp_to_shots(
    start_sec: float | None,
    end_sec: float | None,
    shot_timestamps: list[dict],
) -> list[str]:
    shots = [item for item in shot_timestamps if item.get("shot_id")]
    if not shots:
        raise ValueError("shot_timestamps is empty")
    if start_sec is None and end_sec is None:
        return list(dict.fromkeys(str(item["shot_id"]) for item in shots))
    if start_sec is None or end_sec is None or start_sec < 0 or end_sec < start_sec:
        raise ValueError("issue timestamp range is invalid")

    if start_sec == end_sec:
        for index, item in enumerate(shots):
            start, end = float(item["start_sec"]), float(item["end_sec"])
            if start <= start_sec < end or index == len(shots) - 1 and start_sec == end:
                return [str(item["shot_id"])]
        return []
    return list(dict.fromkeys(
        str(item["shot_id"])
        for item in shots
        if end_sec > float(item["start_sec"]) and start_sec < float(item["end_sec"])
    ))


def validate_combined_judge_report(value: object, video_duration: float, shot_timestamps: list[dict]) -> dict:
    try:
        report = RawCombinedReport.model_validate(value)
    except ValidationError as exc:
        raise RuntimeError(f"Combined Video Judge returned invalid structured output: {exc}") from exc

    threshold = int(COMBINED_VIDEO_JUDGE_CONFIG["approval_threshold"])
    tolerance = float(COMBINED_VIDEO_JUDGE_CONFIG["timestamp_tolerance_seconds"])
    issues, problem_shots = [], []
    for raw_issue in report.issues:
        if raw_issue.end_sec is not None and raw_issue.end_sec > video_duration + tolerance:
            raise RuntimeError("Combined Video Judge returned a timestamp beyond the rough-cut duration.")
        shot_ids = map_timestamp_to_shots(raw_issue.start_sec, raw_issue.end_sec, shot_timestamps)
        if not shot_ids:
            raise RuntimeError("Combined Video Judge issue timestamp does not overlap a known shot.")
        issue = raw_issue.model_dump()
        issue["shot_ids"] = shot_ids
        issues.append(issue)
        problem_shots.extend(shot_ids)

    source, target = report.primary_failure_source, report.retry_target
    if ROUTES[source] != target:
        raise RuntimeError("Combined Video Judge primary failure source and retry target disagree.")
    scores = report.scores.model_dump()
    if report.approved:
        if report.overall_score < threshold or any(scores[metric] < threshold for metric in CRITICAL_METRICS):
            raise RuntimeError("Combined Video Judge approved a result below the configured critical threshold.")
        if any(issue.severity == "critical" for issue in report.issues):
            raise RuntimeError("Combined Video Judge approved a result containing a critical issue.")
        if source != "none" or target != "none" or report.refinement_needed:
            raise RuntimeError("An approved Combined Video Judge result cannot request refinement.")
    else:
        if source == "none" or target == "none" or not report.refinement_needed or not report.issues:
            raise RuntimeError("A rejected Combined Video Judge result needs issues and a retry route.")
        if not any(issue.failure_source == source and issue.retry_target == target for issue in report.issues):
            raise RuntimeError("No issue supports the Combined Video Judge primary retry route.")

    result = report.model_dump()
    result["issues"] = issues
    result["problem_shot_ids"] = list(dict.fromkeys(problem_shots))
    return result


def _narration_segments(state: AgentState) -> list[dict]:
    timing_by_id = {
        item.get("segment_id"): item
        for item in state.get("narration_segment_timings", [])
        if isinstance(item, dict) and item.get("segment_id")
    }
    return [
        {
            **segment,
            **({
                "start_sec": timing_by_id[segment.get("segment_id")].get("start_seconds"),
                "end_sec": timing_by_id[segment.get("segment_id")].get("end_seconds"),
            } if segment.get("segment_id") in timing_by_id else {}),
        }
        for segment in state.get("narration_segments", [])
        if isinstance(segment, dict)
    ]


def _reference_media(state: AgentState) -> tuple[list[str], list[dict]]:
    references, labels = [], []
    shots = state.get("shot_plan", [])
    for index, file_name in enumerate(state.get("image_files", [])):
        if file_name and Path(file_name).is_file():
            references.append(file_name)
            labels.append({"kind": "approved_shot_image", "shot_id": shots[index].get("shot_id") if index < len(shots) else None})
    for file_name in state.get("character_reference_files", []):
        if file_name and Path(file_name).is_file() and file_name not in references:
            references.append(file_name)
            labels.append({"kind": "character_reference"})
    return references, labels


def extract_video_frames(video_file: str, duration: float, output_dir: str) -> tuple[list[str], list[float]]:
    count = int(COMBINED_VIDEO_JUDGE_CONFIG["sample_frames"])
    if count <= 0 or duration <= 0:
        raise RuntimeError("Combined Video Judge frame sampling configuration is invalid.")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output = Path(output_dir) / "frame-%03d.jpg"
    rate = count / duration
    command = [
        "ffmpeg", "-v", "error", "-i", video_file, "-an",
        "-vf", f"fps={rate:.8f},scale={int(COMBINED_VIDEO_JUDGE_CONFIG['frame_max_width'])}:-2:force_original_aspect_ratio=decrease",
        "-frames:v", str(count), "-q:v", "3", str(output),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=float(COMBINED_VIDEO_JUDGE_CONFIG["ffmpeg_timeout_seconds"]),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Could not sample rough-cut frames: {exc}") from exc
    if completed.returncode:
        raise RuntimeError(f"Could not sample rough-cut frames: {completed.stderr.strip()}")
    frames = sorted(str(path) for path in Path(output_dir).glob("frame-*.jpg"))
    if not frames:
        raise RuntimeError("FFmpeg did not extract any rough-cut frames.")
    return frames, [round(min(duration, index / rate), 3) for index in range(len(frames))]


def _image_retry_qa(state: AgentState, shot_ids: list[str], issues: list[dict]) -> list[dict]:
    by_id = {item["shot_id"]: item for item in state.get("shot_image_qa_results", []) if item.get("shot_id")}
    for shot_id in shot_ids:
        corrections = [item["correction"] for item in issues if shot_id in item["shot_ids"]]
        by_id[shot_id] = {
            "shot_id": shot_id,
            "approved": False,
            "animation_ready": False,
            "issues": [{
                "type": "generation_artifact",
                "severity": "major",
                "description": "The combined video judge traced the failure to the approved source image.",
                "correction": " ".join(corrections),
            }],
            "retry_target": "image_generation",
        }
    return [by_id[shot["shot_id"]] for shot in state.get("shot_plan", []) if shot.get("shot_id") in by_id]


def _controlled_failure(state: AgentState, error: Exception | str, *, model: str | None) -> dict:
    failures = int(state.get("combined_video_judge_error_count", 0)) + 1
    exhausted = failures > int(COMBINED_VIDEO_JUDGE_CONFIG["max_inference_retries"])
    logger.error("combined_video_judge failed model=%s attempt=%s exhausted=%s error=%s", model, failures, exhausted, error)
    result = {
        "combined_video_judge_report": None,
        "combined_video_judge_approved": False,
        "combined_video_judge_status": "error",
        "combined_video_judge_error": str(error),
        "combined_video_judge_error_count": failures,
        "combined_video_judge_error_exhausted": exhausted,
        "combined_video_judge_retry_target": "none",
        "combined_video_judge_retry_shots": [],
        "video_judge_refinement_exhausted": False,
        "video_retry_shots": [],
        "motion_plan_retry_shots": [],
        "shot_image_qa_retry_shots": [],
        "pipeline_status": "combined_video_judge_unavailable" if exhausted else "combined_video_judge_api_retry",
    }
    if exhausted:
        result["warnings"] = [
            *state.get("warnings", []),
            f"Final video judge unavailable after {failures} attempts; continuing with the validated rough cut: {error}",
        ]
    return result


def combined_video_judge(state: AgentState) -> dict:
    model = model_name_for("combined-video-judge")
    rough_cut = state.get("rough_cut_file")
    if not COMBINED_VIDEO_JUDGE_CONFIG["enabled"]:
        return _controlled_failure(state, "Combined Video Judge is disabled.", model=model)
    if os.getenv("MODEL_PROVIDER", "ollama").lower() != "ollama":
        return _controlled_failure(state, "Combined Video Judge currently requires MODEL_PROVIDER=ollama.", model=model)
    if not rough_cut or not Path(rough_cut).is_file() or Path(rough_cut).stat().st_size == 0:
        return _controlled_failure(state, "The compiled rough-cut video is missing or empty.", model=model)
    timestamps = state.get("shot_timestamps") or []
    duration = state.get("duration_seconds")
    if not timestamps or duration is None or float(duration) <= 0:
        return _controlled_failure(state, "Rough-cut duration and shot timestamps are required.", model=model)

    started = time.monotonic()
    try:
        references, reference_order = _reference_media(state)
        with tempfile.TemporaryDirectory(prefix="combined-video-judge-") as frame_dir:
            frames, frame_times = extract_video_frames(rough_cut, float(duration), frame_dir)
            reference_start = len(frames) + 1
            payload = {
                "approved_story": state.get("story_outline") or state.get("story"),
                "narration_segments": _narration_segments(state),
                "shot_plan": state.get("shot_plan", []),
                "motion_plans": state.get("motion_plans", []),
                "edit_timeline": state.get("edit_timeline", []),
                "shot_timestamps": timestamps,
                "rough_cut_metadata": {
                    "duration_seconds": duration,
                    "width": state.get("width"),
                    "height": state.get("height"),
                    "fps": state.get("fps"),
                },
                "sampled_video_frames": [
                    {"image_index": index + 1, "timestamp_sec": timestamp}
                    for index, timestamp in enumerate(frame_times)
                ],
                "reference_image_order": [
                    {**label, "image_index": reference_start + index}
                    for index, label in enumerate(reference_order)
                ],
            }
            prompt = (
                f"{COMBINED_VIDEO_JUDGE_CONFIG['system_prompt'].strip()}\n\n"
                f"Required JSON schema:\n{json.dumps(COMBINED_RESPONSE_SCHEMA, ensure_ascii=False)}\n\n"
                f"Approved inputs:\n{json.dumps(payload, ensure_ascii=False)}"
            )
            response, evaluation = invoke_with_images_and_evaluation(
                "combined-video-judge",
                prompt,
                [*frames, *references],
                purpose="judge_compiled_video",
            )
            raw = _parse_json(response.content)
        report = validate_combined_judge_report(raw, float(duration), timestamps)
    except Exception as exc:
        return _controlled_failure(state, exc, model=model)

    evaluations = list(state.get("llm_evaluations", []))
    evaluation["call_number"] = len(evaluations) + 1
    evaluations.append(evaluation)
    current_round = int(state.get("video_judge_refinement_round", 0))
    exhausted = not report["approved"] and current_round >= int(COMBINED_VIDEO_JUDGE_CONFIG["max_refinement_rounds"])
    refinement_round = current_round if report["approved"] or exhausted else current_round + 1
    target = "none" if exhausted else report["retry_target"]
    retry_shots = [
        shot_id for shot_id in report["problem_shot_ids"]
        if any(shot_id in issue["shot_ids"] and issue["retry_target"] == report["retry_target"] for issue in report["issues"])
    ]
    if exhausted:
        retry_shots = []
    feedback = {
        shot_id: " ".join(
            issue["correction"] for issue in report["issues"]
            if shot_id in issue["shot_ids"] and issue["retry_target"] == report["retry_target"]
        )
        for shot_id in retry_shots
    }
    result = {
        "combined_video_judge_report": report,
        "combined_video_judge_approved": report["approved"],
        "combined_video_judge_status": "accepted" if report["approved"] else "exhausted" if exhausted else "retry",
        "combined_video_judge_error": None,
        "combined_video_judge_error_count": 0,
        "combined_video_judge_error_exhausted": False,
        "combined_video_judge_model": model,
        "video_judge_refinement_round": refinement_round,
        "video_judge_refinement_exhausted": exhausted,
        "combined_video_judge_retry_target": target,
        "combined_video_judge_retry_shots": retry_shots,
        "video_retry_shots": retry_shots if target in {"image_generation", "motion_planner", "video_generation"} else [],
        "motion_plan_retry_shots": retry_shots if target == "motion_planner" else [],
        "motion_plan_retry_feedback": feedback if target == "motion_planner" else {},
        "shot_image_qa_retry_shots": retry_shots if target == "image_generation" else [],
        "llm_evaluations": evaluations,
        "pipeline_status": "combined_video_judge_accepted" if report["approved"] else "combined_video_judge_failed" if exhausted else "combined_video_judge_retry",
    }
    if target == "image_generation":
        result["shot_image_qa_results"] = _image_retry_qa(state, retry_shots, report["issues"])
    logger.info("combined_video_judge %s", json.dumps({
        "model": model,
        "video": rough_cut,
        "overall_score": report["overall_score"],
        "scores": report["scores"],
        "issue_count": len(report["issues"]),
        "affected_shot_ids": report["problem_shot_ids"],
        "retry_target": target,
        "refinement_round": refinement_round,
        "approved": report["approved"],
        "latency_seconds": round(time.monotonic() - started, 3),
    }, sort_keys=True))
    return result
