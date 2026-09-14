import logging
from pathlib import Path

import yaml

from schema import AgentState


logger = logging.getLogger(__name__)
with (Path(__file__).parent / "config" / "edit_timeline.yml").open(encoding="utf-8") as config_file:
    EDIT_TIMELINE_CONFIG = yaml.safe_load(config_file)["edit_timeline"]


def plan_edit_timeline(state: AgentState) -> dict:
    if not EDIT_TIMELINE_CONFIG["enabled"]:
        raise RuntimeError("Edit timeline planning is disabled in config/edit_timeline.yml.")
    shots = state.get("shot_plan") or []
    clips = state.get("generated_videos") or []
    validations = {item["shot_id"]: item for item in state.get("video_validation_results", [])}
    if not shots or [item.get("shot_id") for item in clips] != [item.get("shot_id") for item in shots]:
        raise RuntimeError("Edit Timeline Planner needs one ordered validated clip per approved shot.")
    if any(not validations.get(shot["shot_id"], {}).get("valid") for shot in shots):
        raise RuntimeError("Edit Timeline Planner accepts only technically validated clips.")

    narration_duration = float(state.get("actual_narration_seconds") or 0)
    planned = [float(shot.get("estimated_duration_seconds") or shot.get("duration_seconds") or 0) for shot in shots]
    if narration_duration <= 0 or any(duration <= 0 for duration in planned):
        raise RuntimeError("Edit Timeline Planner needs positive narration and shot durations.")
    scale = narration_duration / sum(planned)
    tolerance = float(EDIT_TIMELINE_CONFIG["clip_shortfall_tolerance_seconds"])
    retry_counts = dict(state.get("edit_timeline_retry_counts", {}))
    maximum = int(EDIT_TIMELINE_CONFIG["max_retries_per_shot"])
    timeline, issues, retry_requests, retry_shots, exhausted = [], [], [], [], []
    cursor = 0.0

    for index, (shot, clip, planned_duration) in enumerate(zip(shots, clips, planned), start=1):
        required = narration_duration - cursor if index == len(shots) else planned_duration * scale
        actual = float(validations[shot["shot_id"]]["metadata"]["duration_seconds"] or 0)
        if actual + tolerance < required:
            issue = {
                "type": "insufficient_clip_duration",
                "shot_id": shot["shot_id"],
                "required_duration_seconds": round(required, 3),
                "actual_usable_duration_seconds": round(actual, 3),
                "description": "The approved clip is too short to cover its assigned narration interval.",
            }
            issues.append(issue)
            retry_requests.append({
                "shot_id": shot["shot_id"],
                "retry_target": "video_generation",
                "reason": "Generate a longer usable clip for the same approved shot.",
            })
            count = retry_counts.get(shot["shot_id"], 0)
            if count < maximum:
                retry_counts[shot["shot_id"]] = count + 1
                retry_shots.append(shot["shot_id"])
            else:
                exhausted.append(shot["shot_id"])
        end = cursor + required
        timeline.append({
            "timeline_item_id": f"edit-{index:03}",
            "shot_id": shot["shot_id"],
            "scene_id": shot["scene_id"],
            "video_file": clip["video_file"],
            "source_in": 0.0,
            "source_out": round(min(actual, required), 3),
            "timeline_start": round(cursor, 3),
            "timeline_end": round(end, 3),
            "narration_segment_ids": list(shot.get("source_segment_ids", [])),
            "visual_beat_ids": list(shot.get("visual_beat_ids", [])),
            "transition_in": EDIT_TIMELINE_CONFIG["default_transition"],
            "transition_out": EDIT_TIMELINE_CONFIG["default_transition"],
            "edit_reason": str(shot.get("shot_purpose") or shot.get("visual_action") or "Preserve approved shot order."),
        })
        cursor = end

    valid = not issues
    logger.info("edit_timeline valid=%s retry_shots=%s exhausted_shots=%s", valid, retry_shots, exhausted)
    return {
        "edit_timeline": timeline,
        "narration_duration_seconds": narration_duration,
        "timeline_duration_seconds": round(cursor, 3),
        "timeline_valid": valid,
        "timeline_issues": issues,
        "timeline_retry_requests": retry_requests,
        "edit_timeline_retry_counts": retry_counts,
        "edit_timeline_exhausted_shots": exhausted,
        "video_retry_shots": retry_shots,
        "pipeline_status": "edit_timeline_failed" if exhausted else "edit_timeline_retry" if retry_shots else "edit_timeline_planned",
    }
