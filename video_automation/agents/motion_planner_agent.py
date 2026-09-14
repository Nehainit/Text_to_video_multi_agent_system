import json
import re
from pathlib import Path

from video_automation.models import invoke_with_evaluation, load_model
from video_automation.prompts import MOTION_PLANNING_CONFIG, MOTION_PLANNING_SYSTEM_PROMPT
from video_automation.schema import AgentState
from video_automation.agents.story_agent import _parse_json


INTENSITIES = {"none", "very_low", "low", "medium", "high"}
CAMERA_MOTIONS = {"static", "subtle_push_in", "subtle_pull_back", "slow_pan", "gentle_tilt", "small_lateral_tracking"}


def _validate_motion_plan(value: object, shot_id: str, duration: float) -> dict:
    if not isinstance(value, dict) or type(value.get("needs_revision")) is not bool:
        raise RuntimeError("Motion Planner returned an invalid result.")
    if value["needs_revision"]:
        if not isinstance(value.get("revision_reason"), str) or not value["revision_reason"].strip():
            raise RuntimeError("A motion revision needs a specific reason.")
        return {"shot_id": shot_id, "needs_revision": True, "revision_reason": value["revision_reason"].strip()}

    fields = {
        "subject_motion", "camera_motion", "environment_motion",
        "preserve_identity", "preserve_composition", "video_prompt", "negative_prompt",
        "needs_revision", "revision_reason",
    }
    if not fields <= set(value) or value["revision_reason"] not in (None, ""):
        raise RuntimeError("Approved motion plan does not match the required schema.")
    motion_fields = {"description", "intensity"}
    for key in ("subject_motion", "environment_motion"):
        motion = value[key]
        if not isinstance(motion, dict) or set(motion) != motion_fields or motion.get("intensity") not in INTENSITIES or not str(motion.get("description", "")).strip():
            raise RuntimeError(f"{key} is invalid.")
    camera = value["camera_motion"]
    if not isinstance(camera, dict) or set(camera) != motion_fields | {"type"} or camera.get("type") not in CAMERA_MOTIONS or camera.get("intensity") not in INTENSITIES or not str(camera.get("description", "")).strip():
        raise RuntimeError("camera_motion is invalid.")
    if value["preserve_identity"] is not True or value["preserve_composition"] is not True:
        raise RuntimeError("Motion must preserve identity and composition.")
    if any(not isinstance(value[key], str) or not value[key].strip() for key in ("video_prompt", "negative_prompt")):
        raise RuntimeError("Motion plan needs nonempty video and negative prompts.")
    if re.search(r"\b(?:cut to|scene change|montage)\b", value["video_prompt"], re.IGNORECASE):
        raise RuntimeError("Video prompt cannot introduce cuts, scene changes, or montage behavior.")
    return {
        "shot_id": shot_id,
        "duration_seconds": duration,
        **{key: value[key] for key in fields},
    }


def create_motion_plans(state: AgentState) -> dict:
    if not MOTION_PLANNING_CONFIG["enabled"]:
        raise RuntimeError("Motion Planner is disabled in config/motion_planning.yml.")
    shots = state.get("shot_plan") or []
    requests = state.get("image_prompt_requests") or []
    images = state.get("image_files") or []
    qa_results = state.get("shot_image_qa_results") or []
    if not shots or not (len(shots) == len(requests) == len(images) == len(qa_results)):
        raise RuntimeError("Motion Planner needs one approved image, prompt request, and image QA result per shot.")

    existing = {item["shot_id"]: item for item in state.get("motion_plans", [])}
    visual_feedback_targets = {
        str(item.get("shot_id")) for item in state.get("last_visual_feedback", []) if isinstance(item, dict)
    }
    feedback_targets = set(state.get("motion_plan_retry_shots", [])) or visual_feedback_targets
    feedback_by_id = {
        str(item.get("shot_id")): str(item.get("note", "")).strip()
        for item in state.get("last_visual_feedback", []) if isinstance(item, dict)
    }
    feedback_by_id.update(state.get("motion_plan_retry_feedback", {}))
    targets = feedback_targets or ({shot["shot_id"] for shot in shots} if not existing else set())
    timing_by_id = {
        item.get("segment_id"): item for item in state.get("narration_segment_timings", []) if isinstance(item, dict)
    }
    evaluations = list(state.get("llm_evaluations", []))
    model = None
    plans = []
    revision_reasons = []

    for index, (shot, request, image_file, image_qa) in enumerate(zip(shots, requests, images, qa_results)):
        shot_id = shot["shot_id"]
        if shot_id not in targets and shot_id in existing:
            plans.append(existing[shot_id])
            continue
        if not image_qa.get("approved") or not image_qa.get("animation_ready") or not image_file or not Path(image_file).is_file():
            revision_reasons.append(f"{shot_id}: approved animation-ready image is required.")
            continue
        duration = float(shot.get("estimated_duration_seconds") or shot.get("duration_seconds") or 5)
        segment_ids = list(shot.get("source_segment_ids") or request.get("source_segment_ids") or [])
        payload = {
            "approved_shot": shot,
            "approved_image_prompt_request": request,
            "approved_image_file": image_file,
            "shot_duration_seconds": duration,
            "narration_segment_ids": segment_ids,
            "narration_timing": [timing_by_id[item] for item in segment_ids if item in timing_by_id],
            "character_reference_ids": request.get("character_reference_ids", []),
            "location_id": request.get("location_id"),
            "image_qa_result": image_qa,
            "retry_feedback": feedback_by_id.get(shot_id, ""),
        }
        error = ""
        model = model or load_model("motion-planner")
        for _ in range(int(MOTION_PLANNING_CONFIG["max_retries"]) + 1):
            response, evaluation = invoke_with_evaluation(
                model,
                [
                    {"role": "system", "content": MOTION_PLANNING_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps({**payload, "retry_feedback": error}, ensure_ascii=False)},
                ],
                agent_name="motion-planner",
                purpose="plan_shot_motion",
            )
            evaluation["call_number"] = len(evaluations) + 1
            evaluations.append(evaluation)
            try:
                plan = _validate_motion_plan(_parse_json(response.content), shot_id, duration)
                if plan["needs_revision"]:
                    revision_reasons.append(f"{shot_id}: {plan['revision_reason']}")
                else:
                    plans.append(plan)
                break
            except (RuntimeError, ValueError, TypeError) as exc:
                error = str(exc)
        else:
            raise RuntimeError(error or f"Motion Planner failed for {shot_id}.")

    return {
        "motion_plans": plans,
        "motion_plan_needs_revision": bool(revision_reasons),
        "motion_plan_revision_reason": "; ".join(revision_reasons) or None,
        "shot_plan_needs_revision": bool(revision_reasons),
        "shot_plan_revision_reason": "; ".join(revision_reasons) or None,
        "shot_plan_issues": revision_reasons,
        "last_visual_feedback": [],
        "motion_plan_retry_shots": [],
        "motion_plan_retry_feedback": {},
        "llm_evaluations": evaluations,
    }
