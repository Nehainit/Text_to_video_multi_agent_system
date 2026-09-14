import json
import logging
from pathlib import Path

from gemini_judge import invoke_gemini_judge
from prompts import VISUAL_FIDELITY_JUDGE_CONFIG, VISUAL_FIDELITY_JUDGE_SYSTEM_PROMPT
from schema import AgentState


logger = logging.getLogger(__name__)
METRICS = (
    "character_identity", "anatomy_integrity", "character_scale_consistency",
    "composition_preservation", "framing_preservation", "location_consistency",
    "style_consistency", "object_persistence", "visual_artifact_control",
    "temporal_visual_consistency",
)
SEVERITIES = {"minor", "major", "critical"}


def scored_report_schema(judge: str, metrics: tuple[str, ...]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["judge", "scores", "issues"],
        "properties": {
            "judge": {"type": "string", "enum": [judge]},
            "scores": {
                "type": "object", "additionalProperties": False, "required": list(metrics),
                "properties": {metric: {"type": "integer", "minimum": 1, "maximum": 10} for metric in metrics},
            },
            "issues": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["metric", "severity", "description", "timestamp_or_range"],
                    "properties": {
                        "metric": {"type": "string", "enum": list(metrics)},
                        "severity": {"type": "string", "enum": sorted(SEVERITIES)},
                        "description": {"type": "string", "minLength": 1},
                        "timestamp_or_range": {"type": ["string", "null"]},
                    },
                },
            },
        },
    }


RESPONSE_SCHEMA = scored_report_schema("visual_fidelity", METRICS)


def validate_scored_report(value: object, judge: str, metrics: tuple[str, ...]) -> dict:
    if not isinstance(value, dict) or set(value) != {"judge", "scores", "issues"} or value["judge"] != judge:
        raise RuntimeError(f"{judge} returned an invalid result schema.")
    scores = value["scores"]
    if not isinstance(scores, dict) or set(scores) != set(metrics) or any(
        type(score) is not int or not 1 <= score <= 10 for score in scores.values()
    ):
        raise RuntimeError(f"{judge} scores must be integers from 1 through 10.")
    issues = value["issues"]
    issue_fields = {"metric", "severity", "description", "timestamp_or_range"}
    if not isinstance(issues, list) or any(
        not isinstance(issue, dict) or set(issue) != issue_fields
        or issue["metric"] not in metrics or issue["severity"] not in SEVERITIES
        or not isinstance(issue["description"], str) or not issue["description"].strip()
        or (issue["timestamp_or_range"] is not None and (
            not isinstance(issue["timestamp_or_range"], str) or not issue["timestamp_or_range"].strip()
        ))
        for issue in issues
    ):
        raise RuntimeError(f"{judge} returned an invalid issue.")
    documented = {issue["metric"] for issue in issues}
    missing = [metric for metric, score in scores.items() if score < 8 and metric not in documented]
    if missing:
        raise RuntimeError(f"{judge} omitted evidence for low scores: {', '.join(missing)}.")
    return value


def _validate_report(value: object) -> dict:
    return validate_scored_report(value, "visual_fidelity", METRICS)


def judge_visual_fidelity(video_file: str, image_files: list[str], payload: dict) -> tuple[dict, dict]:
    report, evaluation = invoke_gemini_judge(
        agent_name="visual-fidelity-judge", purpose="judge_visual_fidelity",
        system_prompt=VISUAL_FIDELITY_JUDGE_SYSTEM_PROMPT, response_schema=RESPONSE_SCHEMA,
        payload=payload, video_file=video_file, image_files=image_files,
    )
    return _validate_report(report), evaluation


def review_visual_fidelity(state: AgentState) -> dict:
    if not VISUAL_FIDELITY_JUDGE_CONFIG["enabled"]:
        raise RuntimeError("Visual Fidelity Judge is disabled in config/judge_panel.yml.")
    shots = state.get("shot_plan") or []
    requests = state.get("image_prompt_requests") or []
    generated = state.get("generated_videos") or []
    validations = state.get("video_validation_results") or []
    source_images = state.get("image_files") or []
    if not shots or not (len(shots) == len(requests) == len(generated) == len(validations) == len(source_images)):
        raise RuntimeError("Visual Fidelity Judge needs one request, video, validation, and source image per approved shot.")
    ids = [shot.get("shot_id") for shot in shots]
    if any([item.get("shot_id") for item in collection] != ids for collection in (requests, generated, validations)):
        raise RuntimeError("Visual Fidelity Judge inputs must use the approved shot order.")
    if any(not item.get("valid") for item in validations):
        raise RuntimeError("Visual Fidelity Judge accepts only deterministically validated videos.")

    character_files = state.get("character_reference_files", [])
    reference_by_id = {
        character["character_id"]: character_files[index]
        for index, character in enumerate(character_definitions(state))
        if isinstance(character, dict) and character.get("character_id") and index < len(character_files)
    }
    previous = state.get("visual_fidelity_reports", {})
    targets = set(state.get("judge_retry_shots") or ids)
    evaluations = list(state.get("llm_evaluations", []))
    reports = {}
    for shot, request, generated_video, source_image in zip(shots, requests, generated, source_images):
        shot_id = shot["shot_id"]
        if shot_id not in targets and shot_id in previous:
            reports[shot_id] = previous[shot_id]
            continue
        video_file = generated_video.get("video_file")
        if not source_image or not video_file:
            raise RuntimeError(f"Visual Fidelity Judge media is missing for {shot_id}.")
        character_ids = request.get("character_reference_ids") or shot.get("characters_present") or []
        missing_ids = [character_id for character_id in character_ids if character_id not in reference_by_id]
        if missing_ids:
            raise RuntimeError(f"Visual Fidelity Judge is missing character references: {', '.join(missing_ids)}.")
        location = request.get("location_reference") or {}
        location_file = location.get("reference_file") if isinstance(location, dict) else None
        style_file = state.get("mood_board_file")
        reference_images = [source_image, *(reference_by_id[character_id] for character_id in character_ids)]
        reference_roles = ["approved_source_image", *(f"character_reference:{item}" for item in character_ids)]
        for role, optional_file in (("location_reference", location_file), ("style_reference", style_file)):
            if optional_file and Path(optional_file).is_file():
                reference_images.append(optional_file)
                reference_roles.append(role)
        payload = {
            "shot_id": shot_id, "approved_shot": shot,
            "approved_source_image_file": source_image,
            "character_reference_ids": character_ids,
            "location_reference": location or None,
            "style_reference": {"file": style_file} if style_file else None,
            "reference_image_order": reference_roles,
        }
        report, evaluation = judge_visual_fidelity(video_file, reference_images, payload)
        evaluation["call_number"] = len(evaluations) + 1
        evaluations.append(evaluation)
        reports[shot_id] = report
        logger.info("visual_fidelity_judge %s", json.dumps({
            "shot_id": shot_id, "scores": report["scores"],
            "issue_metrics": [issue["metric"] for issue in report["issues"]],
        }, sort_keys=True))
    return {"visual_fidelity_reports": reports, "llm_evaluations": evaluations}
from continuity_context import character_definitions
