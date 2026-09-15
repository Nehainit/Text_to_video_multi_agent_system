import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from video_automation.agents.image_agent import _image_worker_count
from video_automation.continuity_context import character_definitions
from video_automation.models import invoke_with_images_and_evaluation
from video_automation.prompts import SHOT_IMAGE_QA_CONFIG, SHOT_IMAGE_QA_SYSTEM_PROMPT
from video_automation.schema import AgentState
from video_automation.agents.story_agent import _parse_json


ISSUE_TYPES = {
    "missing_character", "extra_character", "character_identity_drift", "character_scale_mismatch",
    "character_design_mismatch", "action_mismatch", "visual_focus_mismatch", "framing_mismatch",
    "camera_angle_mismatch", "composition_mismatch", "location_mismatch", "emotion_mismatch",
    "style_mismatch", "unsupported_object", "generation_artifact", "text_or_watermark", "other",
}


def _review_job(job: tuple[str, str, list[str]]) -> tuple[str, dict, dict]:
    shot_id, prompt, image_files = job
    response, evaluation = invoke_with_images_and_evaluation(
        "shot-image-qa", prompt, image_files, purpose="review_shot_image",
    )
    return shot_id, _validate_result(_parse_json(response.content), shot_id), evaluation


def _validate_result(value: object, shot_id: str) -> dict:
    fields = {"approved", "animation_ready", "issues"}
    if not isinstance(value, dict) or not fields <= set(value):
        raise RuntimeError("Shot Image QA returned an invalid result schema.")
    issues = value["issues"]
    if type(value["approved"]) is not bool or type(value["animation_ready"]) is not bool or not isinstance(issues, list):
        raise RuntimeError("Shot Image QA returned invalid decision fields.")
    issue_fields = {"type", "severity", "description", "correction"}
    if any(
        not isinstance(issue, dict) or set(issue) != issue_fields or issue["type"] not in ISSUE_TYPES
        or any(not isinstance(issue[key], str) or not issue[key].strip() for key in issue_fields)
        for issue in issues
    ):
        raise RuntimeError("Shot Image QA returned an invalid issue.")
    passed = value["approved"] and value["animation_ready"] and not issues
    failed = not value["approved"] and not value["animation_ready"] and bool(issues)
    if not (passed or failed):
        raise RuntimeError("Shot Image QA decision fields disagree.")
    return {
        "shot_id": shot_id,
        "approved": value["approved"],
        "animation_ready": value["animation_ready"],
        "issues": issues,
        "retry_target": None if passed else "image_generation",
    }


def review_shot_images(state: AgentState) -> dict:
    if not SHOT_IMAGE_QA_CONFIG["enabled"]:
        raise RuntimeError("Shot Image QA Agent is disabled in config/shot_image_qa.yml.")
    shots = state.get("shot_plan") or []
    requests = state.get("image_prompt_requests") or []
    generated = state.get("generated_images") or []
    if not shots or len(shots) != len(requests) or len(shots) != len(generated):
        raise RuntimeError("Shot Image QA needs one prompt request and generated-image result per approved shot.")

    previous = {item["shot_id"]: item for item in state.get("shot_image_qa_results", [])}
    targets = set(state.get("shot_image_qa_pending_shots") or [shot["shot_id"] for shot in shots])
    character_files = state.get("character_reference_files", [])
    reference_by_id = {
        character["character_id"]: character_files[index]
        for index, character in enumerate(character_definitions(state))
        if isinstance(character, dict) and character.get("character_id") and index < len(character_files)
    }
    evaluations = list(state.get("llm_evaluations", []))
    results_by_shot = {}
    jobs = []

    for shot, prompt_request, image_result in zip(shots, requests, generated):
        shot_id = shot["shot_id"]
        if shot_id not in targets and shot_id in previous:
            results_by_shot[shot_id] = previous[shot_id]
            continue
        image_file = image_result.get("image_path")
        if image_result.get("generation_status") != "success" or not image_file or not Path(image_file).is_file():
            results_by_shot[shot_id] = {
                "shot_id": shot_id, "approved": False, "animation_ready": False,
                "issues": [{
                    "type": "generation_artifact", "severity": "major",
                    "description": image_result.get("error") or "The generated image file is missing.",
                    "correction": "Regenerate the same approved shot image.",
                }],
                "retry_target": "image_generation",
            }
            continue

        character_references = [
            {"character_id": character_id, "file": reference_by_id[character_id]}
            for character_id in prompt_request.get("character_reference_ids", [])
            if character_id in reference_by_id
        ]
        location = prompt_request.get("location_reference") or {}
        location_file = location.get("reference_file") if isinstance(location, dict) else None
        style_file = state.get("mood_board_file")
        image_files = [image_file, *(item["file"] for item in character_references)]
        if location_file and Path(location_file).is_file():
            image_files.append(location_file)
        if style_file and Path(style_file).is_file():
            image_files.append(style_file)
        payload = {
            "approved_shot": shot,
            "approved_image_prompt_request": prompt_request,
            "generated_image": {"file": image_file, "image_index": 1},
            "character_reference_images": [
                {**item, "image_index": index + 2} for index, item in enumerate(character_references)
            ],
            "location_reference": location,
            "style_reference": {"file": style_file} if style_file else None,
            "image_order": image_files,
        }
        jobs.append((
            shot_id,
            f"{SHOT_IMAGE_QA_SYSTEM_PROMPT}\n\nInputs:\n{json.dumps(payload, ensure_ascii=False)}",
            image_files,
        ))

    if jobs:
        with ThreadPoolExecutor(
            max_workers=_image_worker_count(len(jobs)), thread_name_prefix="shot-image-qa",
        ) as pool:
            for shot_id, result, evaluation in pool.map(_review_job, jobs):
                evaluation["call_number"] = len(evaluations) + 1
                evaluations.append(evaluation)
                results_by_shot[shot_id] = result

    results = [results_by_shot[shot["shot_id"]] for shot in shots]

    return {
        "shot_image_qa_results": results,
        "shot_image_qa_retry_shots": [item["shot_id"] for item in results if not item["approved"]],
        "shot_image_qa_round": int(state.get("shot_image_qa_round", 0)) + 1,
        "shot_image_qa_pending_shots": [],
        "llm_evaluations": evaluations,
    }
