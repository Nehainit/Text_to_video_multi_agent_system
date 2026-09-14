import json
import logging

from video_automation.agents.gemini_judge import invoke_gemini_judge
from video_automation.prompts import (
    ADVERSARIAL_JUDGE_CONFIG,
    ADVERSARIAL_JUDGE_SYSTEM_PROMPT,
    CONTEXT_INTENT_JUDGE_CONFIG,
    CONTEXT_INTENT_JUDGE_SYSTEM_PROMPT,
    JUDGE_PANEL_CONFIG,
    META_JUDGE_CONFIG,
    META_JUDGE_SYSTEM_PROMPT,
    MOTION_TEMPORAL_JUDGE_CONFIG,
    MOTION_TEMPORAL_JUDGE_SYSTEM_PROMPT,
)
from video_automation.schema import AgentState
from video_automation.agents.visual_fidelity_judge import (
    METRICS as VISUAL_METRICS,
    review_visual_fidelity,
    scored_report_schema,
    validate_scored_report,
)


logger = logging.getLogger(__name__)
MOTION_METRICS = (
    "subject_motion_alignment", "camera_motion_alignment", "environment_motion_alignment",
    "motion_naturalness", "motion_intensity_alignment", "physical_plausibility",
    "identity_stability_over_time", "background_stability", "temporal_consistency", "pacing",
)
CONTEXT_METRICS = (
    "shot_alignment", "narration_alignment", "story_faithfulness", "emotional_intent",
    "shot_purpose", "semantic_coherence", "action_faithfulness",
    "character_role_faithfulness", "absence_of_unsupported_events",
)
ALL_METRICS = VISUAL_METRICS + MOTION_METRICS + CONTEXT_METRICS
MOTION_SCHEMA = scored_report_schema("motion_temporal", MOTION_METRICS)
CONTEXT_SCHEMA = scored_report_schema("context_intent", CONTEXT_METRICS)
SEVERITIES = ("minor", "major", "critical")
JUDGES = ("visual_fidelity", "motion_temporal", "context_intent")


def _ordered_inputs(state: AgentState, *, source_images: bool = False):
    shots = state.get("shot_plan") or []
    plans = state.get("motion_plans") or []
    videos = state.get("generated_videos") or []
    validations = state.get("video_validation_results") or []
    collections = [plans, videos, validations]
    if source_images:
        collections.append(state.get("image_files") or [])
    if not shots or any(len(items) != len(shots) for items in collections):
        raise RuntimeError("Judge Panel needs one motion plan, video, and validation per approved shot.")
    ids = [shot.get("shot_id") for shot in shots]
    if any([item.get("shot_id") for item in collection] != ids for collection in (plans, videos, validations)):
        raise RuntimeError("Judge Panel inputs must use the approved shot order.")
    if any(not item.get("valid") for item in validations):
        raise RuntimeError("Judge Panel accepts only deterministically validated videos.")
    return shots, plans, videos


def _record(evaluations: list[dict], evaluation: dict) -> None:
    evaluation["call_number"] = len(evaluations) + 1
    evaluations.append(evaluation)


def review_motion_temporal(state: AgentState) -> dict:
    if not MOTION_TEMPORAL_JUDGE_CONFIG["enabled"]:
        raise RuntimeError("Motion and Temporal Judge is disabled in config/judge_panel.yml.")
    shots, plans, videos = _ordered_inputs(state, source_images=True)
    source_images = state["image_files"]
    previous = state.get("motion_temporal_reports", {})
    targets = set(state.get("judge_retry_shots") or [shot["shot_id"] for shot in shots])
    evaluations = list(state.get("llm_evaluations", []))
    reports = {}
    for shot, plan, video, source_image in zip(shots, plans, videos, source_images):
        shot_id = shot["shot_id"]
        if shot_id not in targets and shot_id in previous:
            reports[shot_id] = previous[shot_id]
            continue
        if not source_image or not video.get("video_file"):
            raise RuntimeError(f"Motion and Temporal Judge media is missing for {shot_id}.")
        payload = {
            "shot_id": shot_id,
            "approved_source_image_file": source_image,
            "approved_motion_plan": plan,
            "approved_video_prompt": plan.get("video_prompt"),
            "expected_duration_seconds": video.get("requested_duration_seconds") or plan.get("duration_seconds"),
            "reference_image_order": ["approved_source_image"],
        }
        report, evaluation = invoke_gemini_judge(
            agent_name="motion-temporal-judge", purpose="judge_motion_temporal",
            system_prompt=MOTION_TEMPORAL_JUDGE_SYSTEM_PROMPT, response_schema=MOTION_SCHEMA,
            payload=payload, video_file=video["video_file"], image_files=[source_image],
        )
        reports[shot_id] = validate_scored_report(report, "motion_temporal", MOTION_METRICS)
        _record(evaluations, evaluation)
        logger.info("motion_temporal_judge %s", json.dumps({
            "shot_id": shot_id, "scores": reports[shot_id]["scores"],
            "issue_metrics": [issue["metric"] for issue in reports[shot_id]["issues"]],
        }, sort_keys=True))
    return {"motion_temporal_reports": reports, "llm_evaluations": evaluations}


def review_context_intent(state: AgentState) -> dict:
    if not CONTEXT_INTENT_JUDGE_CONFIG["enabled"]:
        raise RuntimeError("Context and Intent Judge is disabled in config/judge_panel.yml.")
    shots, plans, videos = _ordered_inputs(state)
    beat_by_id = {item.get("beat_id"): item for item in state.get("story_beats", []) if isinstance(item, dict)}
    segment_by_id = {
        item.get("segment_id"): item for item in state.get("narration_segments", []) if isinstance(item, dict)
    }
    previous = state.get("context_intent_reports", {})
    targets = set(state.get("judge_retry_shots") or [shot["shot_id"] for shot in shots])
    evaluations = list(state.get("llm_evaluations", []))
    reports = {}
    for shot, plan, video in zip(shots, plans, videos):
        shot_id = shot["shot_id"]
        if shot_id not in targets and shot_id in previous:
            reports[shot_id] = previous[shot_id]
            continue
        if not video.get("video_file"):
            raise RuntimeError(f"Context and Intent Judge video is missing for {shot_id}.")
        segment_ids = shot.get("source_segment_ids") or []
        segments = [segment_by_id[item] for item in segment_ids if item in segment_by_id]
        missing_segments = [item for item in segment_ids if item not in segment_by_id]
        if missing_segments:
            raise RuntimeError(f"Context and Intent Judge is missing narration segments for {shot_id}.")
        beat_ids = list(dict.fromkeys(
            beat_id for segment in segments for beat_id in segment.get("parent_beat_ids", [])
        ))
        if any(item not in beat_by_id for item in beat_ids):
            raise RuntimeError(f"Context and Intent Judge is missing source story beats for {shot_id}.")
        payload = {
            "shot_id": shot_id,
            "approved_shot": shot,
            "source_story_beats": [beat_by_id[item] for item in beat_ids if item in beat_by_id],
            "narration_segments": segments or ([{"text": shot["narration"]}] if shot.get("narration") else []),
            "approved_motion_plan": plan,
        }
        report, evaluation = invoke_gemini_judge(
            agent_name="context-intent-judge", purpose="judge_context_intent",
            system_prompt=CONTEXT_INTENT_JUDGE_SYSTEM_PROMPT, response_schema=CONTEXT_SCHEMA,
            payload=payload, video_file=video["video_file"],
        )
        reports[shot_id] = validate_scored_report(report, "context_intent", CONTEXT_METRICS)
        _record(evaluations, evaluation)
        logger.info("context_intent_judge %s", json.dumps({
            "shot_id": shot_id, "scores": reports[shot_id]["scores"],
            "issue_metrics": [issue["metric"] for issue in reports[shot_id]["issues"]],
        }, sort_keys=True))
    return {"context_intent_reports": reports, "llm_evaluations": evaluations}


def run_primary_judge_panel(state: AgentState) -> dict:
    if not JUDGE_PANEL_CONFIG["enabled"]:
        raise RuntimeError("Gemini Judge Panel is disabled in config/judge_panel.yml.")
    working = dict(state)
    for reviewer in (review_visual_fidelity, review_motion_temporal, review_context_intent):
        working.update(reviewer(working))
    return {
        "visual_fidelity_reports": working["visual_fidelity_reports"],
        "motion_temporal_reports": working["motion_temporal_reports"],
        "context_intent_reports": working["context_intent_reports"],
        "llm_evaluations": working["llm_evaluations"],
        "pipeline_status": "primary_judges_complete",
    }


ADVERSARIAL_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["challenges", "missed_issues"],
    "properties": {
        "challenges": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["judge", "metric", "original_score", "recommended_score", "reason", "timestamp_or_range"],
            "properties": {
                "judge": {"type": "string", "enum": list(JUDGES)},
                "metric": {"type": "string", "enum": list(ALL_METRICS)},
                "original_score": {"type": "integer", "minimum": 1, "maximum": 10},
                "recommended_score": {"type": "integer", "minimum": 1, "maximum": 10},
                "reason": {"type": "string", "minLength": 1},
                "timestamp_or_range": {"type": ["string", "null"]},
            },
        }},
        "missed_issues": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["judge", "metric", "severity", "description", "timestamp_or_range"],
            "properties": {
                "judge": {"type": "string", "enum": list(JUDGES)},
                "metric": {"type": "string", "enum": list(ALL_METRICS)},
                "severity": {"type": "string", "enum": list(SEVERITIES)},
                "description": {"type": "string", "minLength": 1},
                "timestamp_or_range": {"type": ["string", "null"]},
            },
        }},
    },
}


def _judge_metrics(judge: str) -> tuple[str, ...]:
    return {"visual_fidelity": VISUAL_METRICS, "motion_temporal": MOTION_METRICS, "context_intent": CONTEXT_METRICS}[judge]


def _valid_timestamp(value: object) -> bool:
    return value is None or isinstance(value, str) and bool(value.strip())


def validate_adversarial_report(value: object, primary_reports: dict[str, dict]) -> dict:
    if not isinstance(value, dict) or set(value) != {"challenges", "missed_issues"}:
        raise RuntimeError("Adversarial Judge returned an invalid result schema.")
    if not isinstance(value["challenges"], list) or not isinstance(value["missed_issues"], list):
        raise RuntimeError("Adversarial Judge lists are invalid.")
    for challenge in value["challenges"]:
        fields = {"judge", "metric", "original_score", "recommended_score", "reason", "timestamp_or_range"}
        if not isinstance(challenge, dict) or set(challenge) != fields or challenge["judge"] not in JUDGES:
            raise RuntimeError("Adversarial Judge returned an invalid challenge.")
        judge = challenge["judge"]
        metric = challenge["metric"]
        original = challenge["original_score"]
        recommended = challenge["recommended_score"]
        if metric not in _judge_metrics(judge) or type(original) is not int or type(recommended) is not int:
            raise RuntimeError("Adversarial Judge challenge metric or scores are invalid.")
        if primary_reports[judge]["scores"][metric] != original or not 1 <= recommended < original <= 10:
            raise RuntimeError("Adversarial Judge challenge does not match and lower the original score.")
        if not isinstance(challenge["reason"], str) or not challenge["reason"].strip():
            raise RuntimeError("Adversarial Judge challenge needs an evidence-based reason.")
        if not _valid_timestamp(challenge["timestamp_or_range"]):
            raise RuntimeError("Adversarial Judge challenge timestamp is invalid.")
    for issue in value["missed_issues"]:
        fields = {"judge", "metric", "severity", "description", "timestamp_or_range"}
        if not isinstance(issue, dict) or set(issue) != fields or issue["judge"] not in JUDGES:
            raise RuntimeError("Adversarial Judge returned an invalid missed issue.")
        if issue["metric"] not in _judge_metrics(issue["judge"]) or issue["severity"] not in SEVERITIES:
            raise RuntimeError("Adversarial Judge missed issue classification is invalid.")
        if not isinstance(issue["description"], str) or not issue["description"].strip():
            raise RuntimeError("Adversarial Judge missed issue needs a description.")
        if not _valid_timestamp(issue["timestamp_or_range"]):
            raise RuntimeError("Adversarial Judge missed issue timestamp is invalid.")
    return value


def run_adversarial_judge(state: AgentState) -> dict:
    if not ADVERSARIAL_JUDGE_CONFIG["enabled"]:
        raise RuntimeError("Adversarial Judge is disabled in config/judge_panel.yml.")
    shots, plans, videos = _ordered_inputs(state)
    previous = state.get("adversarial_reports", {})
    targets = set(state.get("judge_retry_shots") or [shot["shot_id"] for shot in shots])
    evaluations = list(state.get("llm_evaluations", []))
    reports = {}
    for shot, plan, video in zip(shots, plans, videos):
        shot_id = shot["shot_id"]
        if shot_id not in targets and shot_id in previous:
            reports[shot_id] = previous[shot_id]
            continue
        primary = {
            "visual_fidelity": state["visual_fidelity_reports"][shot_id],
            "motion_temporal": state["motion_temporal_reports"][shot_id],
            "context_intent": state["context_intent_reports"][shot_id],
        }
        payload = {"shot_id": shot_id, "approved_shot": shot, "approved_motion_plan": plan, **primary}
        report, evaluation = invoke_gemini_judge(
            agent_name="adversarial-video-judge", purpose="adversarial_video_review",
            system_prompt=ADVERSARIAL_JUDGE_SYSTEM_PROMPT, response_schema=ADVERSARIAL_SCHEMA,
            payload=payload, video_file=video["video_file"],
        )
        reports[shot_id] = validate_adversarial_report(report, primary)
        _record(evaluations, evaluation)
    return {
        "adversarial_reports": reports,
        "llm_evaluations": evaluations,
        "pipeline_status": "adversarial_judge_complete",
    }


META_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": [
        "approved", "overall_score", "final_scores", "strong_metrics", "weak_metrics",
        "critical_issues", "failure_source", "retry_target", "refinement_needed", "refinement_brief",
    ],
    "properties": {
        "approved": {"type": "boolean"},
        "overall_score": {"type": "number", "minimum": 1, "maximum": 10},
        "final_scores": {
            "type": "object", "additionalProperties": False, "required": list(ALL_METRICS),
            "properties": {metric: {"type": "integer", "minimum": 1, "maximum": 10} for metric in ALL_METRICS},
        },
        "strong_metrics": {"type": "array", "items": {"type": "string", "enum": list(ALL_METRICS)}},
        "weak_metrics": {"type": "array", "items": {"type": "string", "enum": list(ALL_METRICS)}},
        "critical_issues": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "failure_source": {"type": "string", "enum": ["source_image", "motion_plan", "video_generation", "none"]},
        "retry_target": {"type": "string", "enum": ["image_generation", "motion_planner", "video_generation", "none"]},
        "refinement_needed": {"type": "boolean"},
        "refinement_brief": {"type": "array", "items": {"type": "string", "minLength": 1}},
    },
}
CRITICAL_METRICS = {
    "character_identity", "anatomy_integrity", "temporal_consistency",
    "motion_naturalness", "shot_alignment", "story_faithfulness",
}
ROUTES = {
    "source_image": "image_generation",
    "motion_plan": "motion_planner",
    "video_generation": "video_generation",
    "none": "none",
}


def validate_meta_report(value: object) -> dict:
    fields = {
        "approved", "overall_score", "final_scores", "strong_metrics", "weak_metrics",
        "critical_issues", "failure_source", "retry_target", "refinement_needed", "refinement_brief",
    }
    if not isinstance(value, dict) or set(value) != fields or type(value["approved"]) is not bool:
        raise RuntimeError("Meta Judge returned an invalid result schema.")
    scores = value["final_scores"]
    if not isinstance(scores, dict) or set(scores) != set(ALL_METRICS) or any(
        type(score) is not int or not 1 <= score <= 10 for score in scores.values()
    ):
        raise RuntimeError("Meta Judge final_scores are invalid.")
    if type(value["overall_score"]) not in (int, float) or not 1 <= value["overall_score"] <= 10:
        raise RuntimeError("Meta Judge overall_score must be from 1 through 10.")
    for key in ("strong_metrics", "weak_metrics"):
        if not isinstance(value[key], list) or len(value[key]) != len(set(value[key])) or any(item not in ALL_METRICS for item in value[key]):
            raise RuntimeError(f"Meta Judge {key} is invalid.")
    if any(scores[item] < 8 for item in value["strong_metrics"]) or any(scores[item] >= 8 for item in value["weak_metrics"]):
        raise RuntimeError("Meta Judge strong and weak metrics disagree with final_scores.")
    if set(value["weak_metrics"]) != {metric for metric, score in scores.items() if score < 8}:
        raise RuntimeError("Meta Judge weak_metrics must identify every final score below 8.")
    for key in ("critical_issues", "refinement_brief"):
        if not isinstance(value[key], list) or any(not isinstance(item, str) or not item.strip() for item in value[key]):
            raise RuntimeError(f"Meta Judge {key} is invalid.")
    source, target = value["failure_source"], value["retry_target"]
    if source not in ROUTES or ROUTES[source] != target or type(value["refinement_needed"]) is not bool:
        raise RuntimeError("Meta Judge failure source and retry target disagree.")
    if value["approved"]:
        if any(scores[metric] < 8 for metric in CRITICAL_METRICS) or value["critical_issues"]:
            raise RuntimeError("Meta Judge approved a critical failure.")
        if source != "none" or value["refinement_needed"] or value["refinement_brief"]:
            raise RuntimeError("Meta Judge approved result contains retry instructions.")
    elif source == "none" or not value["refinement_needed"] or not value["refinement_brief"]:
        raise RuntimeError("Meta Judge rejection needs a failure source and refinement brief.")
    elif not value["weak_metrics"] and not value["critical_issues"]:
        raise RuntimeError("Meta Judge rejection needs a weak metric or critical issue.")
    return value


def _image_retry_qa(state: AgentState, retry_shots: list[str], reports: dict[str, dict]) -> list[dict]:
    previous = {item["shot_id"]: item for item in state.get("shot_image_qa_results", [])}
    for shot_id in retry_shots:
        previous[shot_id] = {
            "shot_id": shot_id, "approved": False, "animation_ready": False,
            "issues": [{
                "type": "generation_artifact", "severity": "major",
                "description": "The Judge Panel traced the failure to the approved source image.",
                "correction": " ".join(reports[shot_id]["refinement_brief"]),
            }],
            "retry_target": "image_generation",
        }
    return [previous[shot["shot_id"]] for shot in state["shot_plan"] if shot["shot_id"] in previous]


def run_meta_judge(state: AgentState) -> dict:
    if not META_JUDGE_CONFIG["enabled"]:
        raise RuntimeError("Meta Judge is disabled in config/judge_panel.yml.")
    shot_ids = [shot["shot_id"] for shot in state.get("shot_plan", [])]
    previous = state.get("meta_judge_reports", {})
    targets = set(state.get("judge_retry_shots") or shot_ids)
    evaluations = list(state.get("llm_evaluations", []))
    reports = {}
    for shot_id in shot_ids:
        if shot_id not in targets and shot_id in previous:
            reports[shot_id] = previous[shot_id]
            continue
        payload = {
            "visual_fidelity": state["visual_fidelity_reports"][shot_id],
            "motion_temporal": state["motion_temporal_reports"][shot_id],
            "context_intent": state["context_intent_reports"][shot_id],
            "adversarial": state["adversarial_reports"][shot_id],
        }
        report, evaluation = invoke_gemini_judge(
            agent_name="video-meta-judge", purpose="decide_video_quality",
            system_prompt=META_JUDGE_SYSTEM_PROMPT, response_schema=META_SCHEMA, payload=payload,
        )
        reports[shot_id] = validate_meta_report(report)
        input_critical = any(
            issue.get("severity") == "critical"
            for name in ("visual_fidelity", "motion_temporal", "context_intent")
            for issue in payload[name]["issues"]
        ) or any(issue.get("severity") == "critical" for issue in payload["adversarial"]["missed_issues"])
        if reports[shot_id]["approved"] and input_critical:
            raise RuntimeError("Meta Judge approved a critical issue reported by the panel.")
        _record(evaluations, evaluation)

    retry_counts = dict(state.get("judge_retry_counts", {}))
    rejected = [shot_id for shot_id in shot_ids if not reports[shot_id]["approved"]]
    retryable = [
        shot_id for shot_id in rejected
        if retry_counts.get(shot_id, 0) < int(JUDGE_PANEL_CONFIG["max_retries"])
    ]
    exhausted = [shot_id for shot_id in rejected if shot_id not in retryable]
    priority = ("image_generation", "motion_planner", "video_generation")
    retry_target = next((target for target in priority if any(
        reports[shot_id]["retry_target"] == target for shot_id in retryable
    )), None)
    retry_shots = [shot_id for shot_id in retryable if reports[shot_id]["retry_target"] == retry_target]
    for shot_id in retry_shots:
        retry_counts[shot_id] = retry_counts.get(shot_id, 0) + 1
    decision_target = retry_target or next((
        target for target in priority if any(reports[shot_id]["retry_target"] == target for shot_id in rejected)
    ), None)
    result = {
        "meta_judge_reports": reports,
        "llm_evaluations": evaluations,
        "judge_retry_counts": retry_counts,
        "judge_retry_shots": retry_shots,
        "judge_retry_target": retry_target,
        "judge_exhausted_shots": exhausted,
        "video_retry_shots": retry_shots,
        "motion_plan_retry_shots": retry_shots if retry_target == "motion_planner" else [],
        "motion_plan_retry_feedback": {
            shot_id: " ".join(reports[shot_id]["refinement_brief"]) for shot_id in retry_shots
        } if retry_target == "motion_planner" else {},
        "shot_image_qa_retry_shots": retry_shots if retry_target == "image_generation" else [],
        "judge_decision": "ACCEPT" if not rejected else "REFINE" if decision_target in {"image_generation", "motion_planner"} else "REGENERATE",
        "pipeline_status": "judge_retry_exhausted" if exhausted else "judge_retry" if retry_shots else "judge_accepted",
    }
    if retry_target == "image_generation":
        result["shot_image_qa_results"] = _image_retry_qa(state, retry_shots, reports)
    logger.info("video_meta_judge %s", json.dumps({
        "decision": result["judge_decision"], "retry_target": retry_target,
        "retry_shots": retry_shots, "exhausted_shots": exhausted,
    }, sort_keys=True))
    return result
