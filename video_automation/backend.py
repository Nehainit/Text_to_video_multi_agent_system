import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from langgraph.types import Command
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
sys.path.insert(0, str(ROOT))

from video_automation.agent_graph import build_story_graph, persistent_checkpointer
from video_automation.models import validate_model_configuration
from video_automation.agents.story_agent import UnsafeContentError
from video_automation.agents.video_validator import validate_generated_videos
from video_automation.artifact_store import archive_artifacts


class VideoRequest(BaseModel):
    topic: str
    duration: int = Field(default=30, ge=5, le=120)
    language: str = "English"
    quality_mode: Literal["standard", "refine"] = "standard"
    aspect_ratio: Literal["16:9", "9:16", "1:1"] = "9:16"
    video_quality: Literal["standard", "high"] = "standard"


class StoryReviewRequest(BaseModel):
    thread_id: str
    action: Literal["approve", "edit", "regenerate"]
    story: str | None = None
    note: str | None = None


class ReferenceReviewRequest(BaseModel):
    thread_id: str
    action: Literal["approve", "regenerate"]
    note: str | None = None


class ScenePlanReviewRequest(BaseModel):
    thread_id: str
    action: Literal["retry", "revise_narration"]
    note: str | None = None


class VisualPlanReviewRequest(BaseModel):
    thread_id: str
    action: Literal["retry", "revise_scenes"]
    note: str | None = None


class ShotPlanReviewRequest(BaseModel):
    thread_id: str
    action: Literal["retry", "revise_visuals"]
    note: str | None = None


class ShotFeedback(BaseModel):
    shot_id: str
    note: str


class DirectorReviewRequest(BaseModel):
    thread_id: str
    action: Literal["approve", "regenerate"]
    feedback: list[ShotFeedback] = Field(default_factory=list)


class VisualStoryboardReviewRequest(DirectorReviewRequest):
    pass


app = FastAPI()
logger = logging.getLogger(__name__)
graph = build_story_graph(
    checkpointer=persistent_checkpointer(ROOT / ".langgraph-checkpoints.sqlite"),
    video_validator=validate_generated_videos,
)
MAX_AUTO_RETRIES_PER_STAGE = 2
MAX_AUTO_ADVANCE_STEPS = 12


def _output_url(file_path: str | None) -> str | None:
    if not file_path:
        return None
    try:
        return f"/output/{Path(file_path).resolve().relative_to(ROOT).as_posix()}"
    except ValueError:
        return None


def _archive(result: dict, thread_id: str, stage: str) -> dict | None:
    if not result.get("output_dir"):
        return None
    try:
        return archive_artifacts(result, thread_id, stage)
    except Exception as exc:
        logger.exception("MinIO artifact upload failed; local artifact history was retained")
        return {"backend": "local", "error": str(exc)}


def _artifacts(result: dict) -> dict:
    character_board_file = result.get("character_board_file")
    mood_board_file = result.get("mood_board_file")
    image_files = result.get("image_files", [])
    video_file = result.get("video_file")
    video_candidates = [
        {
            **candidate,
            "file_url": _output_url(candidate.get("file")),
        }
        for candidate in result.get("video_candidates", [])
    ]
    return {
        "story": result.get("story"),
        "story_outline": result.get("story_outline"),
        "parsed_requirements": result.get("parsed_requirements"),
        "llm_evaluations": result.get("llm_evaluations", []),
        "planning_attempts": result.get("planning_attempts", {}),
        "narration_script": result.get("narration_script"),
        "narration_segments": result.get("narration_segments", []),
        "estimated_narration_seconds": result.get("estimated_narration_seconds"),
        "actual_narration_seconds": result.get("actual_narration_seconds"),
        "narration_file": result.get("narration_file"),
        "narration_url": _output_url(result.get("narration_file")),
        "narration_alignment_file": result.get("narration_alignment_file"),
        "narration_alignment_url": _output_url(result.get("narration_alignment_file")),
        "narration_segment_timings": result.get("narration_segment_timings", []),
        "scene_timings": result.get("scene_timings", []),
        "needs_story_revision": result.get("needs_story_revision", False),
        "story_revision_reason": result.get("story_revision_reason"),
        "scene_analysis": result.get("scene_analysis", []),
        "scenes": result.get("scenes", []),
        "scene_plan_needs_revision": result.get("scene_plan_needs_revision", False),
        "scene_plan_revision_reason": result.get("scene_plan_revision_reason"),
        "scene_plan_issues": result.get("scene_plan_issues", []),
        "visual_beats": result.get("visual_beats", []),
        "visual_plan_needs_revision": result.get("visual_plan_needs_revision", False),
        "visual_plan_revision_reason": result.get("visual_plan_revision_reason"),
        "visual_plan_issues": result.get("visual_plan_issues", []),
        "shot_plan": result.get("shot_plan", []),
        "shot_plan_needs_revision": result.get("shot_plan_needs_revision", False),
        "shot_plan_revision_reason": result.get("shot_plan_revision_reason"),
        "shot_plan_issues": result.get("shot_plan_issues", []),
        "image_prompt_requests": result.get("image_prompt_requests", []),
        "generated_images": result.get("generated_images", []),
        "shot_image_qa_results": result.get("shot_image_qa_results", []),
        "motion_plans": result.get("motion_plans", []),
        "motion_plan_needs_revision": result.get("motion_plan_needs_revision", False),
        "motion_plan_revision_reason": result.get("motion_plan_revision_reason"),
        "generated_videos": result.get("generated_videos", []),
        "critic_issues": result.get("critic_issues", []),
        "story_beats": result.get("story_beats", []),
        "genre": result.get("genre", "general"),
        "genre_structure": result.get("genre_structure", []),
        "character_board_file": character_board_file,
        "character_board_url": _output_url(character_board_file),
        "mood_board_file": mood_board_file,
        "mood_board_url": _output_url(mood_board_file),
        "director_plan": result.get("director_plan", result.get("storyboard", [])),
        "storyboard": result.get("storyboard", []),
        "sound_design_plan": result.get("sound_design_plan", []),
        "music_file": result.get("music_file"),
        "music_url": _output_url(result.get("music_file")),
        "image_files": image_files,
        "image_urls": [_output_url(path) for path in image_files],
        "quality_mode": result.get("quality_mode", "standard"),
        "aspect_ratio": result.get("aspect_ratio", "9:16"),
        "video_quality": result.get("video_quality", "standard"),
        "video_candidates": video_candidates,
        "video_validation_results": result.get("video_validation_results", []),
        "video_validation_retry_counts": result.get("video_validation_retry_counts", {}),
        "video_validation_exhausted_shots": result.get("video_validation_exhausted_shots", []),
        "edit_timeline": result.get("edit_timeline", []),
        "narration_duration_seconds": result.get("narration_duration_seconds"),
        "timeline_duration_seconds": result.get("timeline_duration_seconds"),
        "timeline_valid": result.get("timeline_valid"),
        "timeline_issues": result.get("timeline_issues", []),
        "timeline_retry_requests": result.get("timeline_retry_requests", []),
        "edit_timeline_retry_counts": result.get("edit_timeline_retry_counts", {}),
        "edit_timeline_exhausted_shots": result.get("edit_timeline_exhausted_shots", []),
        "compilation_status": result.get("compilation_status"),
        "rough_cut_file": result.get("rough_cut_file"),
        "rough_cut_url": _output_url(result.get("rough_cut_file")),
        "shot_timestamps": result.get("shot_timestamps", []),
        "sync_valid": result.get("sync_valid"),
        "compilation_issues": result.get("compilation_issues", []),
        "compilation_error": result.get("compilation_error"),
        "rough_cut_failure_source": result.get("rough_cut_failure_source"),
        "rough_cut_retry_count": result.get("rough_cut_retry_count", 0),
        "rough_cut_retry_exhausted": result.get("rough_cut_retry_exhausted", False),
        "combined_video_judge_report": result.get("combined_video_judge_report"),
        "combined_video_judge_approved": result.get("combined_video_judge_approved"),
        "combined_video_judge_status": result.get("combined_video_judge_status"),
        "combined_video_judge_error": result.get("combined_video_judge_error"),
        "combined_video_judge_error_count": result.get("combined_video_judge_error_count", 0),
        "combined_video_judge_error_exhausted": result.get("combined_video_judge_error_exhausted", False),
        "combined_video_judge_model": result.get("combined_video_judge_model"),
        "combined_video_judge_retry_target": result.get("combined_video_judge_retry_target"),
        "combined_video_judge_retry_shots": result.get("combined_video_judge_retry_shots", []),
        "video_judge_refinement_round": result.get("video_judge_refinement_round", 0),
        "video_judge_refinement_exhausted": result.get("video_judge_refinement_exhausted", False),
        "visual_fidelity_reports": result.get("visual_fidelity_reports", {}),
        "motion_temporal_reports": result.get("motion_temporal_reports", {}),
        "context_intent_reports": result.get("context_intent_reports", {}),
        "adversarial_reports": result.get("adversarial_reports", {}),
        "meta_judge_reports": result.get("meta_judge_reports", {}),
        "judge_retry_counts": result.get("judge_retry_counts", {}),
        "judge_exhausted_shots": result.get("judge_exhausted_shots", []),
        "judge_decision": result.get("judge_decision"),
        "pipeline_status": result.get("pipeline_status"),
        "story_feedback_history": result.get("story_feedback_history", []),
        "reference_feedback_history": result.get("reference_feedback_history", []),
        "director_feedback_history": result.get("director_feedback_history", []),
        "visual_feedback_history": result.get("visual_feedback_history", []),
        "count_adjustment": result.get("count_adjustment", ""),
        "warnings": result.get("warnings", []),
        "video_file": video_file,
        "video_url": _output_url(video_file),
    }


def _graph_response(result: dict, thread_id: str) -> dict:
    artifacts = _artifacts(result)
    if "__interrupt__" in result:
        review = result["__interrupt__"][0].value
        logger.warning("GRAPH PAUSED: thread_id=%s stage=%s", thread_id, review.get("stage", "unknown"))
        if review.get("stage") == "scene_plan_review":
            return {
                "status": "scene_plan_review",
                "thread_id": thread_id,
                "revision_reason": review.get("revision_reason"),
                "issues": review.get("issues", []),
                "actions": review["actions"],
                "artifacts": artifacts,
            }
        if review.get("stage") == "visual_plan_review":
            return {
                "status": "visual_plan_review",
                "thread_id": thread_id,
                "revision_reason": review.get("revision_reason"),
                "issues": review.get("issues", []),
                "actions": review["actions"],
                "artifacts": artifacts,
            }
        if review.get("stage") == "shot_plan_review":
            return {
                "status": "shot_plan_review",
                "thread_id": thread_id,
                "revision_reason": review.get("revision_reason"),
                "issues": review.get("issues", []),
                "actions": review["actions"],
                "artifacts": artifacts,
            }
        if "image_files" in review:
            artifacts.update(
                {
                    "storyboard": review["storyboard"],
                    "director_plan": review["storyboard"],
                    "image_files": review["image_files"],
                    "image_urls": [_output_url(path) for path in review["image_files"]],
                }
            )
            artifacts["storage"] = _archive({**result, **artifacts}, thread_id, "visual-storyboard-review")
            return {
                "status": "visual_storyboard_review",
                "thread_id": thread_id,
                "storyboard": artifacts["storyboard"],
                "image_urls": artifacts["image_urls"],
                "shot_image_qa_results": artifacts["shot_image_qa_results"],
                "actions": review["actions"],
                "feedback": review.get("feedback", []),
                "artifacts": artifacts,
            }
        if "character_board_file" in review:
            artifacts.update(
                {
                    "production_bible": review["production_bible"],
                    "character_board_file": review["character_board_file"],
                    "character_board_url": _output_url(review["character_board_file"]),
                    "mood_board_file": review["mood_board_file"],
                    "mood_board_url": _output_url(review["mood_board_file"]),
                }
            )
            artifacts["storage"] = _archive({**result, **artifacts}, thread_id, "reference-board-review")
            return {
                "status": "reference_board_review",
                "thread_id": thread_id,
                "production_bible": artifacts["production_bible"],
                "character_board_url": artifacts["character_board_url"],
                "mood_board_url": artifacts["mood_board_url"],
                "actions": review["actions"],
                "feedback": review.get("feedback", ""),
                "artifacts": artifacts,
            }
        if "director_plan" in review:
            artifacts["director_plan"] = review["director_plan"]
            artifacts["storyboard"] = review["director_plan"]
            artifacts["count_adjustment"] = review.get("count_adjustment", "")
            artifacts["storage"] = _archive({**result, **artifacts}, thread_id, "director-plan-review")
            return {
                "status": "director_plan_review",
                "thread_id": thread_id,
                "director_plan": artifacts["director_plan"],
                "count_adjustment": artifacts["count_adjustment"],
                "actions": review["actions"],
                "feedback": review.get("feedback", []),
                "artifacts": artifacts,
            }

        artifacts["story"] = review["story"]
        artifacts["storage"] = _archive({**result, **artifacts}, thread_id, "story-review")
        return {
            "status": "story_review",
            "thread_id": thread_id,
            "story": review["story"],
            "actions": review["actions"],
            "feedback": review.get("feedback", ""),
            "artifacts": artifacts,
        }

    artifacts["storage"] = _archive(result, thread_id, result.get("pipeline_status", "complete"))
    if result.get("pipeline_status") and not artifacts["video_file"]:
        logger.warning("GRAPH STOPPED INCOMPLETE: thread_id=%s stage=%s", thread_id, result.get("pipeline_status"))
        return {
            "status": "pipeline_incomplete",
            "thread_id": thread_id,
            "message": "Pipeline has not produced a final video yet.",
            "artifacts": artifacts,
        }
    return {
        "status": result.get("pipeline_status", "complete"),
        "thread_id": thread_id,
        "story": artifacts["story"],
        "storyboard": artifacts["storyboard"],
        "video_file": artifacts["video_file"],
        "video_url": artifacts["video_url"],
        "warnings": artifacts["warnings"],
        "artifacts": artifacts,
    }


def _resume_review(thread_id: str, node: str, command: dict) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = graph.get_state(config)
        if node in snapshot.next:
            value = Command(resume=command)
        elif (
            command.get("action") == "approve"
            and snapshot.next
            and not any(next_node.startswith("review_") for next_node in snapshot.next)
            and snapshot.values.get({"review_story": "approved", "review_visual_storyboard": "visual_approved"}.get(node, ""))
        ):
            value = None
        else:
            raise HTTPException(status_code=404, detail="Review session not found or already completed.")
        return _graph_response(graph.invoke(value, config), thread_id)
    except HTTPException:
        raise
    except UnsafeContentError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc), "categories": exc.categories}) from exc
    except Exception as exc:
        logger.exception("Artifact review failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


def _selected_feedback(thread_id: str, node: str, feedback: list[ShotFeedback]) -> list[dict[str, str]]:
    cleaned = [{"shot_id": item.shot_id.strip(), "note": item.note.strip()} for item in feedback if item.note.strip()]
    if not cleaned:
        raise HTTPException(status_code=400, detail="Add feedback to at least one shot.")
    shot_ids = [item["shot_id"] for item in cleaned]
    if len(shot_ids) != len(set(shot_ids)):
        raise HTTPException(status_code=400, detail="Each shot can have only one feedback note.")
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    if node not in snapshot.next:
        raise HTTPException(status_code=404, detail="Review session not found or already completed.")
    known_shots = {shot["shot_id"] for shot in snapshot.values.get("storyboard", snapshot.values.get("director_plan", []))}
    if not set(shot_ids) <= known_shots:
        raise HTTPException(status_code=400, detail="Feedback contains an unknown shot ID.")
    return cleaned


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.post("/api/create-video")
def create_video(request: VideoRequest) -> dict:
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="Topic is required.")
    try:
        validate_model_configuration("shot-image-qa")
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        thread_id = str(uuid.uuid4())
        result = graph.invoke(
            {
                "thread_id": thread_id,
                "topic": request.topic.strip(),
                "tone": "Infer the tone and visual style from the user's prompt.",
                "duration": f"{request.duration} seconds",
                "language": request.language.strip() or "English",
                "characters": [],
                "output_dir": str(ROOT / "outputs" / thread_id),
                "quality_mode": request.quality_mode,
                "aspect_ratio": request.aspect_ratio,
                "video_quality": request.video_quality,
                "warnings": [],
            },
            {"configurable": {"thread_id": thread_id}},
        )
        # Auto-advance ordinary review checkpoints during unattended generation.
        actions = {
            "story_review": "approve",
            "scene_plan_review": "retry",
            "visual_plan_review": "retry",
            "reference_board_review": "approve",
            "director_plan_review": "approve",
            "visual_storyboard_review": "approve",
            "shot_plan_review": "retry",
        }
        recovery_actions = {
            "scene_plan_review": "revise_narration",
            "visual_plan_review": "revise_scenes",
            "shot_plan_review": "revise_visuals",
        }
        retry_counts: dict[str, int] = {}
        recovered_stages: set[str] = set()
        for _ in range(MAX_AUTO_ADVANCE_STEPS):
            if "__interrupt__" not in result:
                break
            review = result["__interrupt__"][0].value
            if not isinstance(review, dict):
                review = {}
            stage = review.get("stage", "")
            action = actions.get(stage)
            if not action:
                logger.warning("AUTO-ADVANCING UNKNOWN CHECKPOINT: thread_id=%s", thread_id)
                action = "approve"
            if action == "retry":
                retry_count = retry_counts.get(stage, 0)
                if retry_count >= MAX_AUTO_RETRIES_PER_STAGE:
                    action = recovery_actions.get(stage)
                    if not action or stage in recovered_stages:
                        logger.warning("AUTO-RECOVERY STOPPED: thread_id=%s stage=%s retries=%s", thread_id, stage, retry_count)
                        break
                    recovered_stages.add(stage)
                    logger.warning("AUTO-RECOVERY: thread_id=%s stage=%s action=%s", thread_id, stage, action)
                else:
                    retry_counts[stage] = retry_count + 1
            result = graph.invoke(
                Command(resume={"action": action, "note": review.get("revision_reason", "")}),
                {"configurable": {"thread_id": thread_id}},
            )
        else:
            logger.warning("AUTO-ADVANCE STOPPED: thread_id=%s steps=%s", thread_id, MAX_AUTO_ADVANCE_STEPS)
        return _graph_response(result, thread_id)
    except HTTPException:
        raise
    except UnsafeContentError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc), "categories": exc.categories}) from exc
    except Exception as exc:
        logger.exception("Video creation failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


@app.post("/api/review-story")
def submit_story_review(request: StoryReviewRequest) -> dict:
    thread_id = request.thread_id.strip()
    command = {"action": request.action}
    if request.action == "edit":
        if not request.story or not request.story.strip():
            raise HTTPException(status_code=400, detail="Edited story is required.")
        command["story"] = request.story
    elif request.action == "regenerate":
        note = (request.note or "").strip()
        if not note:
            raise HTTPException(status_code=400, detail="Story feedback is required.")
        command["note"] = note
    return _resume_review(thread_id, "review_story", command)


@app.post("/api/review-reference-board")
def submit_reference_review(request: ReferenceReviewRequest) -> dict:
    command = {"action": request.action}
    if request.action == "regenerate":
        note = (request.note or "").strip()
        if not note:
            raise HTTPException(status_code=400, detail="Reference package feedback is required.")
        command["note"] = note
    return _resume_review(request.thread_id.strip(), "review_reference_package", command)


@app.post("/api/review-scene-plan")
def submit_scene_plan_review(request: ScenePlanReviewRequest) -> dict:
    command = {"action": request.action, "note": (request.note or "").strip()}
    return _resume_review(request.thread_id.strip(), "review_scene_plan", command)


@app.post("/api/review-visual-plan")
def submit_visual_plan_review(request: VisualPlanReviewRequest) -> dict:
    command = {"action": request.action, "note": (request.note or "").strip()}
    return _resume_review(request.thread_id.strip(), "review_visual_plan", command)


@app.post("/api/review-shot-plan")
def submit_shot_plan_review(request: ShotPlanReviewRequest) -> dict:
    command = {"action": request.action, "note": (request.note or "").strip()}
    return _resume_review(request.thread_id.strip(), "review_shot_plan", command)


@app.post("/api/review-director-plan")
def submit_director_review(request: DirectorReviewRequest) -> dict:
    command = {"action": request.action}
    if request.action == "regenerate":
        command["feedback"] = _selected_feedback(request.thread_id.strip(), "review_director_plan", request.feedback)
    return _resume_review(request.thread_id.strip(), "review_director_plan", command)


@app.post("/api/review-visual-storyboard")
def submit_visual_storyboard_review(request: VisualStoryboardReviewRequest) -> dict:
    command = {"action": request.action}
    if request.action == "regenerate":
        command["feedback"] = _selected_feedback(request.thread_id.strip(), "review_visual_storyboard", request.feedback)
    return _resume_review(request.thread_id.strip(), "review_visual_storyboard", command)


@app.get("/output/{path:path}")
def output_file(path: str) -> FileResponse:
    file_path = (ROOT / path).resolve()
    try:
        file_path.relative_to((ROOT / "outputs").resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="File not found.") from exc
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
