import sys
import uuid
import logging
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from langgraph.types import Command
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
sys.path.insert(0, str(ROOT))

from agent_graph import build_story_graph
from artifact_store import archive_artifacts


class VideoRequest(BaseModel):
    topic: str
    tone: str = "cinematic"
    duration: str = "30 seconds"
    language: str = "English"
    characters: str = "Raja, Rani"


class StoryReviewRequest(BaseModel):
    thread_id: str
    action: Literal["approve", "edit", "regenerate"]
    story: str | None = None
    note: str | None = None


class StoryboardReviewRequest(BaseModel):
    thread_id: str
    action: Literal["approve", "regenerate"]
    note: str | None = None


class VisualSceneFeedback(BaseModel):
    scene_number: int
    note: str


class VisualStoryboardReviewRequest(BaseModel):
    thread_id: str
    action: Literal["approve", "regenerate"]
    feedback: list[VisualSceneFeedback] = Field(default_factory=list)


app = FastAPI()
logger = logging.getLogger(__name__)
# ponytail: in-memory review sessions; use a shared checkpointer for multi-worker deployment.
graph = build_story_graph()


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


def _graph_response(result: dict, thread_id: str) -> dict:
    video_file = result.get("video_file")
    video_url = _output_url(video_file)
    artifacts = {
        "story": result.get("story"),
        "storyboard": result.get("storyboard", []),
        "mood_board_file": result.get("mood_board_file"),
        "mood_board_url": _output_url(result.get("mood_board_file")),
        "image_files": result.get("image_files", []),
        "image_urls": [_output_url(path) for path in result.get("image_files", [])],
        "visual_feedback_history": result.get("visual_feedback_history", []),
        "video_file": video_file,
        "video_url": video_url,
    }

    if "__interrupt__" in result:
        review = result["__interrupt__"][0].value
        if "image_files" in review:
            artifacts.update(
                {
                    "storyboard": review["storyboard"],
                    "mood_board_file": review["mood_board_file"],
                    "mood_board_url": _output_url(review["mood_board_file"]),
                    "image_files": review["image_files"],
                    "image_urls": [_output_url(path) for path in review["image_files"]],
                }
            )
            artifacts["storage"] = _archive({**result, **artifacts}, thread_id, "visual-storyboard-review")
            return {
                "status": "visual_storyboard_review",
                "thread_id": thread_id,
                "storyboard": artifacts["storyboard"],
                "mood_board_url": artifacts["mood_board_url"],
                "image_urls": artifacts["image_urls"],
                "actions": review["actions"],
                "feedback": review.get("feedback", []),
                "artifacts": artifacts,
            }
        if "storyboard" in review:
            artifacts["storyboard"] = review["storyboard"]
            artifacts["storage"] = _archive({**result, **artifacts}, thread_id, "storyboard-review")
            return {
                "status": "storyboard_review",
                "thread_id": thread_id,
                "story": artifacts["story"],
                "storyboard": artifacts["storyboard"],
                "actions": review["actions"],
                "feedback": review.get("feedback", ""),
                "artifacts": artifacts,
            }

        artifacts["story"] = review["story"]
        artifacts["storage"] = _archive({**result, **artifacts}, thread_id, "story-review")
        return {
            "status": "review",
            "thread_id": thread_id,
            "story": review["story"],
            "actions": review["actions"],
            "artifacts": artifacts,
        }

    artifacts["storage"] = _archive(result, thread_id, "complete")
    return {
        "status": "complete",
        "thread_id": thread_id,
        "story": artifacts["story"],
        "storyboard": artifacts["storyboard"],
        "video_file": video_file,
        "video_url": video_url,
        "artifacts": artifacts,
    }


def _resume_review(thread_id: str, node: str, command: dict) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    try:
        if node not in graph.get_state(config).next:
            raise HTTPException(status_code=404, detail="Review session not found or already completed.")
        return _graph_response(graph.invoke(Command(resume=command), config), thread_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Artifact review failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.post("/api/create-video")
def create_video(request: VideoRequest) -> dict:
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="Topic is required.")

    try:
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        result = graph.invoke(
            {
                "topic": request.topic.strip(),
                "tone": request.tone.strip(),
                "duration": request.duration.strip(),
                "language": request.language.strip(),
                "characters": request.characters.strip(),
                "output_dir": str(ROOT / "outputs" / thread_id),
            },
            config,
        )
        return _graph_response(result, thread_id)
    except Exception as exc:
        logger.exception("Video creation failed")
        raise HTTPException(status_code=500, detail={"error": str(exc)}) from exc


@app.post("/api/review-story")
def submit_story_review(request: StoryReviewRequest) -> dict:
    thread_id = request.thread_id.strip()
    if not thread_id:
        raise HTTPException(status_code=400, detail="Thread ID is required.")

    command = {"action": request.action}
    if request.action == "edit":
        if not request.story or not request.story.strip():
            raise HTTPException(status_code=400, detail="Edited story is required.")
        command["story"] = request.story
    elif request.action == "regenerate":
        command["note"] = (request.note or "").strip() or "Regenerate the story."

    return _resume_review(thread_id, "review_story", command)


@app.post("/api/review-storyboard")
def submit_storyboard_review(request: StoryboardReviewRequest) -> dict:
    thread_id = request.thread_id.strip()
    if not thread_id:
        raise HTTPException(status_code=400, detail="Thread ID is required.")

    command = {"action": request.action}
    if request.action == "regenerate":
        command["note"] = (request.note or "").strip() or "Regenerate the storyboard."
    return _resume_review(thread_id, "review_storyboard", command)


@app.post("/api/review-visual-storyboard")
def submit_visual_storyboard_review(request: VisualStoryboardReviewRequest) -> dict:
    thread_id = request.thread_id.strip()
    if not thread_id:
        raise HTTPException(status_code=400, detail="Thread ID is required.")

    command = {"action": request.action}
    if request.action == "regenerate":
        feedback = [
            {"scene_number": item.scene_number, "note": item.note.strip()}
            for item in request.feedback
            if item.note.strip()
        ]
        if not feedback:
            raise HTTPException(status_code=400, detail="Add feedback to at least one scene.")
        scene_numbers = [item["scene_number"] for item in feedback]
        if len(scene_numbers) != len(set(scene_numbers)):
            raise HTTPException(status_code=400, detail="Each scene can have only one feedback note.")
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = graph.get_state(config)
        if "review_visual_storyboard" not in snapshot.next:
            raise HTTPException(status_code=404, detail="Review session not found or already completed.")
        known_scenes = {scene["scene_number"] for scene in snapshot.values.get("storyboard", [])}
        if not set(scene_numbers) <= known_scenes:
            raise HTTPException(status_code=400, detail="Feedback contains an unknown scene number.")
        command["feedback"] = feedback
    return _resume_review(thread_id, "review_visual_storyboard", command)


@app.get("/output/{path:path}")
def output_file(path: str) -> FileResponse:
    file_path = (ROOT / path).resolve()
    try:
        file_path.relative_to(ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="File not found.") from exc
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
