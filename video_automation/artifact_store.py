import json
import mimetypes
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

try:
    from minio import Minio
except ModuleNotFoundError:
    Minio = None


FILE_FIELDS = (
    "mood_board_file",
    "character_board_file",
    "character_reference_files",
    "image_files",
    "scene_video_files",
    "rough_cut_file",
    "narration_file",
    "narration_alignment_file",
    "sfx_files",
    "subtitle_file",
    "mixed_audio_file",
    "video_file",
)


def _client():
    values = [os.getenv(name) for name in ("MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY")]
    if not any(values):
        return None
    if not all(values):
        raise RuntimeError("Set MINIO_ENDPOINT, MINIO_ACCESS_KEY, and MINIO_SECRET_KEY together.")
    if Minio is None:
        raise RuntimeError("Install the 'minio' package to enable object storage.")
    return Minio(
        values[0],
        access_key=values[1],
        secret_key=values[2],
        secure=os.getenv("MINIO_SECURE", "true").lower() not in {"0", "false", "no"},
    )


def public_artifact_url(state: dict, file_path: str) -> str | None:
    public_endpoint = os.getenv("MINIO_PUBLIC_ENDPOINT")
    if not public_endpoint:
        return None
    client = _client()
    if client is None:
        raise RuntimeError("Configure local MinIO storage before publishing image-to-video inputs.")
    parsed = urlparse(public_endpoint if "://" in public_endpoint else f"https://{public_endpoint}")
    bucket = os.getenv("MINIO_BUCKET", "softframe-artifacts")
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    path = Path(file_path)
    output_dir = Path(state["output_dir"])
    object_name = f"{state['thread_id']}/{path.resolve().relative_to(output_dir.resolve()).as_posix()}"
    client.fput_object(
        bucket,
        object_name,
        str(path),
        content_type=mimetypes.guess_type(path)[0] or "application/octet-stream",
    )
    public_client = Minio(
        parsed.netloc or parsed.path,
        access_key=os.environ["MINIO_ACCESS_KEY"],
        secret_key=os.environ["MINIO_SECRET_KEY"],
        secure=parsed.scheme == "https",
        region=os.getenv("MINIO_REGION", "us-east-1"),
    )
    return public_client.presigned_get_object(bucket, object_name, expires=timedelta(minutes=30))


def _files(state: dict) -> list[Path]:
    paths = []
    seen = set()
    for field in FILE_FIELDS:
        value = state.get(field)
        for item in value if isinstance(value, list) else [value]:
            path = Path(item) if item else None
            if path and path.is_file() and path.resolve() not in seen:
                paths.append(path)
                seen.add(path.resolve())
    for candidate in state.get("video_candidates", []):
        path = Path(candidate["file"]) if candidate.get("file") else None
        if path and path.is_file() and path.resolve() not in seen:
            paths.append(path)
            seen.add(path.resolve())
    return paths


def archive_artifacts(state: dict, thread_id: str, stage: str) -> dict:
    output_dir = Path(state["output_dir"])
    history_dir = output_dir / "artifact_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    version = time.time_ns()
    manifest = history_dir / f"{version}-{stage}.json"
    manifest.write_text(
        json.dumps(
            {
                "thread_id": thread_id,
                "stage": stage,
                "created_at": datetime.now(UTC).isoformat(),
                **{
                    key: state.get(key)
                    for key in (
                        "topic",
                        "story",
                        "story_outline",
                        "story_beats",
                        "parsed_requirements",
                        "llm_evaluations",
                        "planning_attempts",
                        "narration_script",
                        "narration_segments",
                        "estimated_narration_seconds",
                        "actual_narration_seconds",
                        "narration_file",
                        "narration_alignment_file",
                        "narration_segment_timings",
                        "scene_timings",
                        "needs_story_revision",
                        "story_revision_reason",
                        "scenes",
                        "scene_plan_needs_revision",
                        "scene_plan_revision_reason",
                        "scene_plan_issues",
                        "visual_beats",
                        "visual_plan_needs_revision",
                        "visual_plan_revision_reason",
                        "visual_plan_issues",
                        "shot_plan",
                        "shot_plan_needs_revision",
                        "shot_plan_revision_reason",
                        "shot_plan_issues",
                        "image_prompt_requests",
                        "generated_images",
                        "shot_image_qa_results",
                        "motion_plans",
                        "motion_plan_needs_revision",
                        "motion_plan_revision_reason",
                        "generated_videos",
                        "genre",
                        "genre_structure",
                        "character_board_file",
                        "director_plan",
                        "storyboard",
                        "mood_board_file",
                        "image_files",
                        "scene_video_files",
                        "quality_mode",
                        "aspect_ratio",
                        "video_quality",
                        "video_candidates",
                        "video_validation_results",
                        "video_validation_retry_counts",
                        "video_validation_exhausted_shots",
                        "edit_timeline",
                        "narration_duration_seconds",
                        "timeline_duration_seconds",
                        "timeline_valid",
                        "timeline_issues",
                        "timeline_retry_requests",
                        "edit_timeline_retry_counts",
                        "edit_timeline_exhausted_shots",
                        "compilation_status",
                        "rough_cut_file",
                        "duration_seconds",
                        "width",
                        "height",
                        "fps",
                        "shot_timestamps",
                        "sync_valid",
                        "compilation_issues",
                        "compilation_error",
                        "rough_cut_failure_source",
                        "rough_cut_retry_count",
                        "rough_cut_retry_exhausted",
                        "combined_video_judge_report",
                        "combined_video_judge_approved",
                        "combined_video_judge_status",
                        "combined_video_judge_error",
                        "combined_video_judge_error_count",
                        "combined_video_judge_error_exhausted",
                        "combined_video_judge_model",
                        "combined_video_judge_retry_target",
                        "combined_video_judge_retry_shots",
                        "video_judge_refinement_round",
                        "video_judge_refinement_exhausted",
                        "pipeline_status",
                        "video_file",
                        "story_feedback_history",
                        "reference_feedback_history",
                        "director_feedback_history",
                        "visual_feedback_history",
                        "warnings",
                    )
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    client = _client()
    if client is None:
        return {"backend": "local", "manifest": str(manifest)}

    bucket = os.getenv("MINIO_BUCKET", "softframe-artifacts")
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    objects = []
    for path in [*_files(state), manifest]:
        try:
            relative = path.resolve().relative_to(output_dir.resolve())
        except ValueError:
            continue
        object_name = f"{thread_id}/{relative.as_posix()}"
        client.fput_object(
            bucket,
            object_name,
            str(path),
            content_type=mimetypes.guess_type(path)[0] or "application/octet-stream",
        )
        objects.append(object_name)
    return {"backend": "minio", "bucket": bucket, "manifest": objects[-1], "objects": objects}
