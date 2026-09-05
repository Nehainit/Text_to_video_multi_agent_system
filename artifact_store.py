import json
import mimetypes
import os
import time
from datetime import UTC, datetime
from pathlib import Path

try:
    from minio import Minio
except ModuleNotFoundError:
    Minio = None


FILE_FIELDS = (
    "mood_board_file",
    "character_reference_files",
    "image_files",
    "scene_video_files",
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


def _files(state: dict) -> list[Path]:
    paths = []
    for field in FILE_FIELDS:
        value = state.get(field)
        for item in value if isinstance(value, list) else [value]:
            if item and Path(item).is_file():
                paths.append(Path(item))
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
                        "storyboard",
                        "mood_board_file",
                        "image_files",
                        "scene_video_files",
                        "video_file",
                        "visual_feedback_history",
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
