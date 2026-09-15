import json
import os
from pathlib import Path

from video_automation import artifact_store


class FakeMinio:
    def __init__(self):
        self.uploads = []
        self.created = []

    def bucket_exists(self, bucket):
        return False

    def make_bucket(self, bucket):
        self.created.append(bucket)

    def fput_object(self, bucket, object_name, path, content_type):
        self.uploads.append((bucket, object_name, path, content_type))


class FakePublicMinio:
    def __init__(self):
        self.presigned = []

    def bucket_exists(self, _bucket):
        raise AssertionError("Public MinIO endpoint must not be used for storage operations.")

    def presigned_get_object(self, bucket, object_name, expires):
        self.presigned.append((bucket, object_name, expires))
        return f"https://media.example.com/{bucket}/{object_name}"


def test_archives_manifest_and_files_to_minio(tmp_path):
    image = tmp_path / "images" / "scene_01_v001.png"
    image.parent.mkdir()
    image.write_bytes(b"image")
    candidate = tmp_path / "scene_videos" / "shot-001-c1.mp4"
    candidate.parent.mkdir()
    candidate.write_bytes(b"video")
    fake = FakeMinio()
    original_client = artifact_store._client
    original_bucket = os.environ.get("MINIO_BUCKET")
    artifact_store._client = lambda: fake
    os.environ["MINIO_BUCKET"] = "test-artifacts"
    try:
        storage = artifact_store.archive_artifacts(
            {
                "output_dir": str(tmp_path),
                "topic": "A story",
                "story": "Once upon a time.",
                "storyboard": [{"scene_number": 1}],
                "image_files": [str(image)],
                "quality_mode": "refine",
                "video_candidates": [{"candidate_id": "shot-001-c1", "file": str(candidate)}],
            },
            "thread-1",
            "visual-storyboard-review",
        )
    finally:
        artifact_store._client = original_client
        if original_bucket is None:
            os.environ.pop("MINIO_BUCKET", None)
        else:
            os.environ["MINIO_BUCKET"] = original_bucket

    manifest = next((tmp_path / "artifact_history").glob("*-visual-storyboard-review.json"))
    assert json.loads(manifest.read_text())["story"] == "Once upon a time."
    assert json.loads(manifest.read_text())["quality_mode"] == "refine"
    assert fake.created == ["test-artifacts"]
    assert storage["backend"] == "minio"
    assert storage["objects"] == [
        "thread-1/images/scene_01_v001.png",
        "thread-1/scene_videos/shot-001-c1.mp4",
        f"thread-1/artifact_history/{manifest.name}",
    ]


def test_public_url_uploads_locally_and_only_signs_with_public_endpoint(tmp_path, monkeypatch):
    image = tmp_path / "images" / "shot-001.png"
    image.parent.mkdir()
    image.write_bytes(b"image")
    local = FakeMinio()
    public = FakePublicMinio()
    monkeypatch.setattr(artifact_store, "_client", lambda: local)
    monkeypatch.setattr(artifact_store, "Minio", lambda *_args, **_kwargs: public)
    monkeypatch.setenv("MINIO_PUBLIC_ENDPOINT", "https://media.example.com")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "test-access")
    monkeypatch.setenv("MINIO_SECRET_KEY", "test-secret")
    monkeypatch.setenv("MINIO_BUCKET", "test-artifacts")

    url = artifact_store.public_artifact_url(
        {"thread_id": "thread-1", "output_dir": str(tmp_path)}, str(image)
    )

    assert local.created == ["test-artifacts"]
    assert local.uploads[0][:2] == ("test-artifacts", "thread-1/images/shot-001.png")
    assert public.presigned[0][:2] == ("test-artifacts", "thread-1/images/shot-001.png")
    assert url == "https://media.example.com/test-artifacts/thread-1/images/shot-001.png"


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        test_archives_manifest_and_files_to_minio(Path(tmp))
