import json
import os
from pathlib import Path

import artifact_store


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


def test_archives_manifest_and_files_to_minio(tmp_path):
    image = tmp_path / "images" / "scene_01_v001.png"
    image.parent.mkdir()
    image.write_bytes(b"image")
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
    assert fake.created == ["test-artifacts"]
    assert storage["backend"] == "minio"
    assert storage["objects"] == [
        "thread-1/images/scene_01_v001.png",
        f"thread-1/artifact_history/{manifest.name}",
    ]


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        test_archives_manifest_and_files_to_minio(Path(tmp))
