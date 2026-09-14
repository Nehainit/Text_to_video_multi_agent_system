import threading
from pathlib import Path

import shot_video_generation_agent


def _state(tmp_path, count=3, quality_mode="standard"):
    images = []
    shots = []
    plans = []
    for number in range(1, count + 1):
        image = tmp_path / f"shot-{number:03}.png"
        image.write_bytes(b"image")
        images.append(str(image))
        shots.append({"shot_id": f"shot-{number:03}", "scene_id": "scene-001"})
        plans.append({
            "shot_id": f"shot-{number:03}", "duration_seconds": 5,
            "video_prompt": f"Animate shot {number}.", "negative_prompt": "No distortion.",
        })
    return {
        "thread_id": "thread", "topic": "Kindness", "output_dir": str(tmp_path),
        "quality_mode": quality_mode, "shot_plan": shots, "image_files": images,
        "motion_plans": plans,
    }


def test_shots_generate_concurrently_and_keep_storyboard_order(monkeypatch, tmp_path):
    state = _state(tmp_path)
    active = 0
    peak = 0
    lock = threading.Lock()
    overlap = threading.Event()

    def generate(_state, _scene, _image_file, video_file, *_args):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                overlap.set()
        try:
            assert overlap.wait(2), "video jobs did not overlap"
            video_file.write_bytes(b"video")
            return str(video_file)
        finally:
            with lock:
                active -= 1

    monkeypatch.setenv("MINIO_PUBLIC_ENDPOINT", "https://media.example.com")
    monkeypatch.setenv("VIDEO_GENERATION_WORKERS", "2")
    monkeypatch.setattr(shot_video_generation_agent, "public_artifact_url", lambda _state, path: f"https://media.example.com/{Path(path).name}")
    monkeypatch.setattr(shot_video_generation_agent, "_generate_scene_video_file", generate)

    result = shot_video_generation_agent.create_shot_videos(state)

    assert peak == 2
    assert [item["shot_id"] for item in result["generated_videos"]] == ["shot-001", "shot-002", "shot-003"]
    assert "video_candidates" not in result


def test_refine_retry_generates_only_rejected_shot(monkeypatch, tmp_path):
    state = _state(tmp_path, count=2, quality_mode="refine")
    state.update(
        generated_videos=[
            {"shot_id": "shot-001", "video_file": "approved.mp4"},
            {"shot_id": "shot-002", "video_file": "rejected.mp4"},
        ],
        video_candidates=[{"shot_id": "shot-001", "candidate_id": "shot-001-c1"}],
        video_retry_shots=["shot-002"],
        video_refinement_round=0,
    )
    calls = []

    def generate(_state, scene, _image_file, video_file, *_args):
        calls.append(scene["shot_id"])
        video_file.write_bytes(b"video")
        return str(video_file)

    monkeypatch.setenv("MINIO_PUBLIC_ENDPOINT", "https://media.example.com")
    monkeypatch.setattr(shot_video_generation_agent, "public_artifact_url", lambda _state, path: f"https://media.example.com/{Path(path).name}")
    monkeypatch.setattr(shot_video_generation_agent, "_generate_scene_video_file", generate)

    result = shot_video_generation_agent.create_shot_videos(state)

    assert calls == ["shot-002"]
    assert result["generated_videos"][0]["video_file"] == "approved.mp4"
    assert result["video_candidates"][-1]["candidate_id"] == "shot-002-c2"


def test_shot_video_generation_retries_technical_failure_and_isolates_missing_image(monkeypatch, tmp_path):
    source = tmp_path / "shot-001.png"
    source.write_bytes(b"image")
    calls = []

    def generate(_state, scene, image_file, video_file, prompt, image_url, negative_prompt, cfg_scale, aspect_ratio, generate_audio):
        calls.append({
            "scene": scene, "image_file": image_file, "prompt": prompt,
            "image_url": image_url, "negative_prompt": negative_prompt,
            "cfg_scale": cfg_scale, "aspect_ratio": aspect_ratio, "generate_audio": generate_audio,
        })
        if len(calls) == 1:
            raise RuntimeError("temporary provider timeout")
        video_file.write_bytes(b"video")
        return str(video_file)

    monkeypatch.setenv("MINIO_PUBLIC_ENDPOINT", "https://media.example.com")
    monkeypatch.setattr(shot_video_generation_agent, "public_artifact_url", lambda _state, path: f"https://media.example.com/{Path(path).name}")
    monkeypatch.setattr(shot_video_generation_agent, "_generate_scene_video_file", generate)
    result = shot_video_generation_agent.create_shot_videos({
        "topic": "Kindness", "output_dir": str(tmp_path),
        "aspect_ratio": "16:9", "video_quality": "high",
        "shot_plan": [
            {"shot_id": "shot-001", "scene_id": "scene-001"},
            {"shot_id": "shot-002", "scene_id": "scene-001"},
        ],
        "image_files": [str(source), str(tmp_path / "missing.png")],
        "motion_plans": [
            {
                "shot_id": "shot-001", "duration_seconds": 4.4,
                "video_prompt": "Use only approved gentle breathing.",
                "negative_prompt": "No identity drift.",
            },
            {
                "shot_id": "shot-002", "duration_seconds": 5,
                "video_prompt": "Use only approved motion.",
                "negative_prompt": "No distortion.",
            },
        ],
    })

    assert len(calls) == 2
    assert all(item["prompt"] == "Use only approved gentle breathing." for item in calls)
    assert all(item["negative_prompt"] == "No identity drift." for item in calls)
    assert all(item["cfg_scale"] == 0.5 and item["aspect_ratio"] == "widescreen_16_9" for item in calls)
    assert all(item["generate_audio"] is False for item in calls)
    assert result["generated_videos"][0]["generation_status"] == "success"
    assert result["generated_videos"][0]["generation_attempt"] == 2
    assert result["generated_videos"][0]["requested_duration_seconds"] == 4.4
    assert result["generated_videos"][0]["generated_duration_seconds"] == 5.0
    assert result["generated_videos"][0]["aspect_ratio"] == "16:9"
    assert result["generated_videos"][0]["video_quality"] == "high"
    assert result["generated_videos"][1]["failure_type"] == "missing_source_image"
    assert result["scene_video_files"] == [result["generated_videos"][0]["video_file"], None]


def test_local_mode_creates_real_fallback_and_versions_retries(monkeypatch, tmp_path):
    state = _state(tmp_path, count=1)
    state["motion_plans"][0]["duration_seconds"] = 6
    calls = []

    def fallback(_image, duration, _motion, output, width, height):
        calls.append((duration, output.name, width, height))
        output.write_bytes(b"video")

    monkeypatch.delenv("MINIO_PUBLIC_ENDPOINT", raising=False)
    monkeypatch.setattr(shot_video_generation_agent, "_clip", fallback)
    first = shot_video_generation_agent.create_shot_videos(state)
    retried = shot_video_generation_agent.create_shot_videos({**state, **first, "video_retry_shots": ["shot-001"]})
    retried_again = shot_video_generation_agent.create_shot_videos({**state, **retried, "video_retry_shots": ["shot-001"]})

    assert first["generated_videos"][0]["model_used"] == "ffmpeg"
    assert first["generated_videos"][0]["generated_duration_seconds"] == 6
    assert calls == [
        (6, "shot-001-c1.mp4", 1080, 1920),
        (6, "shot-001-c2.mp4", 1080, 1920),
        (6, "shot-001-c3.mp4", 1080, 1920),
    ]


def test_timeline_retry_generates_the_required_longer_clip(monkeypatch, tmp_path):
    state = _state(tmp_path, count=1)
    state.update(
        generated_videos=[{"shot_id": "shot-001", "video_file": "short.mp4"}],
        video_retry_shots=["shot-001"],
        timeline_issues=[{
            "type": "insufficient_clip_duration",
            "shot_id": "shot-001",
            "required_duration_seconds": 7.25,
        }],
    )
    durations = []

    def fallback(_image, duration, _motion, output, _width, _height):
        durations.append(duration)
        output.write_bytes(b"video")

    monkeypatch.delenv("MINIO_PUBLIC_ENDPOINT", raising=False)
    monkeypatch.setattr(shot_video_generation_agent, "_clip", fallback)

    result = shot_video_generation_agent.create_shot_videos(state)

    assert durations == [7.25]
    assert result["generated_videos"][0]["requested_duration_seconds"] == 7.25
    assert result["generated_videos"][0]["generated_duration_seconds"] == 7.25


def test_provider_duration_rounds_up_to_cover_the_shot(monkeypatch, tmp_path):
    state = _state(tmp_path, count=1)
    state["motion_plans"][0]["duration_seconds"] = 6
    durations = []

    def generate(_state, scene, _image, output, *_args):
        durations.append(scene["duration_seconds"])
        output.write_bytes(b"video")
        return str(output)

    monkeypatch.setenv("MINIO_PUBLIC_ENDPOINT", "https://media.example.com")
    monkeypatch.setattr(shot_video_generation_agent, "public_artifact_url", lambda *_args: "https://media.example.com/image.png")
    monkeypatch.setattr(shot_video_generation_agent, "_generate_scene_video_file", generate)

    result = shot_video_generation_agent.create_shot_videos(state)

    assert durations == [10]
    assert result["generated_videos"][0]["generated_duration_seconds"] == 10
