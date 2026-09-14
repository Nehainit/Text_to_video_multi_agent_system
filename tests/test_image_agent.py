import base64
import json
import os
import threading
from pathlib import Path

from video_automation.agents import image_agent
from PIL import Image


PNG_1X1 = base64.b64encode(
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00"
    b"\x00\x04\x00\x01\xf6\x178U\x00\x00\x00\x00IEND\xaeB`\x82"
).decode("ascii")


def shot(number, visual):
    return {
        "shot_id": f"shot-{number:03}",
        "scene_id": "scene-001",
        "scene_number": number,
        "characters_present": ["raja_001"],
        "duration_seconds": 5,
        "start_time": f"00:{(number - 1) * 5:02}",
        "end_time": f"00:{number * 5:02}",
        "visuals": visual,
        "camera": "wide 35mm eye level tracking composition",
        "motion": "pan_right",
        "subject_motion": "Raja walks naturally.",
        "continuity_in": "Raja wears a red coat.",
        "continuity_out": "Raja wears a red coat.",
        "narration": "Raja walks.",
    }


def with_fake_images(callback):
    original_request = image_agent._json_request
    original_headers = image_agent._magnific_headers
    original_sleep = image_agent.time.sleep
    calls = []

    def fake_request(url, payload, _headers):
        calls.append((url, payload))
        return {"data": [{"base64": PNG_1X1}]}

    image_agent._json_request = fake_request
    image_agent._magnific_headers = lambda: {}
    image_agent.time.sleep = lambda _seconds: None
    try:
        callback(calls)
    finally:
        image_agent._json_request = original_request
        image_agent._magnific_headers = original_headers
        image_agent.time.sleep = original_sleep


def test_image_prompt_builder_preserves_approved_shot_and_references(monkeypatch):
    shot_plan = [{
        "shot_id": "shot-001", "scene_id": "scene-001", "visual_beat_ids": ["vb-001"],
        "source_segment_ids": ["segment-001"], "characters_present": ["raja_001"],
        "primary_subject": "raja_001", "visual_action": "Raja stands safely outside his home.",
        "visual_focus": "Raja standing safely outside his home after returning", "framing": "medium",
        "camera_angle": "eye_level",
        "composition": "Raja occupies the foreground while the entrance of his home remains clearly visible behind him.",
        "emotion": "relieved", "location_id": "location-001", "estimated_duration_seconds": 3.0,
        "continuity_in": "Raja has reached the path outside his home.",
        "continuity_out": "Raja remains safely outside his home.", "shot_purpose": "Show the homecoming.",
    }]
    prompt = (
        "Single still keyframe. Raja stands safely outside his home. "
        "Visual focus: Raja standing safely outside his home after returning. Framing: medium. "
        "Camera angle: eye_level. Composition: Raja occupies the foreground while the entrance of his home remains clearly visible behind him. "
        "Emotion: relieved. Location: cottage path. Visual style: storybook watercolor. "
        "Preserve raja_001 exactly from the supplied approved character reference."
    )
    monkeypatch.setattr(image_agent, "load_model", lambda _name: (_ for _ in ()).throw(AssertionError("model should not run")))
    result = image_agent.build_image_prompts({
        "shot_plan": shot_plan,
        "parsed_requirements": {"visual_style": "storybook watercolor"},
        "production_bible": {
            "visual_style": "storybook watercolor",
            "characters": [{"character_id": "raja_001", "name": "Raja"}],
            "locations": [{"location_id": "location-001", "description": "cottage path"}],
        },
        "character_reference_files": ["raja.png"],
    })

    assert result["image_prompt_requests"][0]["character_reference_ids"] == ["raja_001"]
    assert result["image_prompt_requests"][0]["scene_id"] == "scene-001"
    assert result["image_prompt_requests"][0]["visual_beat_ids"] == ["vb-001"]
    assert result["image_prompt_requests"][0]["source_segment_ids"] == ["segment-001"]
    assert result["image_prompt_requests"][0]["location_id"] == "location-001"
    built_prompt = result["image_prompt_requests"][0]["image_prompt"]
    assert all(part in built_prompt for part in (
        "Raja stands safely outside his home.", "Framing: medium.",
        "Location: cottage path.", "Visual style: storybook watercolor.", "Preserve raja_001",
    ))
    assert result["llm_evaluations"] == []


def test_image_prompt_request_validator_reports_traceability_mismatches():
    approved = {
        "shot_id": "shot-001", "scene_id": "scene-001", "visual_beat_ids": ["vb-001"],
        "source_segment_ids": ["segment-001"], "location_id": "location-001",
        "characters_present": ["raja_001"],
    }
    request = {
        "shot_id": "shot-002", "scene_id": "scene-002", "visual_beat_ids": ["vb-002"],
        "source_segment_ids": ["segment-002"], "location_id": "location-002",
        "character_reference_ids": ["chuha_001"],
        "location_reference": {"location_id": "location-003"},
    }

    assert image_agent.validate_image_prompt_request(approved, request) == [
        "shot_id mismatch", "scene_id mismatch", "visual_beat_ids mismatch", "source_segment_ids mismatch",
        "location mismatch: expected location-001, got location-002",
        "character references do not match approved shot", "location_reference does not match location_id",
    ]


def test_video_probe_timeout_marks_video_unusable(monkeypatch, tmp_path):
    video = tmp_path / "stuck.mp4"
    video.write_bytes(b"video")

    def timeout(*_args, **_kwargs):
        raise image_agent.subprocess.TimeoutExpired("ffprobe", 30)

    monkeypatch.setattr(image_agent.subprocess, "run", timeout)
    assert image_agent._video_is_usable(video) is False


def test_visual_storyboard_uses_built_image_prompt(tmp_path):
    def run(calls):
        reference = tmp_path / "reference.png"
        reference.write_bytes(b"character")
        built_prompt = "Prepared single-keyframe prompt for shot-001."
        approved = {
            "shot_id": "shot-001", "scene_id": "scene-001", "visual_beat_ids": ["vb-001"],
            "source_segment_ids": ["segment-001"], "characters_present": ["raja_001"],
            "location_id": "location-001",
        }
        result = image_agent.create_visual_storyboard({
            "topic": "Raja returns home",
            "output_dir": str(tmp_path),
            "aspect_ratio": "16:9",
            "video_quality": "high",
            "production_bible": {"characters": [{"character_id": "raja_001"}]},
            "character_reference_files": [str(reference)],
            "shot_plan": [approved],
            "image_prompt_requests": [{
                "shot_id": "shot-001", "scene_id": "scene-001", "visual_beat_ids": ["vb-001"],
                "source_segment_ids": ["segment-001"], "image_prompt": built_prompt,
                "character_reference_ids": ["raja_001"],
                "location_id": "location-001",
                "location_reference": {"location_id": "location-001", "description": "home"},
                "previous_shot_id": None,
            }],
        })

        assert calls[0][1]["prompt"] == built_prompt
        assert calls[0][1]["aspect_ratio"] == "16:9"
        assert calls[0][1]["resolution"] == "2K"
        assert result["generated_images"][0]["generation_status"] == "success"
        assert result["generated_images"][0]["aspect_ratio"] == "16:9"
        assert result["generated_images"][0]["resolution"] == "2K"
        assert result["generated_images"][0]["generation_attempt"] == 1
        assert result["generated_images"][0]["image_path"] == result["image_files"][0]

    with_fake_images(run)


def _shot_image_state(tmp_path, previous_ids):
    shots = []
    requests = []
    for number, previous_id in enumerate(previous_ids, start=1):
        shot_id = f"shot-{number:03}"
        shot = {
            "shot_id": shot_id, "scene_id": "scene-001", "visual_beat_ids": [f"vb-{number:03}"],
            "source_segment_ids": [f"segment-{number:03}"], "characters_present": [],
            "location_id": "location-001",
        }
        shots.append(shot)
        requests.append({
            "shot_id": shot_id, "scene_id": "scene-001", "visual_beat_ids": shot["visual_beat_ids"],
            "source_segment_ids": shot["source_segment_ids"], "image_prompt": f"Still for {shot_id}.",
            "character_reference_ids": [], "location_id": "location-001",
            "location_reference": {"location_id": "location-001"}, "previous_shot_id": previous_id,
        })
    return {
        "thread_id": "thread", "topic": "Parallel images", "output_dir": str(tmp_path),
        "shot_plan": shots, "image_prompt_requests": requests,
    }


def test_independent_shot_images_generate_concurrently_in_order(monkeypatch, tmp_path):
    state = _shot_image_state(tmp_path, [None, None, None])
    active = 0
    peak = 0
    lock = threading.Lock()
    overlap = threading.Event()

    def generate(_prompt, _aspect_ratio, image_file, _references, **_kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                overlap.set()
        try:
            assert overlap.wait(2), "image jobs did not overlap"
            image_file.write_bytes(b"image")
        finally:
            with lock:
                active -= 1

    monkeypatch.setenv("IMAGE_GENERATION_WORKERS", "2")
    monkeypatch.setattr(image_agent, "_generate_reference_image", generate)

    result = image_agent.create_visual_storyboard(state)

    assert peak == 2
    assert [item["shot_id"] for item in result["generated_images"]] == ["shot-001", "shot-002", "shot-003"]
    assert all(result["image_files"])


def test_shot_image_waits_for_previous_image(monkeypatch, tmp_path):
    state = _shot_image_state(tmp_path, [None, "shot-001"])
    calls = []

    def generate(_prompt, _aspect_ratio, image_file, references, **_kwargs):
        calls.append((image_file.stem.rsplit("_v", 1)[0], list(references)))
        image_file.write_bytes(b"image")

    monkeypatch.setattr(image_agent, "_generate_reference_image", generate)

    result = image_agent.create_visual_storyboard(state)

    assert [shot_id for shot_id, _ in calls] == ["shot-001", "shot-002"]
    assert result["image_files"][0] in calls[1][1]


def test_shot_image_dependency_cycle_stops_without_generation(monkeypatch, tmp_path):
    state = _shot_image_state(tmp_path, ["shot-002", "shot-001"])
    calls = []
    monkeypatch.setattr(image_agent, "_generate_reference_image", lambda *_args, **_kwargs: calls.append(True))

    result = image_agent.create_visual_storyboard(state)

    assert calls == []
    assert all(item["generation_status"] == "failed" for item in result["generated_images"])
    assert all("dependency cycle" in item["error"] for item in result["generated_images"])


def test_shot_image_generation_keeps_success_when_another_shot_exhausts_retries(monkeypatch, tmp_path):
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"character")
    shots = [
        {
            "shot_id": f"shot-{number:03}", "scene_id": "scene-001", "visual_beat_ids": [f"vb-{number:03}"],
            "source_segment_ids": [f"segment-{number:03}"], "characters_present": ["raja_001"],
            "location_id": "location-001",
        }
        for number in (1, 2)
    ]
    requests = [
        {
            "shot_id": item["shot_id"], "scene_id": item["scene_id"],
            "visual_beat_ids": item["visual_beat_ids"], "source_segment_ids": item["source_segment_ids"],
            "image_prompt": f"Approved still for {item['shot_id']}.",
            "character_reference_ids": ["raja_001"], "character_reference_files": [str(reference)],
            "location_id": "location-001",
            "location_reference": {"location_id": "location-001", "description": "home"},
            "previous_shot_id": "shot-001" if item["shot_id"] == "shot-002" else None,
        }
        for item in shots
    ]
    attempts = {item["shot_id"]: 0 for item in shots}
    second_references = []

    def generate(prompt, _size, image_file, reference_files, **kwargs):
        shot_id = image_file.stem.rsplit("_v", 1)[0]
        attempts[shot_id] += 1
        assert kwargs["improve_prompt"] is False
        if shot_id == "shot-002":
            assert "Preserve Raja's approved face." in prompt
            second_references[:] = reference_files
            raise RuntimeError("provider unavailable")
        image_file.write_bytes(b"image")

    monkeypatch.setattr(image_agent, "_generate_reference_image", generate)
    result = image_agent.create_visual_storyboard({
        "topic": "Raja returns home", "output_dir": str(tmp_path),
        "shot_plan": shots, "image_prompt_requests": requests,
        "shot_image_qa_retry_shots": ["shot-002"],
        "shot_image_qa_results": [{
            "shot_id": "shot-002",
            "issues": [{"correction": "Preserve Raja's approved face."}],
        }],
    })

    assert [item["generation_status"] for item in result["generated_images"]] == ["success", "failed"]
    assert attempts == {"shot-001": 1, "shot-002": 3}
    assert Path(result["image_files"][0]).is_file() and result["image_files"][1] is None
    assert result["generated_images"][1]["generation_attempt"] == 3
    assert result["generated_images"][1]["error"] == "provider unavailable"
    assert result["image_files"][0] in second_references


def test_reference_package_is_versioned_and_regenerated_together(tmp_path):
    def run(calls):
        state = {
            "topic": "Raja returns home",
            "tone": "cinematic",
            "characters": ["Raja — red coat"],
            "story": "Raja returns home.",
            "production_bible": {"theme": "homecoming"},
            "output_dir": str(tmp_path),
        }
        first = image_agent.create_reference_package(state)
        second = image_agent.create_reference_package({**state, **first, "reference_feedback": "use colder moonlight"})

        assert Path(first["character_board_file"]).name == "characters_reference_v001.png"
        assert Path(second["character_board_file"]).name == "characters_reference_v002.png"
        assert Path(second["mood_board_file"]).name == "mood_board_v002.png"
        assert all(Path(path).exists() for path in (first["character_board_file"], first["mood_board_file"], second["character_board_file"], second["mood_board_file"]))
        assert len(calls) == 4
        assert "layout reference only" in calls[0][1]["reference_images"][0]["text"]
        assert "use colder moonlight" in calls[-1][1]["prompt"]

    with_fake_images(run)


def test_regenerates_only_selected_shot(tmp_path):
    def run(calls):
        reference = tmp_path / "reference.png"
        mood = tmp_path / "mood.png"
        reference.write_bytes(b"x")
        mood.write_bytes(b"x")
        state = {
            "topic": "Selective regeneration",
            "tone": "cinematic",
            "characters": ["Raja — red coat"],
            "story": "Raja crosses a forest and reaches a lake.",
            "production_bible": {
                "theme": "one moonlit journey",
                "characters": [
                    {"character_id": "raja_001", "name": "Raja", "appearance": "traveler", "wardrobe": "red coat", "props": []}
                ],
            },
            "character_reference_files": [str(reference)],
            "mood_board_file": str(mood),
            "output_dir": str(tmp_path),
            "storyboard": [shot(1, "Raja in a forest."), shot(2, "Raja at a lake.")],
        }
        first = image_agent.create_visual_storyboard(state)
        second = image_agent.create_visual_storyboard(
            {**state, **first, "visual_feedback": [{"shot_id": "shot-002", "note": "make the lake brighter"}]}
        )

        assert first["image_files"][0] == second["image_files"][0]
        assert first["image_files"][1] != second["image_files"][1]
        assert Path(second["image_files"][1]).name == "shot-002_v002.png"
        assert len(calls) == 3
        assert calls[0][0] == image_agent.MAGNIFIC_REFERENCE_IMAGE_URL
        assert "Approved continuity context" in calls[0][1]["prompt"]
        assert "Original user request (highest priority): Selective regeneration" in calls[0][1]["prompt"]
        assert "Never infer status" in calls[0][1]["prompt"]
        assert not {"king", "queen", "royal"} & set(calls[0][1]["prompt"].lower().split())
        assert calls[0][1]["reference_images"][0]["image"] == base64.b64encode(b"x").decode("ascii")
        assert calls[0][1]["reference_images"][1]["image"] == base64.b64encode(b"x").decode("ascii")
        assert "make the lake brighter" in calls[-1][1]["prompt"]

    with_fake_images(run)


def test_reference_package_generates_one_isolated_sheet_per_character(tmp_path):
    def run(calls):
        state = {
            "topic": "A lion and mouse become friends",
            "tone": "playful",
            "characters": ["Leo — lion", "Pip — mouse"],
            "story": "Leo the lion helps Pip the mouse.",
            "production_bible": {
                "theme": "friendship",
                "characters": [
                    {"name": "Leo", "appearance": "golden lion", "wardrobe": "none", "props": []},
                    {"name": "Pip", "appearance": "brown mouse", "wardrobe": "none", "props": []},
                ],
            },
            "output_dir": str(tmp_path),
        }
        result = image_agent.create_reference_package(state)

        assert len(result["character_reference_files"]) == 2
        assert "exactly this one character: Leo" in calls[0][1]["prompt"]
        assert "exactly this one character: Pip" in calls[1][1]["prompt"]
        assert calls[2][1]["reference_images"][0]["image"]
        assert calls[2][1]["reference_images"][1]["image"]
        with Image.open(result["character_board_file"]) as board:
            assert board.size == (1024, 1024)

    with_fake_images(run)


def test_visual_shot_uses_only_declared_cast_references(tmp_path):
    def run(calls):
        lion = tmp_path / "lion.png"
        mouse = tmp_path / "mouse.png"
        mood = tmp_path / "mood.png"
        lion.write_bytes(b"lion")
        mouse.write_bytes(b"mouse")
        mood.write_bytes(b"mood")
        scene = {
            **shot(1, "Lion rests alone in the meadow."),
            "characters_present": ["lion_001"],
            "narration": "Lion rests.",
            "subject_motion": "Lion breathes slowly.",
            "continuity_in": "Lion enters alone.",
            "continuity_out": "Lion remains alone.",
        }
        state = {
            "topic": "A lion rests alone",
            "tone": "quiet",
            "story": "Lion rests alone.",
            "production_bible": {
                "theme": "rest",
                "characters": [
                    {"character_id": "lion_001", "name": "Lion", "appearance": "golden lion", "wardrobe": "none", "props": []},
                    {"character_id": "mouse_001", "name": "Mouse", "appearance": "brown mouse", "wardrobe": "none", "props": []},
                ],
            },
            "character_reference_files": [str(lion), str(mouse)],
            "mood_board_file": str(mood),
            "output_dir": str(tmp_path),
            "storyboard": [scene],
        }
        image_agent.create_visual_storyboard(state)

        payload = calls[0][1]
        assert payload["reference_images"][0]["image"] == base64.b64encode(b"lion").decode("ascii")
        assert payload["reference_images"][1]["image"] == base64.b64encode(b"mood").decode("ascii")
        assert len(payload["reference_images"]) == 2
        assert payload["aspect_ratio"] == "9:16"
        assert payload["resolution"] == "1K"
        assert "Depict exactly 1 recurring character" in payload["prompt"]
        assert "Exact narration for this frame: Lion rests." in payload["prompt"]
        assert "mouse_001" not in payload["prompt"]

    with_fake_images(run)


def test_local_mode_skips_paid_video_and_warns(tmp_path):
    old = os.environ.pop("MINIO_PUBLIC_ENDPOINT", None)
    try:
        files, warnings = image_agent.create_scene_videos(
            {"topic": "Local", "output_dir": str(tmp_path), "storyboard": [shot(1, "Raja walks.")]},
            [str(tmp_path / "shot.png")],
        )
    finally:
        if old is not None:
            os.environ["MINIO_PUBLIC_ENDPOINT"] = old
    assert files == [None]
    assert "FFmpeg" in warnings[0]


def test_public_mode_uses_kling_26_image_url_and_supported_duration(tmp_path):
    original_request = image_agent._json_request
    original_headers = image_agent._magnific_headers
    original_public_url = image_agent.public_artifact_url
    original_video_check = image_agent._video_is_usable
    original_sleep = image_agent.time.sleep
    old_endpoint = os.environ.get("MINIO_PUBLIC_ENDPOINT")
    payloads = []

    def fake_request(_url, payload, _headers):
        payloads.append(payload)
        return {"data": {"generated": [base64.b64encode(b"fake mp4").decode("ascii")]}}

    os.environ["MINIO_PUBLIC_ENDPOINT"] = "https://media.example.com"
    image_agent._json_request = fake_request
    image_agent._magnific_headers = lambda: {}
    image_agent.public_artifact_url = lambda _state, _path: "https://media.example.com/shot.png"
    image_agent._video_is_usable = lambda _path: True
    image_agent.time.sleep = lambda _seconds: None
    scene = shot(1, "Raja walks.")
    scene["duration_seconds"] = 7
    try:
        files, warnings = image_agent.create_scene_videos(
            {
                "thread_id": "thread", "topic": "Remote", "output_dir": str(tmp_path), "storyboard": [scene],
                "motion_plans": [{
                    "shot_id": "shot-001", "duration_seconds": 6,
                    "video_prompt": "Use only the approved gentle walking motion.",
                    "negative_prompt": "No face distortion or camera shake.",
                }],
            },
            [str(tmp_path / "shot.png")],
        )
    finally:
        image_agent._json_request = original_request
        image_agent._magnific_headers = original_headers
        image_agent.public_artifact_url = original_public_url
        image_agent._video_is_usable = original_video_check
        image_agent.time.sleep = original_sleep
        if old_endpoint is None:
            os.environ.pop("MINIO_PUBLIC_ENDPOINT", None)
        else:
            os.environ["MINIO_PUBLIC_ENDPOINT"] = old_endpoint

    assert warnings == []
    assert Path(files[0]).exists()
    assert payloads[0]["image"] == "https://media.example.com/shot.png"
    assert payloads[0]["duration"] == "5"
    assert payloads[0]["prompt"] == "Use only the approved gentle walking motion."
    assert payloads[0]["negative_prompt"] == "No face distortion or camera shake."
    assert payloads[0]["aspect_ratio"] == "social_story_9_16"
    assert payloads[0]["generate_audio"] is False


def test_improved_prompt_respects_both_magnific_limits():
    original_request = image_agent._json_request
    original_headers = image_agent._magnific_headers
    old_setting = os.environ.get("MAGNIFIC_IMPROVE_PROMPTS")
    sent = {}
    start, end = "CAST AND SCENE: ", " KEEP EXACT IDENTITIES AND ADD NO CHARACTERS"

    def fake_request(_url, payload, _headers):
        sent.update(payload)
        return {"data": {"generated": [start + "y" * 4000 + end]}}

    os.environ["MAGNIFIC_IMPROVE_PROMPTS"] = "true"
    image_agent._json_request = fake_request
    image_agent._magnific_headers = lambda: {}
    try:
        result = image_agent._improve_prompt(start + "x" * 4000 + end, "image")
    finally:
        image_agent._json_request = original_request
        image_agent._magnific_headers = original_headers
        if old_setting is None:
            os.environ.pop("MAGNIFIC_IMPROVE_PROMPTS", None)
        else:
            os.environ["MAGNIFIC_IMPROVE_PROMPTS"] = old_setting

    assert len(sent["prompt"]) == image_agent.IMPROVE_PROMPT_LIMIT
    assert len(result) == image_agent.GENERATION_PROMPT_LIMIT
    assert sent["prompt"].startswith(start) and sent["prompt"].endswith(end)
    assert result.startswith(start) and result.endswith(end)


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        test_reference_package_is_versioned_and_regenerated_together(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_regenerates_only_selected_shot(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_reference_package_generates_one_isolated_sheet_per_character(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_visual_shot_uses_only_declared_cast_references(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_local_mode_skips_paid_video_and_warns(Path(tmp))
    with TemporaryDirectory() as tmp:
        test_public_mode_uses_wan_27_image_url_and_director_duration(Path(tmp))
    test_improved_prompt_respects_both_magnific_limits()
