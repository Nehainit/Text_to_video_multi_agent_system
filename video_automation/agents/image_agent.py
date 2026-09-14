import base64
import json
import math
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib import error, request

from PIL import Image, ImageOps

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = lambda *args, **kwargs: None

from video_automation.artifact_store import public_artifact_url
from video_automation.continuity_context import character_definitions, continuity_context
from video_automation.models import invoke_with_evaluation, load_model
from video_automation.prompts import (
    IMAGE_PROMPT_BUILDER_CONFIG,
    IMAGE_PROMPT_BUILDER_SYSTEM_PROMPT,
    IMAGE_PROMPT_REVIEW_SYSTEM_PROMPT,
    SHOT_IMAGE_GENERATION_CONFIG,
    SHOT_VIDEO_GENERATION_CONFIG,
    character_sheet_prompt,
    image_prompt,
    mood_board_prompt,
)
from video_automation.schema import AgentState
from video_automation.agents.story_agent import _parse_json


load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

MAGNIFIC_TEXT_TO_IMAGE_URL = "https://api.magnific.com/v1/ai/text-to-image"
MAGNIFIC_REFERENCE_IMAGE_URL = "https://api.magnific.com/v1/ai/text-to-image/nano-banana-pro-flash"
MAGNIFIC_IMPROVE_PROMPT_URL = "https://api.magnific.com/v1/ai/improve-prompt"
MAGNIFIC_IMAGE_TO_VIDEO_URL = "https://api.magnific.com/v1/ai/image-to-video/kling-v2-6-pro"
MAGNIFIC_IMAGE_TO_VIDEO_STATUS_URL = "https://api.magnific.com/v1/ai/image-to-video/kling-v2-6"
NANO_ASPECT_RATIOS = {"square_1_1": "1:1", "social_story_9_16": "9:16"}
DEFAULT_CHARACTER_SHEET_REFERENCE = Path(__file__).resolve().parents[2] / "assets" / "character_sheet_layout_reference.png"
IMPROVE_PROMPT_LIMIT = 2500
GENERATION_PROMPT_LIMIT = 3000
FFPROBE_TIMEOUT_SECONDS = 30


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "video"


def _env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set {name} in your environment or .env file.")
    return value.strip()


def _out_dir(state: AgentState, *parts: str) -> Path:
    return Path(state.get("output_dir", "outputs")) / _slug(state["topic"]) / Path(*parts)


def _characters(characters: str | list[str]) -> list[str]:
    if isinstance(characters, str):
        return [character.strip() for character in characters.split(",") if character.strip()]
    return characters


def _validate_image_prompt(value: object, shot: dict, location: dict, visual_style: str, context: dict) -> dict:
    if not isinstance(value, dict):
        raise RuntimeError("Image Prompt Builder must return a JSON object.")
    prompt = value.get("image_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError("Image Prompt Builder returned an empty image_prompt.")
    normalized = " ".join(prompt.casefold().split())
    required = [
        shot[key] for key in ("visual_action", "visual_focus", "framing", "camera_angle", "composition", "emotion")
    ] + [location.get("description", ""), visual_style]
    if any(" ".join(str(item).casefold().split()) not in normalized for item in required if str(item).strip()):
        raise RuntimeError("Image prompt changed or omitted approved shot, location, or visual-style details.")
    present = set(shot["characters_present"])
    if present and ("preserve" not in normalized or any(character_id.casefold() not in normalized for character_id in present)):
        raise RuntimeError("Image prompt must preserve every approved character reference ID.")
    absent_ids = {
        str(character.get("character_id", "")).strip().casefold()
        for character in context.get("characters", [])
        if isinstance(character, dict) and character.get("character_id") not in present
    }
    if any(character_id and re.search(rf"\b{re.escape(character_id)}\b", normalized) for character_id in absent_ids):
        raise RuntimeError("Image prompt introduced a character outside characters_present.")
    if re.search(
        r"\b(?:then|afterward|subsequently)\b|\bcamera\s+(?:pans|tilts|orbits|tracks|dollies|zooms)\b|"
        r"\banimat(?:e|ed|ing)\b|\banimation\s+of\b",
        normalized,
    ):
        raise RuntimeError("Image prompt must describe one still frame without sequence, animation, or camera movement.")
    return {"image_prompt": prompt.strip()}


def validate_image_prompt_request(approved_shot: dict, request: dict) -> list[str]:
    issues = []
    if request.get("shot_id") != approved_shot["shot_id"]:
        issues.append("shot_id mismatch")
    if request.get("scene_id") != approved_shot["scene_id"]:
        issues.append("scene_id mismatch")
    if request.get("visual_beat_ids") != approved_shot["visual_beat_ids"]:
        issues.append("visual_beat_ids mismatch")
    if request.get("source_segment_ids") != approved_shot["source_segment_ids"]:
        issues.append("source_segment_ids mismatch")
    if request.get("location_id") != approved_shot["location_id"]:
        issues.append(f"location mismatch: expected {approved_shot['location_id']}, got {request.get('location_id')}")
    if set(approved_shot["characters_present"]) != set(request.get("character_reference_ids") or []):
        issues.append("character references do not match approved shot")
    location_reference = request.get("location_reference")
    if not isinstance(location_reference, dict) or location_reference.get("location_id") != request.get("location_id"):
        issues.append("location_reference does not match location_id")
    return issues


def _image_prompt_review_issues(value: object) -> list[str]:
    if not isinstance(value, dict) or set(value) != {"approved", "issues"} or type(value["approved"]) is not bool:
        raise RuntimeError("Image Prompt Faithfulness Reviewer returned an invalid decision.")
    issues = value["issues"]
    if not isinstance(issues, list) or any(not isinstance(issue, str) or not issue.strip() for issue in issues):
        raise RuntimeError("Image prompt review issues must be nonempty strings.")
    if value["approved"] != (not issues):
        raise RuntimeError("Image prompt review approval and issues disagree.")
    return issues


def review_image_prompt_faithfulness(
    model,
    approved_shot: dict,
    image_prompt_request: dict,
    context: dict,
) -> tuple[dict, dict]:
    payload = {
        "approved_shot": approved_shot,
        "image_prompt_request": image_prompt_request,
        "continuity_context": context,
    }
    response, evaluation = invoke_with_evaluation(
        model,
        [
            {"role": "system", "content": IMAGE_PROMPT_REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        agent_name="image-prompt-faithfulness-review",
        purpose="review_image_prompt_faithfulness",
    )
    return _parse_json(response.content), evaluation


def build_image_prompts(state: AgentState) -> dict:
    if not IMAGE_PROMPT_BUILDER_CONFIG["enabled"]:
        raise RuntimeError("Image Prompt Builder is disabled in config/image_prompt_builder.yml.")
    shots = state.get("shot_plan") or []
    if not shots:
        raise RuntimeError("Image Prompt Builder needs an approved shot plan.")
    bible = continuity_context(state)
    characters = bible.get("characters", [])
    character_by_id = {
        character["character_id"]: character
        for character in characters
        if isinstance(character, dict) and character.get("character_id")
    }
    locations = {
        location["location_id"]: location for location in bible.get("locations", [])
        if isinstance(location, dict) and location.get("location_id")
    }
    visual_style = str(
        bible.get("visual_style")
        or state.get("parsed_requirements", {}).get("visual_style")
        or "cinematic storybook illustration"
    ).strip()
    evaluations = list(state.get("llm_evaluations", []))
    requests = []
    previous_shot = None
    for shot in shots:
        try:
            character_definitions = [character_by_id[character_id] for character_id in shot["characters_present"]]
            location = locations[shot["location_id"]]
        except KeyError as exc:
            raise RuntimeError(f"Image Prompt Builder is missing an approved reference: {exc.args[0]}") from exc
        previous_id = previous_shot["shot_id"] if previous_shot and previous_shot["scene_id"] == shot["scene_id"] else None
        prompt_parts = [
            "Single still keyframe.",
            shot["visual_action"],
            f"Visual focus: {shot['visual_focus']}.",
            f"Framing: {shot['framing']}.",
            f"Camera angle: {shot['camera_angle']}.",
            f"Composition: {shot['composition']}.",
            f"Emotion: {shot['emotion']}.",
            f"Location: {location.get('description', '')}.",
            f"Visual style: {visual_style}.",
            f"Continuity entering the shot: {shot.get('continuity_in', '')}.",
            f"Continuity leaving the shot: {shot.get('continuity_out', '')}.",
        ]
        if character_definitions:
            prompt_parts.append(
                "Preserve " + ", ".join(shot["characters_present"])
                + " exactly from the supplied approved character references: "
                + json.dumps(character_definitions, ensure_ascii=False) + "."
            )
        candidate = _validate_image_prompt(
            {"image_prompt": " ".join(part for part in prompt_parts if part)},
            shot,
            location,
            visual_style,
            bible,
        )
        requests.append({
            "shot_id": shot["shot_id"],
            "scene_id": shot["scene_id"],
            "visual_beat_ids": list(shot["visual_beat_ids"]),
            "source_segment_ids": list(shot["source_segment_ids"]),
            "image_prompt": candidate["image_prompt"],
            "character_reference_ids": list(shot["characters_present"]),
            "location_id": shot["location_id"],
            "location_reference": location,
            "previous_shot_id": previous_id,
        })
        previous_shot = shot
    return {"image_prompt_requests": requests, "llm_evaluations": evaluations}


def _magnific_headers() -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "x-magnific-api-key": _env("MAGNIFIC_API_KEY"),
    }


def _json_request(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    req = request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed: HTTP {exc.code} {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"POST {url} failed: {exc.reason}") from exc


def _get_json(url: str, headers: dict[str, str], retries: int = 8) -> dict[str, Any]:
    req = request.Request(url, headers=headers, method="GET")
    for attempt in range(retries):
        try:
            with request.urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 and attempt < retries - 1:
                time.sleep(min(60, max(0, int(exc.headers.get("Retry-After", "10")))))
                continue
            raise RuntimeError(f"GET {url} failed: HTTP {exc.code} {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"GET {url} failed: {exc.reason}") from exc


def _get_bytes(url: str) -> bytes:
    with request.urlopen(url, timeout=120) as response:
        return response.read()


def _save_generated_image(value: str, image_file: Path) -> None:
    if value.startswith(("http://", "https://")):
        image_file.write_bytes(_get_bytes(value))
    else:
        image_file.write_bytes(base64.b64decode(value.split(",", 1)[-1]))


def _save_generated_video(value: str, video_file: Path) -> None:
    if value.startswith(("http://", "https://")):
        video_file.write_bytes(_get_bytes(value))
    else:
        video_file.write_bytes(base64.b64decode(value.split(",", 1)[-1]))


def _image_as_base64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _character_sheet_reference_files() -> list[str]:
    path = Path(os.getenv("CHARACTER_SHEET_REFERENCE_IMAGE", str(DEFAULT_CHARACTER_SHEET_REFERENCE))).expanduser()
    return [str(path)] if path.is_file() else []


def _limit_prompt(prompt: str, limit: int) -> str:
    if len(prompt) <= limit:
        return prompt
    marker = "\n...[trimmed to API limit]...\n"
    head = int((limit - len(marker)) * 0.7)
    tail = limit - len(marker) - head
    return f"{prompt[:head]}{marker}{prompt[-tail:]}"


def _generate_image(prompt: str, size: str, image_file: Path) -> None:
    _generate_reference_image(prompt, size, image_file, [])


def _generate_reference_image(
    prompt: str,
    size: str,
    image_file: Path,
    reference_files: list[str],
    reference_text: str = "Approved identity and visual-style reference. Preserve the subject exactly.",
    reference_labels: list[str] | None = None,
    improve_prompt: bool = True,
    resolution: str = "1K",
) -> None:
    if improve_prompt:
        prompt = _improve_prompt(prompt, "image")
    payload = {
        "prompt": prompt,
        "aspect_ratio": NANO_ASPECT_RATIOS.get(size, size),
        "resolution": resolution,
        "use_google_search_tool": False,
    }
    if reference_files:
        if len(reference_files) > 3:
            raise RuntimeError("Magnific supports at most three reference images per generation.")
        payload["reference_images"] = []
        for index, path in enumerate(reference_files):
            label = reference_labels[index] if reference_labels and index < len(reference_labels) else "reference image"
            payload["reference_images"].append({
                "image": _image_as_base64(path),
                "text": f"{reference_text} This image is the reference for {label}.",
                "mime_type": "image/png",
            })
    response = _json_request(MAGNIFIC_REFERENCE_IMAGE_URL, payload, _magnific_headers())
    _save_generated_image(_image_value(response, MAGNIFIC_REFERENCE_IMAGE_URL), image_file)
    time.sleep(1)


def _wait_for_magnific_image(url: str, task_id: str) -> str:
    for _ in range(60):
        data = _get_json(f"{url}/{task_id}", _magnific_headers()).get("data", {})
        generated = data.get("generated") or []
        if generated:
            return generated[0]
        if data.get("status") == "FAILED":
            raise RuntimeError(f"Magnific task failed: {task_id}")
        time.sleep(5)
    raise RuntimeError(f"Timed out waiting for Magnific task: {task_id}")


def _image_value(response: dict[str, Any], url: str) -> str:
    data = response.get("data")
    if isinstance(data, list) and data:
        image = data[0].get("base64") or data[0].get("url")
        if image:
            return image
    if isinstance(data, dict):
        generated = data.get("generated") or []
        if generated:
            return generated[0]
        task_id = data.get("task_id")
        if task_id:
            return _wait_for_magnific_image(url, task_id)
    if response.get("task_id"):
        return _wait_for_magnific_image(url, response["task_id"])
    raise RuntimeError("Magnific did not return an image.")


def _wait_for_improved_prompt(task_id: str) -> str:
    for _ in range(60):
        data = _get_json(f"{MAGNIFIC_IMPROVE_PROMPT_URL}/{task_id}", _magnific_headers()).get("data", {})
        generated = data.get("generated") or []
        if generated and isinstance(generated[0], str):
            return _limit_prompt(generated[0], GENERATION_PROMPT_LIMIT)
        if data.get("status") == "FAILED":
            raise RuntimeError(f"Magnific prompt improvement failed: {task_id}")
        time.sleep(2)
    raise RuntimeError(f"Timed out improving Magnific prompt: {task_id}")


def _improve_prompt(prompt: str, generation_type: str) -> str:
    if os.getenv("MAGNIFIC_IMPROVE_PROMPTS", "false").lower() not in {"1", "true", "yes"}:
        return _limit_prompt(prompt, GENERATION_PROMPT_LIMIT)
    response = _json_request(
        MAGNIFIC_IMPROVE_PROMPT_URL,
        {"prompt": _limit_prompt(prompt, IMPROVE_PROMPT_LIMIT), "type": generation_type, "language": "en"},
        _magnific_headers(),
    )
    data = response.get("data", response)
    if isinstance(data, dict):
        generated = data.get("generated") or []
        if generated and isinstance(generated[0], str):
            return _limit_prompt(generated[0], GENERATION_PROMPT_LIMIT)
        task_id = data.get("task_id")
        if task_id:
            return _wait_for_improved_prompt(task_id)
    raise RuntimeError("Magnific did not return an improved prompt task.")


def _wait_for_magnific_video(url: str, task_id: str) -> str:
    for _ in range(120):
        data = _get_json(f"{url}/{task_id}", _magnific_headers()).get("data", {})
        generated = data.get("generated") or []
        if generated:
            return generated[0]
        if data.get("status") == "FAILED":
            raise RuntimeError(f"Magnific video task failed: {task_id}")
        time.sleep(10)
    raise RuntimeError(f"Timed out waiting for Magnific video task: {task_id}")


def _video_value(response: dict[str, Any], url: str) -> str:
    data = response.get("data")
    if isinstance(data, dict):
        generated = data.get("generated") or []
        if generated:
            return generated[0]
        task_id = data.get("task_id")
        if task_id:
            return _wait_for_magnific_video(url, task_id)
    if response.get("task_id"):
        return _wait_for_magnific_video(url, response["task_id"])
    raise RuntimeError("Magnific did not return a video.")


def _scene_video_prompt(scene: dict) -> str:
    zoom_rule = (
        "Use only a subtle zoom of at most 5%; retain safe headroom and never crop the subject."
        if scene.get("motion") in {"zoom_in", "zoom_out"}
        else "Do not zoom, crop, or reframe the image."
    )
    return (
        f"Animate this approved storyboard frame exactly as composed: {scene['visuals']} "
        f"Camera movement only: {scene['camera']}. "
        f"Subject motion and expression only: {scene.get('subject_motion', '')}. "
        f"Motion direction: {scene.get('motion', 'static')}. "
        "Preserve the exact characters, faces, species, wardrobe, props, location, lighting, framing, and composition. "
        "Do not add or remove characters or objects, do not redesign the scene, do not cut to another angle, and do not change the setting. "
        f"Create one continuous short shot with subtle natural movement, no text, no watermark. {zoom_rule}"
    )


def _generate_scene_video_file(
    state: AgentState,
    scene: dict,
    image_file: str,
    video_file: Path,
    prompt: str,
    start_image_url: str | None = None,
    negative_prompt: str = "watermark, text, distortion, blurry, extra limbs",
    cfg_scale: float = 0.5,
    aspect_ratio: str = "social_story_9_16",
    generate_audio: bool = False,
) -> str:
    duration = min((5, 10), key=lambda value: abs(value - float(scene.get("duration_seconds", 5))))
    meta_file = video_file.with_suffix(".json")
    meta = {
        "prompt": prompt,
        "image_file": str(Path(image_file).resolve()),
        "duration": duration,
        "model": "kling-v2-6-pro",
        "negative_prompt": negative_prompt,
        "cfg_scale": cfg_scale,
        "aspect_ratio": aspect_ratio,
        "generate_audio": generate_audio,
    }
    if not video_file.exists() or not meta_file.exists() or json.loads(meta_file.read_text(encoding="utf-8")) != meta or not _video_is_usable(video_file):
        response = _json_request(
            MAGNIFIC_IMAGE_TO_VIDEO_URL,
            {
                "image": start_image_url or public_artifact_url(state, image_file),
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "duration": str(duration),
                "cfg_scale": cfg_scale,
                "aspect_ratio": aspect_ratio,
                "generate_audio": generate_audio,
            },
            _magnific_headers(),
        )
        _save_generated_video(_video_value(response, MAGNIFIC_IMAGE_TO_VIDEO_STATUS_URL), video_file)
        if not _video_is_usable(video_file):
            raise RuntimeError("Magnific returned a corrupted or unreadable video.")
        meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        time.sleep(1)
    return str(video_file)


def _video_is_usable(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=FFPROBE_TIMEOUT_SECONDS,
        )
        return result.returncode == 0 and float(result.stdout.strip()) > 0
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return False


def _next_version(out: Path, stem: str) -> Path:
    versions = [int(path.stem.rsplit("_v", 1)[1]) for path in out.glob(f"{stem}_v*.png")]
    return out / f"{stem}_v{max(versions, default=0) + 1:03}.png"


def _character_specs(state: AgentState) -> list[str]:
    specs = []
    for character in character_definitions(state):
        if isinstance(character, dict):
            details = [character.get(key) for key in ("name", "appearance", "wardrobe", "props")]
            specs.append(" — ".join(str(value) for value in details if value))
    return specs or _characters(state.get("characters", [])) or ["The main visual subject described in the story"]


def _combine_character_sheets(files: list[Path], output: Path) -> None:
    columns = math.ceil(math.sqrt(len(files)))
    rows = math.ceil(len(files) / columns)
    cell = 1024 // max(columns, rows)
    board = Image.new("RGB", (1024, 1024), "white")
    for index, file in enumerate(files):
        with Image.open(file) as source:
            image = ImageOps.contain(source.convert("RGB"), (cell - 16, cell - 16))
        x = (index % columns) * cell + (cell - image.width) // 2
        y = (index // columns) * cell + (cell - image.height) // 2
        board.paste(image, (x, y))
    board.save(output)


def create_character_sheets(state: AgentState) -> dict[str, list[str]]:
    characters = _character_specs(state)
    if not characters and not state.get("story"):
        raise RuntimeError("Image agent needs characters or a story for reference sheets.")

    out = _out_dir(state, "characters")
    out.mkdir(parents=True, exist_ok=True)
    board_file = Path(state.get("character_board_file", out / "characters_reference_v001.png"))
    reference_files = []
    for index, character in enumerate(characters, start=1):
        character_file = out / f"character_{index:03}_{_slug(character)}.png"
        if not character_file.exists():
            _generate_reference_image(
                character_sheet_prompt([character], state.get("story", ""), continuity_context(state), topic=state.get("topic", "")),
                "square_1_1",
                character_file,
                _character_sheet_reference_files(),
                "Fixed five-view layout reference only; preserve this character's approved identity.",
            )
        reference_files.append(str(character_file))
    if not board_file.exists():
        _combine_character_sheets([Path(path) for path in reference_files], board_file)
    return {"character_board_file": str(board_file), "character_reference_files": reference_files}


def create_reference_package(state: AgentState) -> dict:
    feedback = str(state.get("reference_feedback", "")).strip()
    characters_out = _out_dir(state, "characters")
    mood_out = _out_dir(state, "mood_board")
    characters_out.mkdir(parents=True, exist_ok=True)
    mood_out.mkdir(parents=True, exist_ok=True)

    if feedback or not state.get("character_board_file"):
        character_board = _next_version(characters_out, "characters_reference")
        mood_board = _next_version(mood_out, "mood_board")
        character_files = []
        for index, character in enumerate(_character_specs(state), start=1):
            character_file = _next_version(characters_out, f"character_{index:03}_{_slug(character)}")
            _generate_reference_image(
                character_sheet_prompt(
                    [character],
                    state.get("story", ""),
                    {**continuity_context(state), "characters": [character]},
                    feedback,
                    topic=state.get("topic", ""),
                ),
                "square_1_1",
                character_file,
                _character_sheet_reference_files(),
                "Fixed five-view layout reference only; do not copy its person. Derive this character's identity from the approved story and character description.",
            )
            character_files.append(character_file)
        _combine_character_sheets(character_files, character_board)
        _generate_reference_image(
            mood_board_prompt(
                story=state["story"],
                tone=state["tone"],
                characters=state["characters"],
                production_bible=continuity_context(state),
            )
            + (f" Human feedback: {feedback}" if feedback else ""),
            "square_1_1",
            mood_board,
            [str(path) for path in character_files[:3]],
        )
    else:
        character_board = Path(state["character_board_file"])
        mood_board = Path(state["mood_board_file"])
        character_files = [Path(path) for path in state["character_reference_files"]]

    return {
        "character_board_file": str(character_board),
        "character_reference_files": [str(path) for path in character_files],
        "mood_board_file": str(mood_board),
        "reference_approved": False,
        "reference_feedback": "",
        "last_reference_feedback": feedback,
    }


def create_scene_videos(state: AgentState, image_files: list[str]) -> tuple[list[str | None], list[str]]:
    storyboard = state.get("storyboard")
    if not storyboard:
        raise RuntimeError("Video scene agent needs storyboard scenes.")

    out = _out_dir(state, "scene_videos")
    out.mkdir(parents=True, exist_ok=True)
    video_files: list[str | None] = []
    warnings = []
    plans = {item["shot_id"]: item for item in state.get("motion_plans", [])}
    if not os.getenv("MINIO_PUBLIC_ENDPOINT"):
        return [None] * len(storyboard), ["Public MinIO is not configured; every shot uses local FFmpeg image motion."]

    for index, (scene, image_file) in enumerate(zip(storyboard, image_files), start=1):
        shot_id = str(scene.get("shot_id", f"shot-{index:03}"))
        video_file = out / f"{shot_id}.mp4"
        plan = plans.get(shot_id)
        prompt = plan["video_prompt"] if plan else _improve_prompt(_scene_video_prompt(scene), "video")
        negative_prompt = plan["negative_prompt"] if plan else "watermark, text, distortion, blurry, extra limbs"
        scene = {**scene, "duration_seconds": plan["duration_seconds"]} if plan else scene
        try:
            video_files.append(_generate_scene_video_file(
                state,
                scene,
                image_file,
                video_file,
                prompt,
                negative_prompt=negative_prompt,
                cfg_scale=float(SHOT_VIDEO_GENERATION_CONFIG["cfg_scale"]),
                aspect_ratio=str(SHOT_VIDEO_GENERATION_CONFIG["aspect_ratios"][
                    state.get("aspect_ratio", SHOT_VIDEO_GENERATION_CONFIG["default_aspect_ratio"])
                ]),
                generate_audio=bool(SHOT_VIDEO_GENERATION_CONFIG["generate_audio"]),
            ))
        except Exception as exc:
            video_files.append(None)
            warnings.append(f"{shot_id} image-to-video failed; used FFmpeg movement instead: {exc}")
    return video_files, warnings


def _candidate(
    state: AgentState,
    scene: dict,
    image_file: str,
    out: Path,
    shot_id: str,
    candidate_number: int,
    seed: int,
    round_number: int,
    prompt: str,
    start_image_url: str,
    negative_prompt: str,
) -> tuple[dict[str, Any], str | None]:
    candidate_id = f"{shot_id}-c{candidate_number}"
    candidate = {
        "shot_id": shot_id,
        "candidate_id": candidate_id,
        "round": round_number,
        "prompt": prompt,
        "seed": seed,
        "negative_prompt": negative_prompt,
        "file": None,
    }
    error = ""
    max_attempts = int(SHOT_VIDEO_GENERATION_CONFIG["max_retries"]) + 1
    for attempt in range(1, max_attempts + 1):
        try:
            candidate["file"] = _generate_scene_video_file(
                state,
                scene,
                image_file,
                out / f"{shot_id}_candidate_{candidate_number:03}.mp4",
                prompt,
                start_image_url,
                negative_prompt,
                float(SHOT_VIDEO_GENERATION_CONFIG["cfg_scale"]),
                str(SHOT_VIDEO_GENERATION_CONFIG["aspect_ratios"][
                    state.get("aspect_ratio", SHOT_VIDEO_GENERATION_CONFIG["default_aspect_ratio"])
                ]),
                bool(SHOT_VIDEO_GENERATION_CONFIG["generate_audio"]),
            )
            candidate.update(generation_status="success", generation_attempt=attempt, error=None)
            return candidate, None
        except Exception as exc:
            error = str(exc)
    candidate.update(generation_status="failed", generation_attempt=max_attempts, error=error)
    return candidate, f"{candidate_id} generation failed after {max_attempts} attempts: {error}"


def create_scene_video_candidates(state: AgentState) -> dict:
    storyboard = state.get("storyboard") or []
    image_files = state.get("image_files") or []
    if not storyboard:
        raise RuntimeError("Video scene agent needs storyboard scenes.")

    existing = list(state.get("video_candidates", []))
    retry_shots = set(state.get("video_retry_shots", []))
    retrying = bool(existing and retry_shots)
    round_number = state.get("video_refinement_round", 0) + int(retrying)
    warnings = []
    if not os.getenv("MINIO_PUBLIC_ENDPOINT"):
        return {
            "video_candidates": existing,
            "video_refinement_round": round_number,
            "video_retry_shots": [],
            "scene_video_files": state.get("scene_video_files", [None] * len(storyboard)),
            "warnings": [
                *state.get("warnings", []),
                "Public MinIO is not configured; every shot uses local FFmpeg image motion.",
            ],
        }

    out = _out_dir(state, "scene_videos")
    out.mkdir(parents=True, exist_ok=True)
    generated = []
    revised_prompts = state.get("video_revised_prompts", {})
    plans = {item["shot_id"]: item for item in state.get("motion_plans", [])}
    for index, scene in enumerate(storyboard, start=1):
        shot_id = str(scene.get("shot_id", f"shot-{index:03}"))
        if retrying and shot_id not in retry_shots:
            continue
        if index > len(image_files):
            warnings.append(f"{shot_id} has no storyboard image; video generation was skipped.")
            continue
        image_file = image_files[index - 1]
        candidate_numbers = [3] if retrying else [1, 2]
        try:
            plan = plans.get(shot_id)
            if plan:
                prompt = _limit_prompt(plan["video_prompt"], GENERATION_PROMPT_LIMIT)
            elif retrying:
                prompt = _limit_prompt(revised_prompts[shot_id], GENERATION_PROMPT_LIMIT)
            else:
                prompt = _improve_prompt(_scene_video_prompt(scene), "video")
            negative_prompt = plan["negative_prompt"] if plan else "watermark, text, distortion, blurry, extra limbs"
            scene = {**scene, "duration_seconds": plan["duration_seconds"]} if plan else scene
            start_image_url = public_artifact_url(state, image_file)
            if not start_image_url:
                raise RuntimeError("Storyboard image has no public URL.")
            jobs = [
                (
                    state, scene, image_file, out, shot_id, number,
                    index * int(SHOT_VIDEO_GENERATION_CONFIG["seed_base"]) + number,
                    round_number, prompt, start_image_url, negative_prompt,
                )
                for number in candidate_numbers
            ]
            if len(jobs) == 2:
                # ponytail: two workers cap provider pressure; widen only if measured latency requires it.
                with ThreadPoolExecutor(max_workers=2) as pool:
                    results = list(pool.map(lambda args: _candidate(*args), jobs))
            else:
                results = [_candidate(*jobs[0])]
            for candidate, warning in results:
                generated.append(candidate)
                if warning:
                    warnings.append(warning)
        except Exception as exc:
            warnings.append(f"{shot_id} candidate setup failed: {exc}")

    return {
        "video_candidates": [*existing, *generated],
        "video_refinement_round": round_number,
        "warnings": [*state.get("warnings", []), *warnings],
    }


def _shot_id(scene: dict, index: int) -> str:
    return str(scene.get("shot_id", f"shot-{index:03}"))


def _scene_character_indices(scene: dict, production_bible: dict) -> list[int]:
    characters = production_bible.get("characters", [])
    explicit = {str(value).casefold() for value in scene.get("characters_present", [])}
    if "characters_present" not in scene:
        description = " ".join(str(scene.get(key, "")) for key in ("visuals", "narration", "subject_motion", "continuity_in", "continuity_out"))
        explicit = {
            str(character.get("character_id", "")).casefold()
            for character in characters
            if isinstance(character, dict)
            and any(
                value and re.search(rf"\b{re.escape(value)}\b", description, re.IGNORECASE)
                for value in (str(character.get("character_id", "")), str(character.get("name", "")))
            )
        }
    return [
        index
        for index, character in enumerate(characters)
        if isinstance(character, dict)
        and explicit & {str(character.get("character_id", "")).casefold(), str(character.get("name", "")).casefold()}
    ]


def _image_worker_count(job_count: int) -> int:
    try:
        configured = int(os.getenv("IMAGE_GENERATION_WORKERS", "2"))
    except ValueError:
        configured = 2
    return min(job_count, max(1, min(4, configured)))


def _generate_shot_image_job(job: dict) -> dict:
    shot = job["shot"]
    shot_id = shot["shot_id"]
    image_file = job["image_file"]
    started = time.monotonic()
    worker = threading.current_thread().name
    prefix = f"[pipeline:{job['thread_id']}] [image-worker:{worker}]"
    print(f"{prefix} START {shot_id}", flush=True)
    error = ""
    attempt = 0
    for attempt in range(1, job["max_attempts"] + 1):
        try:
            _generate_reference_image(
                job["prompt"],
                job["aspect_ratio"],
                image_file,
                job["reference_files"],
                reference_labels=job["reference_labels"],
                improve_prompt=False,
                resolution=job["resolution"],
            )
            if not image_file.is_file() or image_file.stat().st_size == 0:
                raise RuntimeError("Provider returned an empty image asset.")
            error = ""
            break
        except Exception as exc:
            error = str(exc)
    success = not error
    print(
        f"{prefix} {'DONE' if success else 'ERROR'} {shot_id} elapsed={time.monotonic() - started:.1f}s",
        flush=True,
    )
    return {
        "shot_id": shot_id,
        "scene_id": shot["scene_id"],
        "image_path": str(image_file) if success else None,
        "image_url": None,
        "character_reference_ids": job["reference_ids"],
        "location_id": job["location_id"],
        "generation_status": "success" if success else "failed",
        "model_used": SHOT_IMAGE_GENERATION_CONFIG["model_used"],
        "aspect_ratio": job["aspect_ratio"],
        "resolution": job["resolution"],
        "generation_attempt": attempt,
        "error": None if success else error,
    }


def _generate_requested_shot_images(state: AgentState) -> dict:
    config = SHOT_IMAGE_GENERATION_CONFIG
    if not config["enabled"]:
        raise RuntimeError("Shot Image Generation Agent is disabled in config/shot_image_generation.yml.")
    shots = state.get("shot_plan") or []
    requests = state["image_prompt_requests"]
    if not shots or [item.get("shot_id") for item in requests] != [shot.get("shot_id") for shot in shots]:
        raise RuntimeError("Shot Image Generation Agent needs exactly one ordered request per approved shot.")

    out = _out_dir(state, "images")
    out.mkdir(parents=True, exist_ok=True)
    feedback_ids = {
        *state.get("shot_image_qa_retry_shots", []),
        *(str(item.get("shot_id")) for item in state.get("visual_feedback", [])),
    }
    current_files = state.get("image_files", [])
    approved_images = {
        shot["shot_id"]: current_files[index]
        for index, shot in enumerate(shots)
        if index < len(current_files) and current_files[index] and Path(current_files[index]).is_file()
    }
    previous_metadata = {
        item["shot_id"]: item for item in state.get("generated_images", [])
        if isinstance(item, dict) and item.get("generation_status") == "success"
    }
    generated_by_shot = {}
    pending, pending_qa = [], []
    max_attempts = int(config["max_retries"]) + 1
    max_references = int(config["max_reference_images"])
    aspect_ratio = str(state.get("aspect_ratio", config["default_aspect_ratio"]))
    resolution = str(config["quality_resolutions"][state.get("video_quality", "standard")])
    character_files = state.get("character_reference_files", [])
    reference_by_id = {
        character["character_id"]: character_files[index]
        for index, character in enumerate(character_definitions(state))
        if isinstance(character, dict) and character.get("character_id") and index < len(character_files)
    }
    qa_corrections = {
        item["shot_id"]: " ".join(
            issue.get("correction", "") for issue in item.get("issues", []) if isinstance(issue, dict)
        ).strip()
        for item in state.get("shot_image_qa_results", [])
        if isinstance(item, dict) and item.get("shot_id")
    }
    for shot, prompt_request in zip(shots, requests):
        shot_id = shot["shot_id"]
        current_file = approved_images.get(shot_id)
        if current_file and shot_id not in feedback_ids:
            metadata = previous_metadata.get(shot_id) or {
                "shot_id": shot_id,
                "scene_id": shot["scene_id"],
                "image_path": current_file,
                "image_url": None,
                "character_reference_ids": list(prompt_request.get("character_reference_ids") or []),
                "location_id": prompt_request.get("location_id"),
                "generation_status": "success",
                "model_used": config["model_used"],
                "generation_attempt": 0,
                "error": None,
            }
            generated_by_shot[shot_id] = metadata
            continue
        approved_images.pop(shot_id, None)
        pending.append((shot, prompt_request))
        pending_qa.append(shot_id)

    if pending:
        workers = _image_worker_count(len(pending))
        print(f"[pipeline:{state.get('thread_id', '-')}] IMAGE WORKERS count={workers} jobs={len(pending)}", flush=True)
        attempted = set(generated_by_shot)
        all_shot_ids = {shot["shot_id"] for shot in shots}
        remaining = pending
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="shot-image") as pool:
            while remaining:
                ready = [
                    item for item in remaining
                    if not item[1].get("previous_shot_id")
                    or item[1]["previous_shot_id"] in attempted
                    or item[1]["previous_shot_id"] not in all_shot_ids
                ]
                if not ready:
                    for shot, prompt_request in remaining:
                        shot_id = shot["shot_id"]
                        generated_by_shot[shot_id] = {
                            "shot_id": shot_id, "scene_id": shot["scene_id"], "image_path": None,
                            "image_url": None,
                            "character_reference_ids": list(prompt_request.get("character_reference_ids") or []),
                            "location_id": prompt_request.get("location_id"), "generation_status": "failed",
                            "model_used": config["model_used"], "aspect_ratio": aspect_ratio,
                            "resolution": resolution, "generation_attempt": 0,
                            "error": "previous_shot_id dependency cycle detected",
                        }
                    break

                jobs = []
                for shot, prompt_request in ready:
                    shot_id = shot["shot_id"]
                    issues = validate_image_prompt_request(shot, prompt_request)
                    reference_ids = list(prompt_request.get("character_reference_ids") or [])
                    reference_files = list(prompt_request.get("character_reference_files") or [
                        reference_by_id[character_id] for character_id in reference_ids if character_id in reference_by_id
                    ])
                    if len(reference_files) != len(reference_ids):
                        issues.append("character reference files do not match character_reference_ids")
                    missing = [path for path in reference_files if not Path(path).is_file()]
                    if missing:
                        issues.append(f"missing character reference file: {missing[0]}")
                    location_reference = prompt_request.get("location_reference") or {}
                    location_file = location_reference.get("reference_file") if isinstance(location_reference, dict) else None
                    if location_file and Path(location_file).is_file() and len(reference_files) < max_references:
                        reference_files.append(location_file)
                    previous_file = approved_images.get(prompt_request.get("previous_shot_id"))
                    if previous_file and len(reference_files) < max_references:
                        reference_files.append(previous_file)
                    if len(reference_ids) > max_references:
                        issues.append(f"image model supports at most {max_references} character reference files")
                    image_file = _next_version(out, shot_id)
                    if issues:
                        generated_by_shot[shot_id] = {
                            "shot_id": shot_id, "scene_id": shot["scene_id"], "image_path": None,
                            "image_url": None, "character_reference_ids": reference_ids,
                            "location_id": prompt_request.get("location_id"), "generation_status": "failed",
                            "model_used": config["model_used"], "aspect_ratio": aspect_ratio,
                            "resolution": resolution, "generation_attempt": 0, "error": "; ".join(issues),
                        }
                        continue
                    generation_prompt = prompt_request["image_prompt"]
                    if qa_corrections.get(shot_id):
                        generation_prompt += f"\nRegeneration correction: {qa_corrections[shot_id]}"
                    jobs.append({
                        "thread_id": state.get("thread_id", "-"), "shot": shot,
                        "image_file": image_file, "prompt": generation_prompt,
                        "reference_ids": reference_ids, "reference_files": reference_files,
                        "reference_labels": [
                            *reference_ids,
                            *(["approved location"] if location_file in reference_files else []),
                            *(["previous approved shot"] if previous_file in reference_files else []),
                        ],
                        "location_id": prompt_request.get("location_id"), "aspect_ratio": aspect_ratio,
                        "resolution": resolution, "max_attempts": max_attempts,
                    })

                for metadata in pool.map(_generate_shot_image_job, jobs):
                    generated_by_shot[metadata["shot_id"]] = metadata
                for shot, _ in ready:
                    shot_id = shot["shot_id"]
                    attempted.add(shot_id)
                    metadata = generated_by_shot[shot_id]
                    if metadata["generation_status"] == "success":
                        approved_images[shot_id] = metadata["image_path"]
                ready_ids = {shot["shot_id"] for shot, _ in ready}
                remaining = [item for item in remaining if item[0]["shot_id"] not in ready_ids]

    generated_images = [generated_by_shot[shot["shot_id"]] for shot in shots]
    image_files = [item["image_path"] for item in generated_images]

    return {
        "generated_images": generated_images,
        "image_files": image_files,
        "visual_approved": False,
        "visual_feedback": [],
        "last_visual_feedback": list(state.get("visual_feedback", [])),
        "shot_image_qa_pending_shots": pending_qa,
    }


def create_visual_storyboard(state: AgentState) -> dict:
    if state.get("image_prompt_requests"):
        return _generate_requested_shot_images(state)
    storyboard = state.get("storyboard")
    if not storyboard:
        raise RuntimeError("Visual storyboard agent needs storyboard scenes.")
    if not state.get("character_reference_files") or not state.get("mood_board_file"):
        raise RuntimeError("Visual storyboard needs an approved character and mood board package.")

    out = _out_dir(state, "images")
    out.mkdir(parents=True, exist_ok=True)
    feedback = {
        str(item.get("shot_id") or f"shot-{int(item['scene_number']):03}"): str(item["note"]).strip()
        for item in state.get("visual_feedback", [])
    }
    current_files = state.get("image_files", [])
    requests_by_id = {
        item["shot_id"]: item for item in state.get("image_prompt_requests", [])
        if isinstance(item, dict) and item.get("shot_id")
    }
    approved_images = {
        _shot_id(scene, index): current_files[index - 1]
        for index, scene in enumerate(storyboard, start=1)
        if index <= len(current_files) and Path(current_files[index - 1]).is_file()
    }
    image_files = []
    for index, scene in enumerate(storyboard, start=1):
        shot_id = _shot_id(scene, index)
        character_indices = _scene_character_indices(scene, state["production_bible"])
        scene = {
            **scene,
            "characters_present": [
                state["production_bible"]["characters"][item]["character_id"] for item in character_indices
            ],
        }
        character_references = [
            state["character_reference_files"][item]
            for item in character_indices[:3]
            if item < len(state["character_reference_files"])
        ]
        reference_labels = [
            f"{state['production_bible']['characters'][item].get('name', '')} ({state['production_bible']['characters'][item].get('character_id', '')})"
            for item in character_indices[:3]
        ]
        prompt_request = requests_by_id.get(shot_id)
        previous_reference = approved_images.get(prompt_request.get("previous_shot_id")) if prompt_request else None
        if previous_reference and len(character_references) > 1:
            character_references = [state["character_board_file"]]
            reference_labels = ["the complete approved cast board"]
        if len(character_references) > 2:
            character_references = [state["character_board_file"]]
            reference_labels = ["the complete approved cast board"]
        if character_indices and not character_references:
            character_references = state["character_reference_files"][:1]
            reference_labels = ["the first approved character"]
        if previous_reference:
            character_references.append(previous_reference)
            reference_labels.append(f"approved continuity image for {prompt_request['previous_shot_id']}")
        current_file = Path(current_files[index - 1]) if index <= len(current_files) else None
        if current_file and current_file.is_file() and shot_id not in feedback:
            image_file = current_file
        else:
            image_file = _next_version(out, shot_id)
            prompt = (
                prompt_request["image_prompt"]
                if prompt_request
                else image_prompt(
                    scene,
                    state.get("production_bible"),
                    topic=state.get("topic", ""),
                    story=state.get("story", ""),
                )
            )
            if feedback.get(shot_id):
                prompt = f"MANDATORY HUMAN CORRECTION: {feedback[shot_id]}. {prompt}"
            _generate_reference_image(
                prompt,
                "social_story_9_16",
                image_file,
                [
                    *character_references,
                    state["mood_board_file"],
                ],
                reference_labels=[*reference_labels, "the approved mood board"],
            )
        image_files.append(str(image_file))

    return {
        "image_files": image_files,
        "visual_approved": False,
        "visual_feedback": [],
        "last_visual_feedback": list(state.get("visual_feedback", [])),
    }


def create_scene_videos_node(state: AgentState) -> dict:
    if state.get("quality_mode") == "refine":
        return create_scene_video_candidates(state)
    scene_video_files, warnings = create_scene_videos(state, state["image_files"])
    return {"scene_video_files": scene_video_files, "warnings": [*state.get("warnings", []), *warnings]}


def create_scene_images(state: AgentState) -> dict:
    visuals = create_visual_storyboard(state)
    return {**visuals, **create_scene_videos_node({**state, **visuals})}
