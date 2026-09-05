import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib import error, request

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = lambda *args, **kwargs: None

from prompts import character_sheet_prompt, image_prompt, mood_board_prompt
from schema import AgentState


load_dotenv(Path(__file__).with_name(".env"), override=True)

MAGNIFIC_TEXT_TO_IMAGE_URL = "https://api.magnific.com/v1/ai/text-to-image"
MAGNIFIC_IMAGE_TO_VIDEO_URL = "https://api.magnific.com/v1/ai/image-to-video/wan-v2-2-580p"


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
                time.sleep(int(exc.headers.get("Retry-After", "10")))
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


def _generate_image(prompt: str, size: str, image_file: Path) -> None:
    response = _json_request(
        MAGNIFIC_TEXT_TO_IMAGE_URL,
        {
            "prompt": prompt,
            "negative_prompt": "text, watermark, logo, extra limbs, distorted face, blurry",
            "guidance_scale": 2,
            "num_images": 1,
            "image": {"size": size},
            "filter_nsfw": True,
        },
        _magnific_headers(),
    )
    _save_generated_image(_image_value(response, MAGNIFIC_TEXT_TO_IMAGE_URL), image_file)
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
    return (
        f"{scene['visuals']} "
        f"Camera movement: {scene['camera']}. "
        f"Motion direction: {scene.get('motion', 'static')}. "
        "Cinematic movement, natural character motion, atmospheric lighting, no text, no watermark."
    )


def create_character_sheets(state: AgentState) -> dict[str, list[str]]:
    characters = _characters(state["characters"])
    if not characters:
        raise RuntimeError("Image agent needs characters for reference sheets.")

    out = _out_dir(state, "characters")
    out.mkdir(parents=True, exist_ok=True)
    image_file = out / "characters_reference.png"
    if not image_file.exists():
        _generate_image(character_sheet_prompt(characters), "square_1_1", image_file)

    return {"character_reference_files": [str(image_file)]}


def create_scene_videos(state: AgentState, image_files: list[str]) -> list[str]:
    storyboard = state.get("storyboard")
    if not storyboard:
        raise RuntimeError("Video scene agent needs storyboard scenes.")

    out = _out_dir(state, "scene_videos")
    out.mkdir(parents=True, exist_ok=True)
    video_files = []
    for scene, image_file in zip(storyboard, image_files):
        scene_number = scene["scene_number"]
        video_file = out / f"scene_{scene_number:02}.mp4"
        meta_file = out / f"scene_{scene_number:02}.json"
        prompt = _scene_video_prompt(scene)
        meta = {"prompt": prompt, "image_file": str(Path(image_file).resolve()), "model": "wan-v2-2-580p"}
        if not video_file.exists() or not meta_file.exists() or json.loads(meta_file.read_text(encoding="utf-8")) != meta:
            response = _json_request(
                MAGNIFIC_IMAGE_TO_VIDEO_URL,
                {
                    "image": _image_as_base64(image_file),
                    "prompt": prompt,
                    "duration": "5",
                    "aspect_ratio": "social_story_9_16",
                    "seed": scene_number,
                },
                _magnific_headers(),
            )
            _save_generated_video(_video_value(response, MAGNIFIC_IMAGE_TO_VIDEO_URL), video_file)
            meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            time.sleep(1)
        video_files.append(str(video_file))
    return video_files


def _next_scene_image(out: Path, scene_number: int) -> Path:
    versions = [int(path.stem.rsplit("_v", 1)[1]) for path in out.glob(f"scene_{scene_number:02}_v*.png")]
    return out / f"scene_{scene_number:02}_v{max(versions, default=0) + 1:03}.png"


def create_visual_storyboard(state: AgentState) -> dict:
    storyboard = state.get("storyboard")
    if not storyboard:
        raise RuntimeError("Visual storyboard agent needs storyboard scenes.")
    reference_files = state.get("character_reference_files") or create_character_sheets(state)["character_reference_files"]

    mood_out = _out_dir(state, "mood_board")
    mood_out.mkdir(parents=True, exist_ok=True)
    mood_board_file = Path(state.get("mood_board_file", mood_out / "mood_board.png"))
    if not mood_board_file.exists():
        _generate_image(
            mood_board_prompt(
                story=state["story"],
                tone=state["tone"],
                characters=state["characters"],
                storyboard=storyboard,
            ),
            "square_1_1",
            mood_board_file,
        )

    out = _out_dir(state, "images")
    out.mkdir(parents=True, exist_ok=True)
    feedback = {int(item["scene_number"]): str(item["note"]).strip() for item in state.get("visual_feedback", [])}
    current_files = state.get("image_files", [])
    image_files = []
    for index, scene in enumerate(storyboard):
        scene_number = int(scene["scene_number"])
        current_file = Path(current_files[index]) if index < len(current_files) else None
        if current_file and current_file.is_file() and scene_number not in feedback:
            image_file = current_file
        else:
            image_file = _next_scene_image(out, scene_number)
            _generate_image(image_prompt(scene, feedback.get(scene_number, "")), "social_story_9_16", image_file)
        image_files.append(str(image_file))

    return {
        "mood_board_file": str(mood_board_file),
        "character_reference_files": reference_files,
        "image_files": image_files,
        "visual_approved": False,
        "visual_feedback": [],
        "last_visual_feedback": list(state.get("visual_feedback", [])),
    }


def create_scene_videos_node(state: AgentState) -> dict[str, list[str]]:
    try:
        scene_video_files = create_scene_videos(state, state["image_files"])
    except RuntimeError as exc:
        if "error consuming credits" not in str(exc).lower():
            raise
        # ponytail: use editor motion until the Magnific account can consume video credits.
        scene_video_files = []
    return {"scene_video_files": scene_video_files}


def create_scene_images(state: AgentState) -> dict:
    visuals = create_visual_storyboard(state)
    return {**visuals, **create_scene_videos_node({**state, **visuals})}
