import json
import os
import re
from pathlib import Path
from typing import Any
from urllib import error, parse, request

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = lambda *args, **kwargs: None

from video_automation.schema import AgentState
from video_automation.models import load_model
from video_automation.prompts import sound_design_prompt
from video_automation.agents.story_agent import _duration_seconds


load_dotenv()
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_ELEVENLABS_SFX_MODEL = "eleven_text_to_sound_v2"
ELEVENLABS_SFX_URL = "https://api.elevenlabs.io/v1/sound-generation"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "video"


def _env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set {name} in your environment or .env file.")
    return value.strip()


def _out_dir(state: AgentState, *parts: str) -> Path:
    return Path(state.get("output_dir", "outputs")) / _slug(state["topic"]) / Path(*parts)


def _elevenlabs_headers() -> dict[str, str]:
    return {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": os.getenv("ELEVENLABS_API_KEY") or _env("elevnlas"),
    }


def _bytes_request(url: str, payload: dict[str, Any], headers: dict[str, str]) -> bytes:
    req = request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=180) as response:
            return response.read()
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed: HTTP {exc.code} {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"POST {url} failed: {exc.reason}") from exc


def _timestamp_seconds(value: str) -> float:
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0.0


def _scene_duration(scene: dict, fallback: float) -> float:
    start = _timestamp_seconds(str(scene.get("start_time", "0:00")))
    end = _timestamp_seconds(str(scene.get("end_time", "0:00")))
    return min(max(end - start or fallback, 0.5), 30)


def _timing_duration(state: AgentState, scene_number: int) -> float | None:
    for timing in state.get("scene_timings", []):
        if timing.get("scene_number") == scene_number:
            return max(0.5, float(timing["end_seconds"]) - float(timing["start_seconds"]))
    return None


def _sfx_prompt(scene: dict) -> str:
    text = scene.get("sfx_prompt")
    if text:
        return str(text).strip()
    return (
        f"Cinematic sound design for this scene: {scene.get('visuals', '')}. "
        f"Mood from narration: {scene.get('narration', '')}. "
        "No speech, no music, only atmospheric sound effects."
    ).strip()


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
    return json.loads(text)


def create_sound_design_plan(state: AgentState) -> dict[str, list[dict[str, Any]]]:
    storyboard = state.get("storyboard")
    if not storyboard:
        raise RuntimeError("Sound Design Agent needs storyboard scenes.")
    model = load_model("soundfx")
    data = _parse_json(model.invoke(sound_design_prompt(
        story=state.get("story", ""),
        storyboard=storyboard,
        duration_seconds=_duration_seconds(state["duration"]),
        scene_timings=state.get("scene_timings", []),
    )).content)
    plan = data.get("sound_design_plan")
    if not isinstance(plan, list) or len(plan) != len(storyboard):
        raise RuntimeError("Sound Design Agent must return one entry per storyboard shot.")
    expected = [str(scene["shot_id"]) for scene in storyboard]
    if [str(item.get("shot_id")) for item in plan] != expected:
        raise RuntimeError("Sound Design Agent changed storyboard shot order.")
    return {"sound_design_plan": plan}


def create_soundfx(state: AgentState) -> dict[str, object]:
    storyboard = state.get("storyboard")
    if not storyboard:
        raise RuntimeError("SoundFX agent needs storyboard scenes.")

    out = _out_dir(state, "audio")
    out.mkdir(parents=True, exist_ok=True)
    model_id = state.get("elevenlabs_sfx_model", DEFAULT_ELEVENLABS_SFX_MODEL)
    fallback_duration = _duration_seconds(state["duration"]) / max(len(storyboard), 1)
    sfx_files = []

    for index, scene in enumerate(storyboard):
        scene_number = scene["scene_number"]
        sfx_file = out / f"sfx_scene_{scene_number:02}.mp3"
        meta_file = out / f"sfx_scene_{scene_number:02}.json"
        design = next((item for item in state.get("sound_design_plan", []) if item.get("shot_id") == scene.get("shot_id")), {})
        cues = design.get("sound_effects", []) if isinstance(design, dict) else []
        prompt = "; ".join(str(cue.get("prompt", "")).strip() for cue in cues if isinstance(cue, dict) and cue.get("prompt")) or _sfx_prompt(scene)
        duration_seconds = _timing_duration(state, scene_number) or _scene_duration(scene, fallback_duration)
        current_transition = scene.get("transition_to_next") or {}
        previous_transition = storyboard[index - 1].get("transition_to_next") or {} if index else {}
        if current_transition.get("audio_bridge") == "l_cut":
            duration_seconds += float(current_transition.get("duration_seconds") or 0.5)
        if previous_transition.get("audio_bridge") == "j_cut":
            duration_seconds += float(previous_transition.get("duration_seconds") or 0.5)
        duration_seconds = min(duration_seconds, 30)
        meta = {"text": prompt, "duration_seconds": duration_seconds, "model_id": model_id}

        if not sfx_file.exists() or not meta_file.exists() or json.loads(meta_file.read_text(encoding="utf-8")) != meta:
            audio = _bytes_request(
                f"{ELEVENLABS_SFX_URL}?{parse.urlencode({'output_format': 'mp3_44100_128'})}",
                {
                    "text": prompt,
                    "duration_seconds": duration_seconds,
                    "prompt_influence": 0.3,
                    "model_id": model_id,
                },
                _elevenlabs_headers(),
            )
            sfx_file.write_bytes(audio)
            meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        sfx_files.append(str(sfx_file))

    result: dict[str, object] = {"sfx_files": sfx_files}
    if os.getenv("MUSIC_ENABLED", "false").lower() in {"1", "true", "yes"}:
        music_cues = [str(item.get("music_cue", "")).strip() for item in state.get("sound_design_plan", []) if item.get("music_cue")]
        if music_cues:
            music_file = out / "music.mp3"
            music_prompt = "Instrumental background music only, no speech, no sound effects. " + "; ".join(music_cues)
            music_meta = out / "music.json"
            meta = {"text": music_prompt, "duration_seconds": _duration_seconds(state["duration"]), "model_id": model_id}
            if not music_file.exists() or not music_meta.exists() or json.loads(music_meta.read_text(encoding="utf-8")) != meta:
                music_file.write_bytes(_bytes_request(
                    f"{ELEVENLABS_SFX_URL}?{parse.urlencode({'output_format': 'mp3_44100_128'})}",
                    {"text": music_prompt, "duration_seconds": meta["duration_seconds"], "prompt_influence": 0.3, "model_id": model_id},
                    _elevenlabs_headers(),
                ))
                music_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            result["music_file"] = str(music_file)
    return result
