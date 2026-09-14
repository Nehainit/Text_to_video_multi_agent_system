import base64
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
from video_automation.agents.story_agent import _duration_seconds


load_dotenv()
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
DEFAULT_ELEVENLABS_TTS_MODEL = "eleven_flash_v2_5"
ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech"


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
        "Accept": "application/json",
        "Content-Type": "application/json",
        "xi-api-key": os.getenv("ELEVENLABS_API_KEY") or _env("elevnlas"),
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


def _narration_items(state: AgentState) -> list[tuple[int | str, str]]:
    segments = state.get("narration_segments") or []
    if segments:
        return [(str(segment["segment_id"]), str(segment["text"]).strip()) for segment in segments]
    if state.get("narration_script"):
        return [(1, state["narration_script"])]
    storyboard = state.get("storyboard") or []
    items = [
        (int(scene.get("scene_number", index)), str(scene.get("narration", "")).strip())
        for index, scene in enumerate(storyboard, start=1)
        if scene.get("narration")
    ]
    if items:
        return items
    if state.get("story"):
        return [(1, state["story"])]
    raise RuntimeError("Narration agent needs story or storyboard narration.")


def _scene_timings(items: list[tuple[int | str, str]], alignment: dict[str, Any]) -> list[dict[str, float | int | str]]:
    starts = alignment.get("character_start_times_seconds") or []
    ends = alignment.get("character_end_times_seconds") or []
    timings = []
    offset = 0
    previous_end = 0.0

    for item_id, line in items:
        indexes = [i for i in range(offset, min(offset + len(line), len(starts))) if not line[i - offset].isspace()]
        if indexes:
            start = float(starts[indexes[0]])
            end = float(ends[indexes[-1]])
        else:
            start = previous_end
            end = previous_end + 0.5
        if end <= start:
            end = start + 0.5
        timings.append({
            "segment_id" if isinstance(item_id, str) else "scene_number": item_id,
            "start_seconds": start,
            "end_seconds": end,
        })
        previous_end = end
        offset += len(line) + 1

    return timings


def _narration_result(
    state: AgentState,
    narration_file: Path,
    alignment_file: Path,
    alignment: dict[str, Any],
    scene_timings: list[dict[str, float | int]],
) -> dict[str, object]:
    ends = alignment.get("character_end_times_seconds") or []
    spoken_seconds = float(ends[-1]) if ends else 0.0
    allowed_seconds = _duration_seconds(state["duration"])
    result = {
        "narration_file": str(narration_file),
        "narration_alignment_file": str(alignment_file),
        "narration_alignment": alignment,
        "narration_segment_timings": scene_timings,
        "scene_timings": scene_timings,
        "actual_narration_seconds": spoken_seconds,
    }
    if spoken_seconds > allowed_seconds:
        note = (
            f"Shorten this shot's narration while preserving its story meaning. "
            f"The combined voice-over is {spoken_seconds:.1f}s but the movie is {allowed_seconds}s."
        )
        return {
            **result,
            "narration_feedback": note,
        }
    return result


def create_narration(state: AgentState) -> dict[str, object]:
    out = _out_dir(state, "audio")
    out.mkdir(parents=True, exist_ok=True)
    narration_file = out / "narration.mp3"
    meta_file = out / "narration.json"
    alignment_file = out / "narration_alignment.json"

    voice_id = state.get("elevenlabs_voice_id", DEFAULT_ELEVENLABS_VOICE_ID)
    model_id = state.get("elevenlabs_tts_model") or os.getenv("ELEVENLABS_TTS_MODEL", DEFAULT_ELEVENLABS_TTS_MODEL)
    items = _narration_items(state)
    text = "\n".join(line for _, line in items)
    meta = {"text": text, "voice_id": voice_id, "model_id": model_id, "timestamps": True}

    if narration_file.exists() and meta_file.exists() and alignment_file.exists():
        if json.loads(meta_file.read_text(encoding="utf-8")) == meta:
            alignment_doc = json.loads(alignment_file.read_text(encoding="utf-8"))
            alignment = alignment_doc["alignment"]
            scene_timings = alignment_doc.get("narration_segment_timings") or alignment_doc.get("scene_timings") or _scene_timings(items, alignment)
            return _narration_result(state, narration_file, alignment_file, alignment, scene_timings)

    url = f"{ELEVENLABS_TTS_URL}/{voice_id}/with-timestamps?{parse.urlencode({'output_format': 'mp3_44100_128'})}"
    data = _json_request(url, {"text": text, "model_id": model_id}, _elevenlabs_headers())
    alignment = data.get("alignment") or data.get("normalized_alignment")
    if not data.get("audio_base64") or not alignment:
        raise RuntimeError("Narration agent needs ElevenLabs audio and alignment data.")

    scene_timings = _scene_timings(items, alignment)
    narration_file.write_bytes(base64.b64decode(data["audio_base64"]))
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    alignment_file.write_text(
        json.dumps({"text": text, "alignment": alignment, "narration_segment_timings": scene_timings}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return _narration_result(state, narration_file, alignment_file, alignment, scene_timings)
