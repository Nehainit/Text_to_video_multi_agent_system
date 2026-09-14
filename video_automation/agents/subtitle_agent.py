import json
import re
from pathlib import Path

from video_automation.schema import AgentState


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "video"


def _out_dir(state: AgentState, *parts: str) -> Path:
    return Path(state.get("output_dir", "outputs")) / _slug(state["topic"]) / Path(*parts)


def _seconds(value: str) -> float:
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0.0


def _srt_time(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def _subtitle_text(scene: dict) -> str:
    text = str(scene.get("narration", "")).strip()
    if len(text) > 100:
        text = text[:97].rsplit(" ", 1)[0] + "..."
    return text


def _words_from_alignment(alignment: dict) -> list[dict[str, object]]:
    chars = alignment.get("characters") or []
    starts = alignment.get("character_start_times_seconds") or []
    ends = alignment.get("character_end_times_seconds") or []
    words = []
    text = ""
    start = None
    end = None

    for char, char_start, char_end in zip(chars, starts, ends):
        if str(char).isspace():
            if text:
                words.append({"text": text, "start_seconds": start, "end_seconds": end})
                text = ""
                start = None
            continue
        if start is None:
            start = float(char_start)
        text += str(char)
        end = float(char_end)

    if text:
        words.append({"text": text, "start_seconds": start, "end_seconds": end})
    return words


def _scene_number(seconds: float, scene_timings: list[dict]) -> int:
    for timing in scene_timings:
        if float(timing["start_seconds"]) <= seconds < float(timing["end_seconds"]):
            return int(timing["scene_number"])
    return int(scene_timings[-1]["scene_number"]) if scene_timings else 1


def _word_subtitles(alignment_file: str) -> list[dict[str, object]]:
    data = json.loads(Path(alignment_file).read_text(encoding="utf-8"))
    words = _words_from_alignment(data["alignment"])
    scene_timings = data.get("scene_timings", [])
    subtitles = []
    group = []

    for word in words:
        group.append(word)
        if len(group) == 6 or float(group[-1]["end_seconds"]) - float(group[0]["start_seconds"]) >= 2.5:
            subtitles.append(
                {
                    "scene_number": _scene_number(float(group[0]["start_seconds"]), scene_timings),
                    "start_seconds": float(group[0]["start_seconds"]),
                    "end_seconds": float(group[-1]["end_seconds"]),
                    "text": " ".join(str(item["text"]) for item in group),
                }
            )
            group = []

    if group:
        subtitles.append(
            {
                "scene_number": _scene_number(float(group[0]["start_seconds"]), scene_timings),
                "start_seconds": float(group[0]["start_seconds"]),
                "end_seconds": float(group[-1]["end_seconds"]),
                "text": " ".join(str(item["text"]) for item in group),
            }
        )
    return subtitles


def create_subtitles(state: AgentState) -> dict[str, object]:
    storyboard = state.get("storyboard")
    if not storyboard:
        raise RuntimeError("Subtitle agent needs storyboard scenes.")

    out = _out_dir(state, "subtitles")
    out.mkdir(parents=True, exist_ok=True)
    subtitle_file = out / "story.srt"

    if state.get("narration_alignment_file"):
        subtitles = _word_subtitles(state["narration_alignment_file"])
    else:
        subtitles = []
        for scene in storyboard:
            subtitles.append(
                {
                    "scene_number": scene["scene_number"],
                    "start_seconds": _seconds(str(scene.get("start_time", "00:00"))),
                    "end_seconds": _seconds(str(scene.get("end_time", "00:00"))),
                    "text": _subtitle_text(scene),
                }
            )

    blocks = [
        (
            f"{index}\n"
            f"{_srt_time(item['start_seconds'])} --> {_srt_time(item['end_seconds'])}\n"
            f"{item['text']}"
        )
        for index, item in enumerate(subtitles, start=1)
    ]
    subtitle_file.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return {"subtitles": subtitles, "subtitle_file": str(subtitle_file)}
