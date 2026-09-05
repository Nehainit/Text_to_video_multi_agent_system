import re
import subprocess
from pathlib import Path

from schema import AgentState


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


def _scene_duration(scene: dict) -> float:
    return max(0.1, _seconds(str(scene["end_time"])) - _seconds(str(scene["start_time"])))


def _validate_counts(state: AgentState, timings: list[dict[str, float | int]]) -> None:
    counts = {
        "storyboard": len(state["storyboard"]),
        "image_files": len(state["image_files"]),
        "sfx_files": len(state["sfx_files"]),
        "scene_timings": len(timings),
    }
    if state.get("scene_video_files"):
        counts["scene_video_files"] = len(state["scene_video_files"])
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"Editor agent asset counts do not match: {counts}.")


def _scene_timings(state: AgentState) -> list[dict[str, float | int]]:
    timings = state.get("scene_timings")
    if timings:
        return timings
    return [
        {
            "scene_number": int(scene["scene_number"]),
            "start_seconds": _seconds(str(scene["start_time"])),
            "end_seconds": _seconds(str(scene["end_time"])),
        }
        for scene in state["storyboard"]
    ]


def _motion_filter(motion: str, duration: float) -> str:
    frames = max(1, int(round(duration * 30)))
    base = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
    if motion == "zoom_in":
        return f"{base},zoompan=z='min(zoom+0.002,1.18)':d={frames}:s=1080x1920:fps=30,format=yuv420p"
    if motion == "zoom_out":
        return f"{base},zoompan=z='if(eq(on,0),1.18,max(zoom-0.002,1.0))':d={frames}:s=1080x1920:fps=30,format=yuv420p"
    if motion == "pan_left":
        return f"{base},zoompan=z=1.12:x='iw-(iw/zoom)-on*(iw-iw/zoom)/{frames}':y='(ih-ih/zoom)/2':d={frames}:s=1080x1920:fps=30,format=yuv420p"
    if motion == "pan_right":
        return f"{base},zoompan=z=1.12:x='on*(iw-iw/zoom)/{frames}':y='(ih-ih/zoom)/2':d={frames}:s=1080x1920:fps=30,format=yuv420p"
    if motion == "tilt_up":
        return f"{base},zoompan=z=1.12:x='(iw-iw/zoom)/2':y='ih-(ih/zoom)-on*(ih-ih/zoom)/{frames}':d={frames}:s=1080x1920:fps=30,format=yuv420p"
    if motion == "tilt_down":
        return f"{base},zoompan=z=1.12:x='(iw-iw/zoom)/2':y='on*(ih-ih/zoom)/{frames}':d={frames}:s=1080x1920:fps=30,format=yuv420p"
    return f"{base},setsar=1,format=yuv420p"


def _font(size: int):
    from PIL import ImageFont

    for path in (
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _wrapped_text(draw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font, stroke_width=2)[2] <= max_width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines or [text]


def _caption_image(text: str, path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = _font(58)
    lines = _wrapped_text(draw, text, font, 920)
    line_height = 74
    box_height = line_height * len(lines) + 56
    y = 1620 - box_height
    draw.rounded_rectangle((54, y, 1026, y + box_height), radius=28, fill=(0, 0, 0, 170))
    for index, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=2)
        x = (1080 - (bbox[2] - bbox[0])) / 2
        draw.text((x, y + 28 + index * line_height), line, font=font, fill="white", stroke_width=2, stroke_fill="black")
    image.save(path)


def _caption_overlays(state: AgentState, out: Path) -> list[tuple[Path, float, float]]:
    overlays = []
    captions_out = out / "captions"
    captions_out.mkdir(parents=True, exist_ok=True)
    for index, subtitle in enumerate(state.get("subtitles", []), start=1):
        caption_file = captions_out / f"caption_{index:03}.png"
        _caption_image(str(subtitle["text"]), caption_file)
        overlays.append((caption_file, float(subtitle["start_seconds"]), float(subtitle["end_seconds"])))
    return overlays


def _run_ffmpeg(command: list[str]) -> None:
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)


def _clip(
    image_file: str,
    duration: float,
    motion: str,
    clip_file: Path,
) -> None:
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            image_file,
            "-t",
            str(duration),
            "-vf",
            _motion_filter(motion, duration),
            "-an",
            "-r",
            "30",
            str(clip_file),
        ]
    )


def _video_clip(video_file: str, duration: float, clip_file: Path) -> None:
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-stream_loop",
            "-1",
            "-i",
            video_file,
            "-t",
            str(duration),
            "-vf",
            "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30,format=yuv420p",
            "-an",
            "-r",
            "30",
            str(clip_file),
        ]
    )


def edit_video(state: AgentState) -> dict[str, str]:
    for key in ("storyboard", "image_files", "narration_file", "sfx_files", "subtitles"):
        if not state.get(key):
            raise RuntimeError(f"Editor agent needs {key}.")

    out = _out_dir(state, "video")
    out.mkdir(parents=True, exist_ok=True)
    clips_out = out / "clips"
    clips_out.mkdir(parents=True, exist_ok=True)
    concat_file = out / "images.txt"
    mixed_audio_file = out / "mixed_audio.mp3"
    video_file = out / "final_reel.mp4"

    timings = _scene_timings(state)
    _validate_counts(state, timings)
    total_seconds = max(float(timing["end_seconds"]) for timing in timings)
    lines = []
    cursor = 0.0
    previous_image = None
    clip_index = 1
    scene_video_files = state.get("scene_video_files") or []
    for index, (scene, image_file, timing) in enumerate(zip(state["storyboard"], state["image_files"], timings)):
        start = float(timing["start_seconds"])
        end = float(timing["end_seconds"])
        if start > cursor:
            hold_file = clips_out / f"clip_{clip_index:03}_hold.mp4"
            _clip(previous_image or image_file, start - cursor, "static", hold_file)
            lines.append(f"file '{hold_file.resolve()}'")
            clip_index += 1
        duration = max(0.1, end - start)
        clip_file = clips_out / f"scene_{int(scene['scene_number']):02}.mp4"
        if scene_video_files:
            _video_clip(scene_video_files[index], duration, clip_file)
        else:
            _clip(image_file, duration, str(scene.get("motion", "static")), clip_file)
        lines.append(f"file '{clip_file.resolve()}'")
        cursor = end
        previous_image = image_file
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    mix_inputs = ["-i", state["narration_file"]]
    filter_parts = []
    labels = ["[0:a]"]
    for index, (sfx_file, timing) in enumerate(zip(state["sfx_files"], timings), start=1):
        start_ms = int(float(timing["start_seconds"]) * 1000)
        mix_inputs.extend(["-i", sfx_file])
        filter_parts.append(f"[{index}:a]adelay={start_ms}|{start_ms},volume=0.35[sfx{index}]")
        labels.append(f"[sfx{index}]")
    filter_parts.append(f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:dropout_transition=0[a]")

    _run_ffmpeg(
        ["ffmpeg", "-y", *mix_inputs, "-filter_complex", ";".join(filter_parts), "-map", "[a]", str(mixed_audio_file)]
    )
    overlays = _caption_overlays(state, out)
    final_command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-i",
        str(mixed_audio_file),
    ]
    for caption_file, _, _ in overlays:
        final_command.extend(["-loop", "1", "-i", str(caption_file)])

    if overlays:
        current = "[0:v]"
        filters = []
        for index, (_, start, end) in enumerate(overlays, start=2):
            output = f"[v{index}]"
            filters.append(f"{current}[{index}:v]overlay=0:0:enable='between(t,{start},{end})'{output}")
            current = output
        final_command.extend(
            [
                "-filter_complex",
                ";".join(filters),
                "-map",
                current,
                "-map",
                "1:a:0",
            ]
        )
    else:
        final_command.extend(["-map", "0:v:0", "-map", "1:a:0"])

    final_command.extend(
        [
            "-t",
            str(total_seconds),
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(video_file),
        ]
    )
    _run_ffmpeg(final_command)
    return {"mixed_audio_file": str(mixed_audio_file), "video_file": str(video_file)}
