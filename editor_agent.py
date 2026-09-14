import re
import subprocess
from pathlib import Path

from schema import AgentState

FFMPEG_TIMEOUT_SECONDS = 300
RESOLUTIONS = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}


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
        "scene_timings": len(timings),
    }
    if state.get("scene_video_files"):
        counts["scene_video_files"] = len(state["scene_video_files"])
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"Editor agent asset counts do not match: {counts}.")


def _scene_timings(state: AgentState) -> list[dict[str, float | int]]:
    if state.get("edit_timeline"):
        return [
            {
                "scene_number": index,
                "start_seconds": item["timeline_start"],
                "end_seconds": item["timeline_end"],
            }
            for index, item in enumerate(state["edit_timeline"], start=1)
        ]
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


def _resolution(state: AgentState) -> tuple[int, int]:
    if state.get("width") and state.get("height"):
        return int(state["width"]), int(state["height"])
    return RESOLUTIONS.get(str(state.get("aspect_ratio", "9:16")), RESOLUTIONS["9:16"])


def _motion_filter(motion: str, duration: float, width: int = 1080, height: int = 1920) -> str:
    frames = max(1, int(round(duration * 30)))
    base = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
    if motion == "zoom_in":
        return f"{base},zoompan=z='min(zoom+0.0006,1.05)':d={frames}:s={width}x{height}:fps=30,format=yuv420p"
    if motion == "zoom_out":
        return f"{base},zoompan=z='if(eq(on,0),1.05,max(zoom-0.0006,1.0))':d={frames}:s={width}x{height}:fps=30,format=yuv420p"
    if motion == "pan_left":
        return f"{base},zoompan=z=1.05:x='iw-(iw/zoom)-on*(iw-iw/zoom)/{frames}':y='(ih-ih/zoom)/2':d={frames}:s={width}x{height}:fps=30,format=yuv420p"
    if motion == "pan_right":
        return f"{base},zoompan=z=1.05:x='on*(iw-iw/zoom)/{frames}':y='(ih-ih/zoom)/2':d={frames}:s={width}x{height}:fps=30,format=yuv420p"
    if motion == "tilt_up":
        return f"{base},zoompan=z=1.05:x='(iw-iw/zoom)/2':y='ih-(ih/zoom)-on*(ih-ih/zoom)/{frames}':d={frames}:s={width}x{height}:fps=30,format=yuv420p"
    if motion == "tilt_down":
        return f"{base},zoompan=z=1.05:x='(iw-iw/zoom)/2':y='on*(ih-ih/zoom)/{frames}':d={frames}:s={width}x{height}:fps=30,format=yuv420p"
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


def _caption_image(text: str, path: Path, width: int, height: int) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    scale = width / 1080
    font = _font(max(24, int(58 * scale)))
    margin = max(24, int(54 * scale))
    padding = max(18, int(28 * scale))
    line_height = max(32, int(74 * scale))
    lines = _wrapped_text(draw, text, font, width - 2 * (margin + padding))
    box_height = line_height * len(lines) + 2 * padding
    y = int(height * 0.88) - box_height
    draw.rounded_rectangle((margin, y, width - margin, y + box_height), radius=padding, fill=(0, 0, 0, 170))
    for index, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=2)
        x = (width - (bbox[2] - bbox[0])) / 2
        draw.text((x, y + padding + index * line_height), line, font=font, fill="white", stroke_width=2, stroke_fill="black")
    image.save(path)


def _caption_overlays(state: AgentState, out: Path) -> list[tuple[Path, float, float]]:
    overlays = []
    width, height = _resolution(state)
    captions_out = out / "captions"
    captions_out.mkdir(parents=True, exist_ok=True)
    for index, subtitle in enumerate(state.get("subtitles", []), start=1):
        caption_file = captions_out / f"caption_{index:03}.png"
        _caption_image(str(subtitle["text"]), caption_file, width, height)
        overlays.append((caption_file, float(subtitle["start_seconds"]), float(subtitle["end_seconds"])))
    return overlays


def _run_ffmpeg(command: list[str]) -> None:
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"FFmpeg failed or timed out: {exc}") from exc
    if result.returncode:
        raise RuntimeError(result.stderr)


def _clip(
    image_file: str,
    duration: float,
    motion: str,
    clip_file: Path,
    width: int = 1080,
    height: int = 1920,
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
            _motion_filter(motion, duration, width, height),
            "-an",
            "-r",
            "30",
            str(clip_file),
        ]
    )


def _video_clip(video_file: str, duration: float, clip_file: Path, source_in: float = 0.0, width: int = 1080, height: int = 1920) -> None:
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(source_in),
            "-i",
            video_file,
            "-t",
            str(duration),
            "-vf",
            f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},fps=30,format=yuv420p",
            "-an",
            "-r",
            "30",
            str(clip_file),
        ]
    )


def _transition(scene: dict) -> tuple[str, float]:
    value = scene.get("transition_to_next") or {}
    kind = str(value.get("type", "cut"))
    seconds = float(value.get("duration_seconds", 0)) if kind == "dissolve" else 0.0
    return kind, seconds


def _join_clips(clips: list[Path], scenes: list[dict], durations: list[float], output: Path) -> None:
    inputs = [part for clip in clips for part in ("-i", str(clip))]
    filters = [f"[{index}:v]setpts=PTS-STARTPTS[v{index}]" for index in range(len(clips))]
    current = "[v0]"
    offset = durations[0]
    for index in range(1, len(clips)):
        kind, transition_seconds = _transition(scenes[index - 1])
        target = f"[joined{index}]"
        if kind == "dissolve" and transition_seconds:
            filters.append(
                f"{current}[v{index}]xfade=transition=fade:duration={transition_seconds}:offset={offset}{target}"
            )
        else:
            filters.append(f"{current}[v{index}]concat=n=2:v=1:a=0{target}")
        current = target
        offset += durations[index]
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            ";".join(filters),
            "-map",
            current,
            "-t",
            str(sum(durations)),
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
    )


def edit_video(state: AgentState) -> dict[str, str]:
    for key in ("storyboard", "image_files", "narration_file", "subtitles"):
        if not state.get(key):
            raise RuntimeError(f"Editor agent needs {key}.")

    out = _out_dir(state, "video")
    out.mkdir(parents=True, exist_ok=True)
    clips_out = out / "clips"
    clips_out.mkdir(parents=True, exist_ok=True)
    picture_track = out / "picture_track.mp4"
    mixed_audio_file = out / "mixed_audio.mp3"
    video_file = out / "final_reel.mp4"

    timings = _scene_timings(state)
    _validate_counts(state, timings)
    total_seconds = max(float(timing["end_seconds"]) for timing in timings)
    width, height = _resolution(state)
    approved_rough_cut = Path(state["rough_cut_file"]) if state.get("combined_video_judge_approved") and state.get("rough_cut_file") else None
    clips = []
    durations = []
    scene_video_files = state.get("scene_video_files") or []
    timeline = state.get("edit_timeline") or []
    if approved_rough_cut and approved_rough_cut.is_file():
        picture_track = approved_rough_cut
    else:
        for index, (scene, image_file, timing) in enumerate(zip(state["storyboard"], state["image_files"], timings)):
            start = float(timing["start_seconds"])
            end = float(timing["end_seconds"])
            duration = max(0.1, end - start)
            if index < len(timeline):
                scene = {
                    **scene,
                    "transition_to_next": {
                        "type": timeline[index].get("transition_out", "cut"),
                        "duration_seconds": 0,
                    },
                }
            _, transition_seconds = _transition(scene)
            render_duration = duration + transition_seconds
            clip_file = clips_out / f"{scene.get('shot_id', f'shot-{index + 1:03}')}.mp4"
            if index < len(scene_video_files) and scene_video_files[index]:
                source_in = float(timeline[index].get("source_in", 0)) if index < len(timeline) else 0.0
                _video_clip(scene_video_files[index], render_duration, clip_file, source_in, width, height)
            else:
                _clip(image_file, render_duration, str(scene.get("motion", "static")), clip_file, width, height)
            clips.append(clip_file)
            durations.append(duration)
        edit_scenes = [
            {
                **scene,
                "transition_to_next": {
                    "type": timeline[index].get("transition_out", "cut"),
                    "duration_seconds": 0,
                },
            } if index < len(timeline) else scene
            for index, scene in enumerate(state["storyboard"])
        ]
        _join_clips(clips, edit_scenes, durations, picture_track)

    mix_inputs = ["-i", state["narration_file"]]
    filter_parts = []
    labels = ["[0:a]"]
    audio_index = 1
    for scene, sfx_file, timing in zip(state["storyboard"], state.get("sfx_files") or [None] * len(timings), timings):
        if not sfx_file:
            continue
        start_ms = int(float(timing["start_seconds"]) * 1000)
        mix_inputs.extend(["-i", sfx_file])
        filter_parts.append(f"[{audio_index}:a]adelay={start_ms}|{start_ms},volume=0.35[sfx{audio_index}]")
        labels.append(f"[sfx{audio_index}]")
        audio_index += 1
    if state.get("music_file"):
        mix_inputs.extend(["-i", state["music_file"]])
        music_index = len(mix_inputs) // 2 - 1
        filter_parts.append(f"[{music_index}:a]volume=0.16[music]")
        labels.append("[music]")
    filter_parts.append(f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:dropout_transition=0[a]")

    _run_ffmpeg(
        ["ffmpeg", "-y", *mix_inputs, "-filter_complex", ";".join(filter_parts), "-map", "[a]", str(mixed_audio_file)]
    )
    overlays = _caption_overlays(state, out)
    final_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(picture_track),
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
