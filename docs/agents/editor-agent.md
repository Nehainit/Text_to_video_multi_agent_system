# Editor Agent

## Purpose

Builds the final film from approved shots, generated clips, narration, effects, music, transitions, and subtitles.

## Inputs

Storyboard, image/video clips, narration audio, scene timings, SFX tracks, optional music, and subtitles.

## Outputs

`mixed_audio_file` and final `video_file`.

## Rules

- Keep shot order and requested runtime.
- Use generated clips when available and controlled image motion as fallback.
- Keep fallback zoom at or below five percent.
- Mix narration above sound effects and music.

## Implementation

[`editor_agent.py`](../../video_automation/agents/editor_agent.py)
