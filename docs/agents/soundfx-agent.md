# SoundFX Agent

## Purpose

Generates the planned atmospheric and action sound for each shot and optional background music.

## Inputs

Storyboard, sound-design plan, narration timings, transition audio bridges, and ElevenLabs settings.

## Outputs

Ordered `sfx_files` and optional `music_file`.

## Rules

- Generate no speech in effects tracks.
- Match each effect duration to its narration/shot timing.
- Respect J-cuts and L-cuts.
- Generate music only when `MUSIC_ENABLED` is enabled.

## Implementation

[`soundfx_agent.py`](../../video_automation/agents/soundfx_agent.py) — `create_soundfx`
