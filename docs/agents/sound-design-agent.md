# Sound Design Agent

## Purpose

Plans ambience, music cues, and timed non-speech effects around narration and visible actions.

## Inputs

Story, storyboard, target duration, and actual narration timings.

## Outputs

One ordered `sound_design_plan` entry per storyboard shot.

## Rules

- Keep effects tied to visible actions and locations.
- Keep music beneath narration.
- Use silence when it serves the scene.
- Never change shot order.

## Implementation

[`soundfx_agent.py`](../../video_automation/agents/soundfx_agent.py) — `create_sound_design_plan`
