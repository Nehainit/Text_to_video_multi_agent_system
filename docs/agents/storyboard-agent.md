# Storyboard Agent

## Purpose

Generates one static visual frame per approved shot so the user can review the film before video generation.

## Inputs

Validated storyboard entries, exact narration segments, derived continuity context, per-character sheets, mood board, and optional per-shot human feedback.

## Outputs

Ordered `image_files` corresponding one-to-one with storyboard shots.

## Rules

- Depict the exact narrated moment for the current shot.
- Pass only the character references for characters visible in that shot.
- Preserve framing, continuity, identity, style, and safe margins.
- Regenerate only shots that received feedback.

## Implementation

[`image_agent.py`](../../video_automation/agents/image_agent.py) — `create_visual_storyboard`
