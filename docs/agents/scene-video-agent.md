# Legacy Scene Video Helper

## Purpose

Animates each approved storyboard frame as an independent short video clip.

## Inputs

Storyboard image, shot duration, camera instruction, subject movement, and public artifact URL.

## Outputs

This helper remains for compatibility. The production graph now uses the Shot Video Generation Agent and returns ordered `scene_video_files` plus `generated_videos` metadata.

## Rules

- Preserve the source frame, cast, wardrobe, location, lighting, and composition.
- Add movement and expression only; do not invent another scene.
- Do not zoom unless requested, and limit requested zoom to five percent.
- Fall back to editor-generated image motion when the API fails.

## Implementation

[`image_agent.py`](../../video_automation/agents/image_agent.py) — `create_scene_videos_node`
