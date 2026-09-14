# Shot Image Generation Agent

## Purpose

Generates exactly one still keyframe for each approved shot from its validated image prompt and reference package.

## Behavior

- Passes the approved `image_prompt` to Magnific unchanged.
- Resolves the request's character IDs to the newly generated character sheets, then attaches supported location and previous-shot references.
- Saves each image under its `shot_id` and returns traceable `generated_images` metadata.
- Retries each failed shot independently; successful shots remain available and are not regenerated.

## Output

Each result records the shot and scene IDs, image path or URL, character and location IDs, model, attempt count, status, and any final error. Ordered `image_files` remains available for downstream video generation.

## Implementation

[`config/shot_image_generation.yml`](../../config/shot_image_generation.yml) stores the system prompt and generation controls. [`image_agent.py`](../../video_automation/agents/image_agent.py) implements the agent in `_generate_requested_shot_images`.
