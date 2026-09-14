# Image Prompt Builder

## Purpose

Converts each approved shot into one concise still-keyframe prompt and an explicit reference package.

## Inputs

Shot, character definitions derived from approved requirements, approved scene location, visual style, and the previous shot ID when applicable.

## Outputs

The agent returns only `image_prompt`. The backend attaches shot, scene, visual-beat, segment, character, location, and previous-shot mappings. Character files are generated and attached afterward.

Every request passes an independent Image Prompt Faithfulness Review against the approved shot, continuity context, character references, location, and visual style. Review issues are returned to the builder; the reviewer never rewrites the prompt.

## Rules

- Preserve the shot's action, focus, framing, static angle, composition, emotion, location, and style.
- Include only declared characters and require their approved reference IDs.
- Describe one frozen moment without animation, sequence, or camera movement.
- Retry invalid output twice and record token and cost evaluation for every call.

## Implementation

[`config/image_prompt_builder.yml`](../../config/image_prompt_builder.yml) stores the prompt and output contract. [`image_agent.py`](../../video_automation/agents/image_agent.py) implements `build_image_prompts` and passes its output to keyframe generation.
