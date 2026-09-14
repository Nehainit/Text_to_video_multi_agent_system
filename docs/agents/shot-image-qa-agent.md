# Shot Image QA Agent

## Purpose

Visually checks whether every generated keyframe matches its approved shot, image prompt, character references, location, and style and is suitable for animation.

## Behavior

- Returns only approval, animation readiness, and issues; the backend attaches `shot_id` and derives `retry_target`.
- Reviews the generated image and its reference images with a vision-capable model.
- Returns typed, actionable issues for identity, scale, action, framing, composition, location, emotion, style, artifacts, and unsupported additions.
- Regenerates only rejected shots, applying the reviewer's correction without replacing the approved prompt.
- Preserves passing shots and stops after two automatic regeneration rounds before the human storyboard checkpoint.
- Records token usage and cost for each visual review call.

## Implementation

[`config/shot_image_qa.yml`](../../config/shot_image_qa.yml) stores the reviewer contract. [`shot_image_qa_agent.py`](../../video_automation/agents/shot_image_qa_agent.py) implements visual review and strict output validation.
