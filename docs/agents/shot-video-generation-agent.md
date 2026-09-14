# Shot Video Generation Agent

## Purpose

Executes one approved motion plan against its approved still image and saves the resulting image-to-video clip.

## Behavior

- Uses the approved source image, `video_prompt`, and `negative_prompt` unchanged.
- Uses only configured Kling 2.6 Pro generation settings and records any adjustment to its supported 5 or 10 second duration.
- Maps the requested `16:9`, `9:16`, or `1:1` aspect ratio to Kling's provider value. Kling 2.6 Pro has no selectable resolution field, so output quality is recorded while the Pro endpoint determines clip quality.
- Disables provider prompt extension for Motion Planner prompts.
- Maintains shot and scene traceability in `generated_videos`.
- Retries technical failures twice without changing creative inputs; failures remain isolated by shot.
- Leaves semantic and visual evaluation to the future Judge Panel.

## Implementation

[`config/shot_video_generation.yml`](../../config/shot_video_generation.yml) stores the execution contract and provider settings. [`shot_video_generation_agent.py`](../../video_automation/agents/shot_video_generation_agent.py) implements standard and refined generation metadata.
