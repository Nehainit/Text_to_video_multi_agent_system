# Shot Planner Agent

## Purpose

Converts approved visual beats into image-ready still-keyframe shots without generating media.

## Inputs

Approved story, narration and timing, scene plan, visual beats, derived character definitions, and user requirements.

## Outputs

`shot_plan` with scene, visual-beat, and narration traceability; cast, subject, action, focus, static framing and angle, composition, location, estimated duration, continuity, and purpose.

## Rules

- Return exactly one ordered set of creative shot fields per approved visual beat; the backend assigns shot, visual-beat, scene, segment, location, and timing mappings.
- Allow one or more shots per visual beat when a still frame cannot show the whole moment clearly.
- Cover every visual beat in order and preserve its supported segments, scene, cast, and location.
- Preserve each scene's narration duration and shot-to-shot continuity.
- Reject camera movement, image prompts, motion prompts, filler, and unsupported story details.
- Retry invalid output twice, then request human review to retry or revise the visual plan.

## Implementation

[`config/shot_planning.yml`](../../config/shot_planning.yml) stores the prompt and controls. [`preproduction_agents.py`](../../preproduction_agents.py) implements `plan_shots`.
