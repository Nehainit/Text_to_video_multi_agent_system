# Motion Planner Agent

## Purpose

Converts each approved, animation-ready still image into the motion and negative prompts used for image-to-video generation.

## Behavior

- Plans one primary subject motion and optional restrained camera and environment motion.
- Preserves character identity, anatomy, scale, composition, framing, location, and story meaning.
- Uses the approved shot duration and matching narration timing.
- Is not revisited for deterministic video-validation failures.
- Returns creative motion fields only; the backend attaches the approved `shot_id` and duration.
- Sends an unsafe-to-animate result back to shot-plan review.
- Records token usage and cost for every LLM call.

## Implementation

[`config/motion_planning.yml`](../../config/motion_planning.yml) stores the prompt and retry controls. [`motion_planner_agent.py`](../../motion_planner_agent.py) validates and returns ordered `motion_plans`.
