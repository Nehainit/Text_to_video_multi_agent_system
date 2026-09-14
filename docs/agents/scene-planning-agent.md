# Scene Planning Agent

## Purpose

Converts approved narration segments and actual ElevenLabs timing into continuous visual scenes without assuming one segment equals one scene.

## Inputs

Approved story, narration segments, narration timing, actual audio duration,
deterministic character definitions, and parsed user requirements.

## Outputs

`scenes` contains consecutive scene IDs, source segment IDs, provider-backed
start/end timing, backend-assigned location and character IDs, scene goal, visible
actions, emotion, and story purpose. `scene_analysis` is a compatibility view
used by the Visual Beat Agent.

## Controls

[`config/scene_planning.yml`](../../config/scene_planning.yml) contains the
system prompt, faithfulness-review prompt, validation switches, timing
tolerance, and two-retry policy.

Code validates complete segment coverage and order, supplied timing, no
overlap, audio-duration bounds, approved cast, supported locations, visible actions,
and absence of camera instructions. A separate LLM review then checks semantic
faithfulness without rewriting the plan.

After two failed retries, or when the planner returns `needs_revision`, the
graph pauses at `scene_plan_review`. The API endpoint
`POST /api/review-scene-plan` accepts `retry` or `revise_narration`.

## Implementation

[`preproduction_agents.py`](../../preproduction_agents.py) — `plan_scenes`
