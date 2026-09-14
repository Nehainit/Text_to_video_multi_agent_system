# Visual Planner Agent

## Purpose

Converts approved scenes into the minimum ordered visual beats needed before shot planning.

## Inputs

Approved story, narration segments and timing, scene plan, approved characters, requested visual style, and parsed user requirements.

## Outputs

The agent returns creative beats grouped positionally by scene. The backend adds `visual_beat_id`, `scene_id`, `source_segment_ids`, and `parent_story_beat_ids`.

## Rules

- Create beats dynamically for meaningful visible changes, independent of narration boundaries.
- Preserve scene, narration-segment, and story-beat traceability and ordering.
- Cover every scene, required action, and narration segment without filler.
- Preserve approved cast, locations, props, identity, style, and continuity.
- Leave camera, shot, lens, clip, and video-prompt decisions to later agents.
- Retry schema, coverage, continuity, traceability, and faithfulness failures twice, then request human review.

The independent Visual Faithfulness Review checks semantic support after deterministic validation. Human review can retry the visual plan or send the work back to Scene Planning through `POST /api/review-visual-plan`.

## Implementation

[`config/visual_planning.yml`](../../config/visual_planning.yml) stores the prompt and controls. [`preproduction_agents.py`](../../preproduction_agents.py) implements `create_visual_beats`.
