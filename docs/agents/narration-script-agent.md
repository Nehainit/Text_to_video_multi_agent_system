# Narration Script Agent

## Purpose

Converts the approved story idea and structure into the voice-over text that drives later planning.

## Inputs

Approved `story.idea`, `story.structure`, cached parsed requirements, and any runtime correction from narration synthesis.

## Outputs

- `narration_segments`: agent-written text completed by the backend with consecutive `segment_id` values and one parent-beat mapping per approved story beat.
- `narration_script`: the segment text joined for downstream agents.
- `estimated_narration_seconds`: a speaking-rate estimate; TTS determines the final timing.
- `needs_story_revision` and `story_revision_reason` when the approved plot cannot be narrated coherently.

## Rules

- Preserve the story's cast, causal arc, climax, and ending.
- Use a duration-aware word range while leaving room for music, pauses, and effects.
- Return one ordered segment per approved story beat; the backend assigns `parent_beat_ids`.
- Run an independent semantic review of every segment against only its declared parent beats.
- Reject invented objects, actions, causes, outcomes, emotions, dialogue, or sensory details and regenerate with segment-specific feedback.
- Do not invent new events.
- Return to the Story Agent and human story review when `needs_story_revision` is true.
- Run immediately after story approval, then send approved segments to ElevenLabs before scene planning.
- Record token/cost evaluation for every generation and traceability-review attempt.

## Implementation

[`preproduction_agents.py`](../../preproduction_agents.py) — `create_narration_script`

The traceability reviewer returns `{"approved": true, "issues": []}`. Rejected
issues identify the `segment_id`, unsupported phrase, and insufficient
`parent_beat_ids`; each generation and review call records its own evaluation.
