# Narration Voice Agent

## Purpose

Synthesizes the traceability-approved narration before scene planning and returns character-level timing from ElevenLabs.

## Inputs

Approved `narration_segments`, language/voice settings, and target duration.

## Outputs

`narration_file`, `narration_alignment_file`, `narration_alignment`,
`narration_segment_timings`, `actual_narration_seconds`, and initial `scene_timings`.
Scene Planning replaces `scene_timings` with timings for its final scene partition.

## Rules

- Read approved narration segments in order.
- Use provider alignment rather than invented timestamps.
- Request a narration-script rewrite before scene planning when speech exceeds the film duration.

## Implementation

[`narration_agent.py`](../../video_automation/agents/narration_agent.py)
