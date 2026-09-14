# Subtitle Agent

## Purpose

Produces readable SRT subtitles synchronized to narration.

## Inputs

Storyboard narration, narration alignment file, and scene timings.

## Outputs

Structured `subtitles` and `subtitle_file`.

## Rules

- Prefer word-level provider alignment.
- Fall back to storyboard timing only when alignment is unavailable.
- Preserve narration text and shot order.

## Implementation

[`subtitle_agent.py`](../../video_automation/agents/subtitle_agent.py)
