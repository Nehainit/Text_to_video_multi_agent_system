# Director/Critic Agent

## Purpose

Reviews and corrects the complete shot plan before expensive storyboard images are generated.

## Inputs

Narration script, visual beats, approved character definitions, requested visual style, duration, and proposed shot plan.

## Outputs

Validated `director_plan`, final textual `storyboard`, `director_approved`, and `critic_issues`.

## Rules

- Check narration-to-visual alignment, story completeness, cast, continuity, direction, framing, transitions, and runtime.
- Correct invalid shot details without changing shot count or narration order.
- Reject random or disconnected shots.

## Implementation

[`preproduction_agents.py`](../../video_automation/agents/preproduction_agents.py) — `critique_shots`
