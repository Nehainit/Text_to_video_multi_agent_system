# Orchestrator Agent

## Purpose

Runs the LangGraph workflow, persists thread state, routes revisions, and pauses only at configured human-review gates.

## Inputs

Initial topic, duration, language, output directory, and graph checkpoint configuration.

## Outputs

A completed shared `AgentState` containing the story, references, plans, media artifacts, QA evidence, and final video.

## Rules

- Run agents in dependency order.
- Resume the same thread after human feedback.
- Character references are generated automatically.
- Pause for story review and visual-storyboard review.

## Implementation

[`agent_graph.py`](../../video_automation/agent_graph.py)
