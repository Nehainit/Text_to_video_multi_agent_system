# Character Sheet Agent

## Purpose

Generates one separate identity reference sheet per recurring character and a mood-board reference package before storyboard creation.

## Inputs

Approved story, original request, character definitions from approved requirements, and fixed sheet-layout reference image.

## Outputs

`character_reference_files`, combined `character_board_file`, and `mood_board_file`.

## Rules

- One character per source sheet.
- Preserve species, anatomy, face, body, wardrobe, and props.
- Use the layout reference only for arrangement, never for identity.
- Continue automatically without human approval.

## Implementation

[`image_agent.py`](../../video_automation/agents/image_agent.py) — `create_reference_package`
