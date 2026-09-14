# Visual Fidelity Judge

## Purpose

Uses Gemini to inspect each complete, deterministically validated shot video against its approved source image, relevant character references, approved shot, and optional location/style references.

## Output

`visual_fidelity_reports` is keyed by `shot_id`. Each value contains the exact `visual_fidelity` judge report: ten integer scores from 1 through 10 and timestamped evidence for every score below 8.

## Boundaries

This judge evaluates only visual fidelity. It does not evaluate story quality, creative camera choices, motion intent, or rewrite prompts. Its report continues to the adversarial and meta judges with the other primary reports.

## Implementation

[`config/judge_panel.yml`](../../config/judge_panel.yml) stores the Gemini model and judge prompt. [`visual_fidelity_judge.py`](../../video_automation/agents/visual_fidelity_judge.py) assembles and validates this judge's report; [`gemini_judge.py`](../../video_automation/agents/gemini_judge.py) handles shared full-video upload and schema-constrained Gemini invocation.
