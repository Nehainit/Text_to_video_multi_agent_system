# Gemini Video Judge Panel

## Flow

Deterministically valid clips receive independent Visual Fidelity, Motion and Temporal, and Context and Intent reports. The Adversarial Judge then inspects the clip and challenges only observable oversights. The Meta Judge receives only those four reports and makes the final per-shot decision.

## Routing

- `ACCEPT` continues to sound, subtitles, and final editing.
- `REFINE` routes source-image failures to `create_storyboard` and motion-plan failures to `plan_motion`.
- `REGENERATE` routes rendering failures to `create_scene_videos` without changing the approved image or prompt.

Retries are tracked per shot. Reports for unaffected shots are reused. Exhausted semantic retries terminate with `pipeline_status="judge_retry_exhausted"`.

## Implementation

[`config/judge_panel.yml`](../../config/judge_panel.yml) contains all five judge contracts and shared controls. [`video_judge_panel.py`](../../video_automation/agents/video_judge_panel.py) validates reports, reconciles retry state, and provides the LangGraph nodes.
