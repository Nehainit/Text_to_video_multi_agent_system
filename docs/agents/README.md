# Agent contracts

The current production flow is:

```text
Input Safety Control → Story Agent → story review → Narration Script Agent
→ Narration Traceability Review → Narration Voice Agent (ElevenLabs Flash v2.5)
→ Scene Planning Agent → Scene Faithfulness Review
→ Visual Planner Agent → Visual Faithfulness Review
→ Shot Planner Agent → Director/Critic Review → Image Prompt Builder → Image Prompt Faithfulness Review
→ Character Sheet Agent → Sound Design Agent → Shot Image Generation Agent ↔ Shot Image QA → storyboard review
→ Motion Planner Agent → Shot Video Generation Agent → Deterministic Video Validator
→ Gemini Primary Judge Panel → Adversarial Video Judge → Video Meta Judge → assembly or targeted retry
```

The deterministic validator rejects only technical failures. The Meta Judge accepts clips for final assembly or routes only failed shots to image generation, motion planning, or video generation according to the identified source.

The [Orchestrator Agent](orchestrator-agent.md) controls this order. Character sheets are created automatically and are reference inputs for the Storyboard Agent; they are not the storyboard itself.

- [Story Agent](story-agent.md)
- [Character Sheet Agent](character-sheet-agent.md)
- [Narration Script Agent](narration-script-agent.md)
- [Scene Planning Agent](scene-planning-agent.md)
- [Visual Planner Agent](visual-beat-agent.md)
- [Shot Planner Agent](shot-planner-agent.md)
- [Image Prompt Builder](image-prompt-builder-agent.md)
- [Shot Image Generation Agent](shot-image-generation-agent.md)
- [Shot Image QA Agent](shot-image-qa-agent.md)
- [Motion Planner Agent](motion-planner-agent.md)
- [Shot Video Generation Agent](shot-video-generation-agent.md)
- [Director/Critic Agent](director-critic-agent.md)
- [Narration Voice Agent](narration-voice-agent.md)
- [Sound Design Agent](sound-design-agent.md)
- [Storyboard Agent](storyboard-agent.md)
- [Deterministic Video Validator](video-validator.md)
- [Visual Fidelity Judge](visual-fidelity-judge.md)
- [Gemini Video Judge Panel](video-judge-panel.md)
- [SoundFX Agent](soundfx-agent.md)
- [Subtitle Agent](subtitle-agent.md)
- [Editor Agent](editor-agent.md)
