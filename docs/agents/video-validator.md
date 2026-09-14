# Deterministic Video Validator

## Purpose

Rejects generated clips that are technically unsuitable for expensive multimodal judging. It performs no semantic or visual-quality evaluation.

## Checks

Uses `ffprobe` for container, stream, duration, resolution, aspect ratio, FPS, frame count, codec, and audio metadata. Uses `ffmpeg` to decode the complete video stream and detect corruption.

Thresholds and retry limits live in [`config/video_validation.yml`](../../config/video_validation.yml). Missing optional audio passes unless `require_audio` is enabled. Expected width, height, and aspect ratio are checked only when supplied.

## Routing

Valid clips enter the Gemini primary Judge Panel. Invalid clips retry only Shot Video Generation, per shot, without changing the approved image or motion prompt. Exhausted failures terminate with `pipeline_status="video_validation_failed"`.

## Implementation

[`video_validator.py`](../../video_automation/agents/video_validator.py)
