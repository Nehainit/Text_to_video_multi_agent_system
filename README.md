# Magnific Video Automation Agent

**Status: Active Development**

Core pipeline is functional. Current work focuses on improving generation quality, video animation quality, evaluation, retry logic, and production readiness.

A LangGraph pipeline that turns a story prompt into a narrated short video. It plans the story and shots, generates reference art and scene media, validates each stage, assembles a rough cut, and produces the final edit.

## Demo

<p align="center">
  <a href="https://raw.githubusercontent.com/Nehainit/Text_to_video_multi_agent_system/main/demo/robot-story.mp4">
    <img src="demo/robot-story-poster.jpg" width="280" alt="Robot story video preview">
  </a>
  <br>
  <a href="https://raw.githubusercontent.com/Nehainit/Text_to_video_multi_agent_system/main/demo/robot-story.mp4"><strong>▶ Watch the 17-second 1080×1920 demo</strong></a>
</p>

## Engineering snapshot

| **189** automated tests | **24** graph stages | **4** judge retry routes | **2 + 3** default media workers |
|---:|---:|---:|---:|
| **3** aspect ratios | **4** LLM providers | **SQLite** checkpoint + resume | **Per-shot** regeneration |

## Architecture

```mermaid
flowchart TB
    U["Story prompt + controls"] --> API["FastAPI + browser UI"] --> G["LangGraph orchestrator"]
    CP[("SQLite checkpoints")] <-. "persist + resume" .-> G

    subgraph PRE["Pre-production agents"]
        direction LR
        S["Safety + Story"] --> HS{"Story HITL"}
        HS --> N["Narration script + voice"]
        N --> P["Scene + visual + shot planning"]
        P --> HP{"Planning HITL gates"}
        HP --> D["Director critique + image prompts"]
    end

    subgraph GEN["Media generation"]
        direction LR
        R["Character + mood references"] --> I["Shot images<br/>ThreadPool: 2 workers"]
        I --> IQ["Image QA"] --> HV{"Storyboard HITL"}
        HV --> M["Motion plans"] --> V["Shot videos<br/>ThreadPool: 3 workers"]
    end

    subgraph POST["Deterministic validation + assembly"]
        direction LR
        DV["Video validator"] --> T["Edit timeline"] --> RC["FFmpeg rough cut"]
        RC --> J["Combined video judge"] --> SUB["Subtitles"] --> F["Final edit"]
    end

    G --> S
    D --> R
    V --> DV

    IQ -. "failed shots" .-> I
    HV -. "selected regeneration" .-> I
    DV -. "technical retry" .-> V
    T -. "clip-duration retry" .-> V
    J -. "source image" .-> I
    J -. "motion plan" .-> M
    J -. "video generation" .-> V
    J -. "edit timeline" .-> T

    A[("Versioned artifacts<br/>JSON · PNG · MP3 · MP4")]
    N -.-> A
    R -.-> A
    I -.-> A
    V -.-> A
    RC -.-> A
    F -.-> A
    A -. "optional publishing" .-> O[("MinIO")]
```

Human review checkpoints and SQLite persistence allow interrupted runs to be reviewed or resumed. Detailed agent contracts are in [`docs/agents`](docs/agents/README.md).

## Interface

![Softframe text-to-video interface](demo/softframe-ui.png)

## Project structure

```text
video_automation/        Python application package
  agents/                Pipeline agents, judges, validators, and media assembly
  agent_graph.py         LangGraph orchestration
  backend.py             FastAPI entrypoint
  main.py                Interactive CLI entrypoint
  models.py              Model-provider adapters
  prompts.py             Prompt/config loader
  schema.py              Shared state and validation
tests/                   Automated tests
config/                  Prompt and runtime YAML configuration
docs/agents/             Agent contracts
frontend/                Browser UI
assets/                  Bundled reference assets
demo/                    Example generated video
```

## Technology stack

- **Orchestration:** LangGraph with human-review interrupts and SQLite checkpoints
- **API and UI:** FastAPI, Uvicorn, and a dependency-free HTML/CSS/JavaScript frontend
- **LLMs:** Ollama/LangChain Ollama, Hugging Face Inference, Gemini, or Groq
- **Media generation:** Magnific image/video APIs and ElevenLabs narration
- **Media processing:** FFmpeg, ffprobe, and Pillow
- **Storage:** local artifacts with optional MinIO publishing
- **Configuration:** YAML plus environment variables loaded by python-dotenv

## Parallel processing

The pipeline uses Python's standard-library `ThreadPoolExecutor` for the network-bound generation stages:

- **Shot images:** `IMAGE_GENERATION_WORKERS` defaults to 2 and is clamped to 1–4. Independent shots run concurrently; shots with `previous_shot_id` continuity dependencies wait for their predecessor.
- **Shot videos:** `VIDEO_GENERATION_WORKERS` defaults to 3 and is clamped to 1–4. Independent shot-video jobs run concurrently, while retries and FFmpeg fallback remain isolated per shot.

Both paths use `pool.map`, so results stay in shot order even though provider requests execute in parallel. The rest of the LangGraph workflow remains sequential because each stage consumes the previous stage's validated output.

## Requirements

- Python 3.10+
- FFmpeg and ffprobe
- Ollama with text and vision models
- Magnific API key for image/video generation
- ElevenLabs API key for narration
- MinIO only when externally reachable source media is required

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your API keys to `.env`, then install the default local models:

```bash
ollama pull qwen2.5:3b-instruct
ollama pull qwen2.5vl:3b
ollama serve
```

## Run the web app

In a second terminal:

```bash
source .venv/bin/activate
python -m video_automation.backend
```

Open [http://127.0.0.1:8001](http://127.0.0.1:8001), enter a story prompt, and start generation. Artifacts are written under `outputs/<thread-id>/`.

## Run the interactive CLI

```bash
source .venv/bin/activate
python -m video_automation.main
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

The suite currently contains 189 tests.
