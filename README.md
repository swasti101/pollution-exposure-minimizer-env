---
title: Pollution Exposure Minimizer Environment
emoji: 🌫️
colorFrom: gray
colorTo: green
sdk: docker
pinned: false
app_port: 7680
base_path: /demo
tags:
  - openenv
  - reinforcement-learning
  - route-planning
  - air-quality
---

# Pollution Exposure Minimizer Environment

`pollution-exposure-minimizer-environment` is an OpenEnv benchmark for urban commuting under air-quality constraints. The environment uses a deterministic Delhi-inspired transport graph with pollution exposure on each segment, time-dependent traffic effects, and four tasks that get progressively harder from static walking to dynamic multimodal routing.

This is meant to model a real decision problem: choosing how to commute through a polluted city while trading off exposure, travel time, and route structure.

## Why this environment exists

Many agent benchmarks focus on text-only workflow tasks. This project covers a different real-world problem: sequential route planning under health risk. A strong agent should be able to:

- read a constrained transport observation
- pick legal movement actions
- reason about immediate and future exposure
- avoid wasting steps on loops or low-value waits
- finish the commute efficiently

## Environment design

The environment is built on a fixed city graph in [data/pollution_city_graph.json](./data/pollution_city_graph.json). Each node represents a Delhi-inspired zone and each edge represents a travel corridor.

Pollution dynamics come from [server/aqi.py](./server/aqi.py):

- node AQI depends on zone type and traffic sensitivity
- harder tasks add hour-of-day traffic spikes
- pollution diffuses across neighboring nodes
- exposure depends on node AQI, road type, distance, and transport mode

The core OpenEnv runtime lives in [server/pollution_exposure_minimizer_environment.py](./server/pollution_exposure_minimizer_environment.py).

## Action space

The agent emits a typed `PollutionAction` from [models.py](./models.py):

- `move`
  - `target_node_id`
  - `mode` in `walk`, `bus`, `metro`
- `wait`
  - `target_node_id=null`
  - `mode=null`

Only legal actions exposed in the observation may be used.

## Observation space

Each `PollutionObservation` includes:

- task metadata
- current node and destination
- current hour
- cumulative exposure
- cumulative travel time
- cumulative weighted cost
- remaining steps
- current-node AQI
- legal actions with estimated exposure and time
- full graph snapshot for plotting or agent-side planning

## Tasks

Task definitions live in [server/tasks.py](./server/tasks.py).

### `easy_static_route`

- Goal: walk from North Campus to Nehru Place
- Modes: `walk`
- Wait: not allowed
- AQI: static
- What makes it hard: exposure-aware path choice with no transport shortcuts

### `medium_multimodal_route`

- Goal: commute from Karol Bagh to Saket
- Modes: `walk`, `bus`, `metro`
- Wait: not allowed
- AQI: static
- What makes it hard: balancing speed and exposure across several southbound route choices

### `hard_dynamic_peak_route`

- Goal: travel from Civil Lines to Okhla Phase II during peak conditions
- Modes: `walk`, `bus`, `metro`
- Wait: allowed
- AQI: dynamic
- What makes it hard: peak-hour pollution, diffusion, and the risk of spending steps on low-value waits

### `bonus_dynamic_cross_city_route`

- Goal: travel from Karol Bagh to Okhla Phase II during peak conditions
- Modes: `walk`, `bus`, `metro`
- Wait: allowed
- AQI: dynamic
- What makes it hard: a longer cross-city commute with the same peak-hour tradeoffs and more room for route mistakes

## Reward design

Rewards are dense and shaped over the full trajectory.

- every move incurs exposure and time cost
- moving closer to the destination adds a small progress bonus
- reaching the destination gives an arrival bonus
- illegal or malformed moves are penalized
- waiting is legal only on the hard tasks

Final grading is deterministic and normalized to `[0.01, 0.99]`.

## Grader

The grader in [server/grader.py](./server/grader.py) computes:

- exposure component
- time component
- wait component
- failure penalty for unfinished episodes
- remaining-distance penalty when the destination is not reached

Final score:

```text
score = clamp((baseline_cost - agent_cost) / (baseline_cost - oracle_cost), 0.01, 0.99)
```

This means:

- `1.0` = oracle-level route
- around `0.5` = clearly better than the simple baseline but not optimal
- `0.0` = baseline-level or worse

## Baselines

Reference planners live in [server/baseline.py](./server/baseline.py):

- `baseline`: simple deterministic greedy planner
- `oracle`: best-cost planner under the task cost function

Current reference costs:

- `easy_static_route`: baseline `4183.68`, oracle `2008.50`
- `medium_multimodal_route`: baseline `3618.00`, oracle `893.13`
- `hard_dynamic_peak_route`: baseline `6086.32`, oracle `1755.62`
- `bonus_dynamic_cross_city_route`: baseline `5518.72`, oracle `1819.94`

These values can be re-generated with:

```powershell
.\.venv\Scripts\python.exe scripts\preview_tasks.py
```

## Inference script

The submission inference runner is [inference.py](./inference.py). It:

- uses the OpenAI client
- reads configuration from `.env`
- supports either a running local server or a Docker image
- logs exactly in `[START]`, `[STEP]`, `[END]` format
- uses a deterministic fallback policy only when the model output is invalid or the API call fails

The preferred model response format is:

```text
A1
```

The parser also accepts short text like `Answer: A1` and extracts the action id.

Latest verified router run with `openai/gpt-oss-120b:fastest`:

- `easy_static_route`: `0.278`
- `medium_multimodal_route`: `0.628`
- `hard_dynamic_peak_route`: `0.257`

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

If you want notebook support locally, install that separately.

## Configure `.env`

Copy [.env.example](./.env.example) to `.env` and fill in the provider you want to use.

Hugging Face router example:

```env
HF_TOKEN=your_hf_token_here
OPENAI_API_KEY=
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=openai/gpt-oss-120b:fastest
ENV_BASE_URL=http://localhost:7680
LOCAL_IMAGE_NAME=
TASK_LIST=easy_static_route,medium_multimodal_route,hard_dynamic_peak_route,bonus_dynamic_cross_city_route
MAX_STEPS=12
```

OpenAI direct example:

```env
HF_TOKEN=
OPENAI_API_KEY=your_openai_api_key_here
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
ENV_BASE_URL=http://localhost:7680
LOCAL_IMAGE_NAME=
TASK_LIST=easy_static_route,medium_multimodal_route,hard_dynamic_peak_route,bonus_dynamic_cross_city_route
MAX_STEPS=12
```

You can also set `API_BASE_URL=auto`. In that mode, [inference.py](./inference.py) automatically switches to `https://api.openai.com/v1` when `MODEL_NAME` starts with `gpt-`, `o1`, `o3`, or `o4` and `OPENAI_API_KEY` is present.

## Space UI

When deployed to Hugging Face Spaces:

- `/` and `/demo` serve a custom interactive map UI
- `/web` serves the built-in OpenEnv web interface when available
- `/tasks`, `/baseline`, `/grader`, `/reset`, `/step`, and `/state` remain available for programmatic use

The custom UI includes:

- a live graph view of the Delhi-inspired map
- node colors based on current AQI
- current position, destination, and route trail
- clickable legal actions
- a deterministic demo-agent autoplay mode
- raw observation/state JSON for debugging

## Using the Built-in Web UI

Open the standard OpenEnv playground at `/web`.

### Start an episode

1. Click `Reset`.
2. Use an empty reset payload to cycle to the next task:

```json
{}
```

Each reset without a task_id rotates through:

- easy_static_route
- medium_multimodal_route
- hard_dynamic_peak_route
- bonus_dynamic_cross_city_route
  To open a specific task directly, reset with:

  ```json
  { "task_id": "easy_static_route" }
  ```

  or

  ```json
  { "task_id": "medium_multimodal_route" }
  ```

  or

  ```json
  { "task_id": "hard_dynamic_peak_route" }
  ```

  or

  ```json
  { "task_id": "bonus_dynamic_cross_city_route" }
  ```

  ### Take a step

  Use only legal adjacent moves. The agent cannot jump directly to the final destination unless that destination is a legal next node.

  Example move payload:

  ```json
  {
    "action_type": "move",
    "target_node_id": "civil_lines",
    "mode": "walk"
  }
  ```

  Example wait payload for the hard task:

  ```json
  {
    "action_type": "wait",
    "target_node_id": null,
    "mode": null
  }
  ```

  ### Interpreting reward and done
  - Rewards are dense and are often negative during intermediate steps.
  - Negative reward is expected because exposure and travel time both add cost.
  - done stays false until:
    - the destination is reached, or
    - the task runs out of allowed steps.
  - Illegal actions are penalized but do not instantly end the episode.

  ### Helpful routes
  - / or /demo: custom visual map UI
  - /web: built-in OpenEnv playground
  - /tasks: task definitions
  - /baseline: baseline and oracle reference values
  - /grader: grading formula and score references

## Run locally

Start the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn server.app:app --host 0.0.0.0 --port 7680 --reload
```

Local development uses port `7680`.

Inspect helper routes:

```powershell
Invoke-RestMethod http://localhost:7680/tasks | ConvertTo-Json -Depth 6
Invoke-RestMethod http://localhost:7680/baseline | ConvertTo-Json -Depth 6
Invoke-RestMethod http://localhost:7680/grader | ConvertTo-Json -Depth 6
```

Run inference:

```powershell
.\.venv\Scripts\python.exe inference.py
```

## Docker

The Docker image listens on port `7680`.

Build:

```powershell
docker build -t pollution-exposure-minimizer-environment -f Dockerfile .
```

Run:

```powershell
docker run -p 7680:7680 pollution-exposure-minimizer-environment
```

## OpenEnv validation

```powershell
openenv validate
```

## Hugging Face Spaces deployment

1. Create a Docker Space on Hugging Face.
2. Push this repo to the Space.
3. Make sure the Space is tagged with `openenv`.
4. Confirm `/reset`, `/step`, `/state`, `/tasks`, `/baseline`, and `/grader` respond.

If you use the OpenEnv CLI:

```powershell
openenv push --repo-id <your-hf-username>/pollution-exposure-minimizer-environment
```

## Project layout

```text
pollution-exposure-minimizer-env/
├── client.py
├── Dockerfile
├── .dockerignore
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── README.md
├── data/
│   └── pollution_city_graph.json
├── scripts/
│   └── preview_tasks.py
└── server/
    ├── aqi.py
    ├── app.py
    ├── baseline.py
    ├── city_graph.py
    ├── grader.py
    ├── pollution_exposure_minimizer_environment.py
    ├── tasks.py
    └── __init__.py
```
