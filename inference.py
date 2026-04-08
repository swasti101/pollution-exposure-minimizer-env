"""Baseline inference runner for the Pollution Exposure Minimizer Environment.

Optional environment variables:
    API_BASE_URL
    API_KEY
    MODEL_NAME
    ENV_BASE_URL            Use an already-running environment server
    LOCAL_IMAGE_NAME        Local Docker image name for from_docker_image()
    TASK_LIST               Comma-separated task ids to evaluate
    MAX_STEPS               Override episode step cap
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import PollutionExposureMinimizerEnv
from models import ActionOption, PollutionAction, PollutionObservation

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_HF_BASE_URL = "https://router.huggingface.co/v1"
API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_HF_BASE_URL).strip()
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct:together").strip()
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "").strip()
LOCAL_IMAGE_NAME = (
    os.getenv("LOCAL_IMAGE_NAME", "").strip()
    or os.getenv("IMAGE_NAME", "").strip()
    or "pollution-exposure-minimizer-environment"
)
TASK_LIST = [
    task.strip()
    for task in os.getenv(
        "TASK_LIST",
        "easy_static_route,medium_multimodal_route,hard_dynamic_peak_route",
    ).split(",")
    if task.strip()
]
if not TASK_LIST:
    TASK_LIST = [
        "easy_static_route",
        "medium_multimodal_route",
        "hard_dynamic_peak_route",
    ]
try:
    MAX_STEPS_OVERRIDE = int(os.getenv("MAX_STEPS", "12"))
except ValueError:
    MAX_STEPS_OVERRIDE = 12
TEMPERATURE = 0.0
MAX_TOKENS = 32
BENCHMARK = "pollution-exposure-minimizer-environment"

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an urban mobility agent minimizing pollution exposure.

    You must choose exactly one legal action from the provided observation.
    Always return exactly one action id such as A1 or A2.
    Do not incluse any extra text, JSON, markdown, punctuation, or code fences. Return only the action id.

    Rules:
    - Use only an action id that appears in legal_actions.
    - Prefer lower exposure, but remember that time also matters.
    - On the hard task, waiting is legal only if shown in legal_actions.
    - Return only the action id. No JSON. No markdown. No explanation. No code fences.
    """
).strip()

PLAIN_JSON_RETRY_PROMPT = textwrap.dedent(
    """
    Return exactly one action id on a single line, such as A1.
    Do not include JSON, markdown, prose, punctuation, or code fences.
    """
).strip()


def build_retry_prompt(num_actions: int) -> str:
    legal_choice_ids = ", ".join(f"A{index}" for index in range(1, num_actions + 1))
    return (
        f"Choose exactly one of these action ids: {legal_choice_ids}. "
        "Return only the action id. Do not include any extra text."
    )


def require_env(name: str, value: Optional[str]) -> str:
    if value:
        return value
    raise RuntimeError(f"Missing required environment variable: {name}")


def resolve_api_config() -> tuple[str, str]:
    normalized_base_url = API_BASE_URL or DEFAULT_HF_BASE_URL
    return normalized_base_url, API_KEY


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def sanitize_for_log(text: str) -> str:
    return text.replace("\n", " ").replace("\r", " ").strip()


def serialize_action(action: PollutionAction) -> str:
    return json.dumps(
        {
            "action_type": action.action_type,
            "target_node_id": action.target_node_id,
            "mode": action.mode,
        },
        separators=(",", ":"),
    )


def action_from_option(option: ActionOption, rationale: str) -> PollutionAction:
    return PollutionAction(
        action_type=option.action_type,
        target_node_id=option.target_node_id,
        mode=option.mode,
        rationale=rationale,
    )


def extract_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def fallback_option_score(
    option: ActionOption,
    observation: PollutionObservation,
    previous_node_id: Optional[str],
    visit_counts: dict[str, int],
) -> float:
    node_positions = {
        node.node_id: (node.x, node.y)
        for node in observation.graph_nodes
    }
    current_x, current_y = node_positions[observation.current_node_id]
    destination_x, destination_y = node_positions[observation.destination_node_id]
    current_distance = abs(destination_x - current_x) + abs(destination_y - current_y)
    score = option.estimated_exposure + (12.0 * option.estimated_time_minutes)

    if option.action_type == "move" and option.target_node_id == observation.destination_node_id:
        score -= 400.0

    if option.action_type == "wait":
        score += 180.0
        score += visit_counts.get(observation.current_node_id, 0) * 140.0

    if previous_node_id and option.target_node_id == previous_node_id:
        score += 90.0

    if option.target_node_id:
        target_x, target_y = node_positions[option.target_node_id]
        target_distance = abs(destination_x - target_x) + abs(destination_y - target_y)
        score += target_distance * 24.0
        score -= max(0.0, current_distance - target_distance) * 48.0
        score += visit_counts.get(option.target_node_id, 0) * 55.0

    return score


def choose_fallback_action(
    observation: PollutionObservation,
    previous_node_id: Optional[str],
    visit_counts: dict[str, int],
) -> PollutionAction:
    best_option = min(
        observation.legal_actions,
        key=lambda option: fallback_option_score(
            option,
            observation,
            previous_node_id,
            visit_counts,
        ),
    )
    return action_from_option(best_option, rationale="deterministic_fallback")


def build_user_prompt(step: int, observation: PollutionObservation, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    legal_actions = []
    for index, option in enumerate(observation.legal_actions, start=1):
        legal_actions.append(
            {
                "choice_id": f"A{index}",
                "action_type": option.action_type,
                "target_node_id": option.target_node_id,
                "target_label": option.target_label,
                "mode": option.mode,
                "estimated_exposure": option.estimated_exposure,
                "estimated_time_minutes": option.estimated_time_minutes,
                "description": option.description,
            }
        )
    return textwrap.dedent(
        f"""
        Step: {step}
        Task id: {observation.task_id}
        Task name: {observation.task_name}
        Difficulty: {observation.difficulty}
        Current node: {observation.current_node_label} ({observation.current_node_id})
        Destination: {observation.destination_node_label} ({observation.destination_node_id})
        Current hour: {observation.current_hour:.2f}
        Cumulative exposure: {observation.cumulative_exposure:.2f}
        Cumulative time minutes: {observation.cumulative_time_minutes}
        Cumulative cost: {observation.cumulative_cost:.2f}
        Steps remaining: {observation.steps_remaining}
        Legal actions:
        {json.dumps(legal_actions, indent=2)}
        Previous steps:
        {history_block}
        Return only one action id selecting a single legal action.
        Required format: A1
        Return only the action id. Do not add any surrounding text.
        """
    ).strip()


def extract_choice_id(text: str, legal_choice_ids: list[str]) -> Optional[str]:
    cleaned = text.strip()
    if cleaned in legal_choice_ids:
        return cleaned

    # Accept simple variants like "Answer: A2" or "I choose A2"
    matches = re.findall(r"\bA\d+\b", cleaned.upper())
    for match in matches:
        if match in legal_choice_ids:
            return match

    return None


def parse_model_output(text: str, observation: PollutionObservation) -> PollutionAction:
    legal_by_choice = {}
    for index, option in enumerate(observation.legal_actions, start=1):
        legal_by_choice[f"A{index}"] = option

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("empty_model_output")

    extracted = extract_first_json_object(cleaned)
    if extracted is not None:
        try:
            payload: dict[str, object] = json.loads(extracted)
        except json.JSONDecodeError:
            raise ValueError("invalid_json_output")

        choice_id = str(payload.get("choice_id") or payload.get("action_id") or "").strip().upper()
        if choice_id in legal_by_choice:
            option = legal_by_choice[choice_id]
            return action_from_option(option, rationale="model_selected_choice_id")

        action_type = payload.get("action_type")
        target_node_id = payload.get("target_node_id")
        mode = payload.get("mode")
        for option in observation.legal_actions:
            if (
                option.action_type == action_type
                and option.target_node_id == target_node_id
                and option.mode == mode
            ):
                return action_from_option(option, rationale="model_selected_json_action")

        raise ValueError("json_output_did_not_match_legal_action")

    choice_id = extract_choice_id(text, list(legal_by_choice.keys()))
    if choice_id is not None:
        option = legal_by_choice[choice_id]
        return action_from_option(option, rationale="model_selected_choice_id")

    raise ValueError("invalid_choice_id")


def get_model_action(
    client: OpenAI,
    step: int,
    observation: PollutionObservation,
    history: List[str],
    previous_node_id: Optional[str],
    visit_counts: dict[str, int],
) -> tuple[PollutionAction, str, Optional[str]]:
    user_prompt = build_user_prompt(step, observation, history)
    text = ""
    model_error: Optional[str] = None
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        model_error = f"model_request_failed:{type(exc).__name__}"

    try:
        action = parse_model_output(text, observation)
        parse_error: Optional[str] = None
    except Exception:
        action = choose_fallback_action(observation, previous_node_id, visit_counts)
        parse_error = "parse_failed"

    action_text = serialize_action(action)
    if model_error and parse_error:
        return action, action_text, f"{model_error}|{parse_error}"
    if model_error:
        return action, action_text, model_error
    return action, action_text, parse_error


def apply_guardrail(
    action: PollutionAction,
    action_error: Optional[str],
    observation: PollutionObservation,
    previous_node_id: Optional[str],
    visit_counts: dict[str, int],
) -> tuple[PollutionAction, str, Optional[str]]:
    if action_error is not None:
        fallback = choose_fallback_action(observation, previous_node_id, visit_counts)
        return (
            fallback,
            serialize_action(fallback),
            f"{action_error}|fallback_policy",
        )

    if (
        action.action_type == "move"
        and previous_node_id is not None
        and action.target_node_id == previous_node_id
        and len(observation.legal_actions) > 1
    ):
        fallback = choose_fallback_action(observation, previous_node_id, visit_counts)
        if fallback.target_node_id != action.target_node_id or fallback.mode != action.mode:
            return (
                fallback,
                serialize_action(fallback),
                "backtrack_override",
            )

    return action, serialize_action(action), action_error


def run_task(client: OpenAI, env: PollutionExposureMinimizerEnv, task_id: str) -> float:
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    previous_node_id: Optional[str] = None
    visit_counts: dict[str, int] = {}

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        try:
            result = env.reset(task_id=task_id)
        except Exception:
            return 0.0

        visit_counts[result.observation.current_node_id] = 1

        for step in range(1, min(MAX_STEPS_OVERRIDE, result.observation.max_steps) + 1):
            if result.done:
                break

            current_node_id = result.observation.current_node_id
            action, action_text, action_error = get_model_action(
                client=client,
                step=step,
                observation=result.observation,
                history=history,
                previous_node_id=previous_node_id,
                visit_counts=visit_counts,
            )
            action, action_text, action_error = apply_guardrail(
                action=action,
                action_error=action_error,
                observation=result.observation,
                previous_node_id=previous_node_id,
                visit_counts=visit_counts,
            )
            try:
                result = env.step(action)
            except Exception as exc:
                log_step(
                    step=step,
                    action=sanitize_for_log(action_text),
                    reward=0.0,
                    done=False,
                    error=f"step_failed:{type(exc).__name__}",
                )
                break
            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step
            previous_node_id = current_node_id
            visit_counts[result.observation.current_node_id] = (
                visit_counts.get(result.observation.current_node_id, 0) + 1
            )

            log_step(
                step=step,
                action=sanitize_for_log(action_text),
                reward=reward,
                done=result.done,
                error=action_error,
            )

            history.append(
                f"step={step} action={sanitize_for_log(action_text)} reward={reward:.2f} "
                f"node={result.observation.current_node_id}"
            )

            if result.done:
                break

        state = env.state()
        score = state.episode_score if state.episode_score is not None else 0.0
        score = max(0.0, min(1.0, float(score)))
        reached_destination = state.current_node_id == state.destination_node_id or (
            result.observation.current_node_id == result.observation.destination_node_id
        )
        success = bool(reached_destination)
        return score
    except Exception:
        return 0.0
    finally:
        try:
            final_state = env.state()
            final_score = final_state.episode_score if final_state.episode_score is not None else 0.0
        except Exception:
            final_score = score
        final_score = max(0.0, min(1.0, float(final_score)))
        log_end(
            success=success,
            steps=steps_taken,
            score=final_score,
            rewards=rewards,
        )


async def main() -> None:
    base_url, api_key = resolve_api_config()
    client = OpenAI(base_url=base_url, api_key=api_key)
    task_started = False
    emit_final_end = False
    try:
        try:
            env_client = (
                PollutionExposureMinimizerEnv(base_url=ENV_BASE_URL)
                if ENV_BASE_URL
                else await PollutionExposureMinimizerEnv.from_docker_image(LOCAL_IMAGE_NAME)
            )
        except Exception:
            log_start(task="startup", env=BENCHMARK, model=MODEL_NAME)
            emit_final_end = True
            return

        with env_client.sync() as env:
            for task_id in TASK_LIST:
                task_started = True
                run_task(client, env, task_id)
    except Exception:
        if not task_started:
            log_start(task="startup", env=BENCHMARK, model=MODEL_NAME)
        emit_final_end = True
    finally:
        if emit_final_end or not task_started:
            log_end(success=False, steps=0, score=0.0, rewards=[])


if __name__ == "__main__":
    asyncio.run(main())
