"""Baseline inference runner for the Pollution Exposure Minimizer Environment.

Required environment variables:
    API_BASE_URL
    MODEL_NAME
    HF_TOKEN

Optional environment variables:
    OPENAI_API_KEY          Alternate auth variable; falls back to HF_TOKEN
    ENV_BASE_URL            Use an already-running environment server
    LOCAL_IMAGE_NAME        Local Docker image name for from_docker_image()
    TASK_LIST               Comma-separated task ids to evaluate
    MAX_STEPS               Override episode step cap
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from typing import List, Optional

from openai import OpenAI
from dotenv import load_dotenv

from client import PollutionExposureMinimizerEnv
from models import ActionOption, PollutionAction, PollutionObservation

load_dotenv()

DEFAULT_HF_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_HF_BASE_URL)
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct:together")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = HF_TOKEN or OPENAI_API_KEY
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
TASK_LIST = [
    task.strip()
    for task in os.getenv(
        "TASK_LIST",
        "easy_static_route,medium_multimodal_route,hard_dynamic_peak_route",
    ).split(",")
    if task.strip()
]
MAX_STEPS_OVERRIDE = int(os.getenv("MAX_STEPS", "12"))
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
    normalized_base_url = API_BASE_URL.strip()
    normalized_model_name = MODEL_NAME.strip().lower()

    use_openai_direct = bool(OPENAI_API_KEY) and (
        normalized_base_url in {"", "auto", DEFAULT_OPENAI_BASE_URL}
        or normalized_model_name.startswith("gpt-")
        or normalized_model_name.startswith("o1")
        or normalized_model_name.startswith("o3")
        or normalized_model_name.startswith("o4")
    )

    if use_openai_direct:
        return DEFAULT_OPENAI_BASE_URL, require_env("OPENAI_API_KEY", OPENAI_API_KEY)

    return normalized_base_url or DEFAULT_HF_BASE_URL, require_env(
        "HF_TOKEN or OPENAI_API_KEY",
        API_KEY,
    )


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


def parse_action_text(text: str, observation: PollutionObservation) -> tuple[PollutionAction, Optional[str]]:
    legal_by_choice = {}
    for index, option in enumerate(observation.legal_actions, start=1):
        legal_by_choice[f"A{index}"] = option

    choice_id = extract_choice_id(text, list(legal_by_choice.keys()))
    if choice_id is not None:
        option = legal_by_choice[choice_id]
        return (
            PollutionAction(
                action_type=option.action_type,
                target_node_id=option.target_node_id,
                mode=option.mode,
                rationale="model_selected_choice_id",
            ),
            None,
        )
    fallback = observation.legal_actions[0]
    return (
        PollutionAction(
            action_type=fallback.action_type,
            target_node_id=fallback.target_node_id,
            mode=fallback.mode,
            rationale="fallback_due_to_invalid_choice_id",
        ),
        "invalid_choice_id",
    )


def get_model_action(
    client: OpenAI,
    step: int,
    observation: PollutionObservation,
    history: List[str],
) -> tuple[PollutionAction, str, Optional[str]]:
    user_prompt = build_user_prompt(step, observation, history)
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
        first_error = type(exc).__name__
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": PLAIN_JSON_RETRY_PROMPT},
                    {"role": "user", "content": build_retry_prompt(len(observation.legal_actions))},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            text = (completion.choices[0].message.content or "").strip()
        except Exception as retry_exc:
            fallback = observation.legal_actions[0]
            fallback_text = serialize_action(
                PollutionAction(
                    action_type=fallback.action_type,
                    target_node_id=fallback.target_node_id,
                    mode=fallback.mode,
                )
            )
            return (
                PollutionAction(
                    action_type=fallback.action_type,
                    target_node_id=fallback.target_node_id,
                    mode=fallback.mode,
                    rationale="fallback_due_to_model_error",
                ),
                fallback_text,
                f"model_request_failed:{first_error}|retry_failed:{type(retry_exc).__name__}",
            )

    action, parse_error = parse_action_text(text, observation)
    return action, text or serialize_action(action), parse_error


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
    reached_destination = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    result = env.reset(task_id=task_id)
    visit_counts[result.observation.current_node_id] = 1
    try:
        for step in range(1, min(MAX_STEPS_OVERRIDE, result.observation.max_steps) + 1):
            if result.done:
                break

            current_node_id = result.observation.current_node_id
            action, action_text, action_error = get_model_action(
                client=client,
                step=step,
                observation=result.observation,
                history=history,
            )
            action, action_text, action_error = apply_guardrail(
                action=action,
                action_error=action_error,
                observation=result.observation,
                previous_node_id=previous_node_id,
                visit_counts=visit_counts,
            )
            result = env.step(action)
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

        score = float(result.observation.metadata.get("baseline", {}).get("oracle_cost", 0.0))
        state = env.state()
        reached_destination = (
            result.observation.current_node_id == result.observation.destination_node_id
        )
        if state.episode_score is not None:
            score = float(state.episode_score)
        else:
            score = 0.0

        success = reached_destination and result.done
        return score
    finally:
        final_state = env.state()
        final_score = float(final_state.episode_score or 0.0)
        log_end(
            success=success,
            steps=steps_taken,
            score=final_score,
            rewards=rewards,
        )


def main() -> None:
    base_url, api_key = resolve_api_config()
    client = OpenAI(base_url=base_url, api_key=api_key)

    if ENV_BASE_URL:
        with PollutionExposureMinimizerEnv(base_url=ENV_BASE_URL).sync() as env:
            for task_id in TASK_LIST:
                run_task(client, env, task_id)
        return

    image_name = require_env("LOCAL_IMAGE_NAME", LOCAL_IMAGE_NAME)
    with PollutionExposureMinimizerEnv.from_docker_image(image_name).sync() as env:
        for task_id in TASK_LIST:
            run_task(client, env, task_id)


if __name__ == "__main__":
    main()
