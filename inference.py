"""
Inference Script — SupportEnv (Customer Support Ticket Resolution)
===================================
MANDATORY
- Before submitting, ensure the following variables are defined:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local Docker image (if using from_docker_image)

STDOUT FORMAT
- [START] task=<task_name> env=<benchmark> model=<model_name>
- [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
- [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import re
import textwrap
from typing import Dict, List, Optional

from openai import OpenAI

from server.support_environment import SupportEnvironment

# ---------------------------------------------------------------------------
# Configuration  (matches sample inference.py exactly)
# ---------------------------------------------------------------------------
IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("SUPPORT_ENV_BENCHMARK", "support_env")

# Task configs: name -> (max_steps, max_tokens)
TASKS: Dict[str, Dict] = {
    "simple_inquiry": {"max_steps": 10, "max_tokens": 500, "temperature": 0.3},
    "complaint_resolution": {"max_steps": 15, "max_tokens": 700, "temperature": 0.3},
    "complex_escalation": {"max_steps": 25, "max_tokens": 1000, "temperature": 0.3},
}

SUCCESS_SCORE_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Logging (exact format required by hackathon)
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_clean = action.replace("\n", " ").replace("\r", "")
    if len(action_clean) > 200:
        action_clean = action_clean[:197] + "..."
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
You are an expert customer support agent. You are interacting with a support ticket resolution environment.

Your job is to:
1. READ the customer's message carefully
2. INVESTIGATE by looking up orders, customer profiles, and knowledge base articles
3. TAKE ACTION — issue refunds, reset passwords, escalate issues as needed
4. RESPOND to the customer with a clear, empathetic, comprehensive reply
5. CLOSE the ticket

IMPORTANT RULES:
- Issue exactly ONE command per turn (no explanations, no reasoning, just the command)
- Always investigate before taking action
- Always send a response before closing
- Be efficient — don't repeat commands unnecessarily

Available commands:
  view_ticket                         — View the current ticket details
  lookup_order <order_id>             — Retrieve order details
  lookup_customer <customer_id>       — View customer profile & history
  search_kb <query>                   — Search knowledge base articles
  check_policy <policy_name>          — Check company policy (refund_policy, shipping_policy, escalation_policy, discount_policy, warranty_policy)
  issue_refund <order_id> <amount>    — Process a refund
  apply_discount <customer_id> <pct>  — Apply goodwill discount (percentage)
  send_replacement <order_id>         — Ship replacement item
  reset_password <customer_id>        — Reset customer password
  escalate <department> <reason>      — Escalate to specialist (billing, technical, management, shipping, legal)
  update_status <order_id> <status>   — Update order status
  add_note <text>                     — Add internal note to ticket
  send_response <message>             — Send reply to customer (write the full message)
  close_ticket                        — Close the ticket (do this last)

Reply with ONLY the command. No explanations, no markdown, no quotes.
""").strip()


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------
def get_model_command(
    client: OpenAI,
    task_config: Dict,
    observation: Dict,
    history: List[str],
) -> str:
    """Ask the LLM for the next command given the current observation."""
    history_block = "\n".join(history[-6:]) if history else "None"

    user_prompt = textwrap.dedent(f"""
CURRENT TICKET:
Customer message: {observation.get('customer_message', '')}

TASK: {observation.get('task_description', '')}

LAST ACTION RESULT:
{observation.get('action_result', 'No action taken yet.')}

PROGRESS: {observation.get('resolution_progress', 0.0):.0%} complete
STEP: {observation.get('steps_taken', 0)} / {observation.get('max_steps', 10)}

PREVIOUS ACTIONS:
{history_block}

What is your next command?
""").strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=task_config["temperature"],
            max_tokens=task_config["max_tokens"],
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Clean up: remove markdown code blocks if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        # Take only the first line (the command)
        first_line = text.split("\n")[0].strip()
        return first_line if first_line else "view_ticket"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "view_ticket"


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------
def run_task(
    llm_client: OpenAI,
    env: SupportEnvironment,
    task_name: str,
    task_config: Dict,
) -> None:
    """Run a single task episode."""
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment with this task
        result = env.reset(task=task_name)
        obs = result["observation"]
        history: List[str] = []

        for step in range(1, task_config["max_steps"] + 1):
            if result.get("done", False):
                break

            # Get LLM command
            command = get_model_command(llm_client, task_config, obs, history)

            # Execute step directly on the environment object
            result = env.step({"command": command})
            obs = result["observation"]
            reward = result.get("reward") or 0.0
            done = result.get("done", False)
            error = obs.get("last_action_error")

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=command,
                reward=reward,
                done=done,
                error=error,
            )

            history.append(f"Step {step}: {command}")

            if done:
                break

        # Extract the final graded score from the environment
        action_result = obs.get("action_result", "")
        match = re.search(r"Final Score:\s*([\d.]+)", action_result)
        if match:
            score = float(match.group(1))
        else:
            # Fallback: use resolution progress from state
            env_state = env.state
            score = env_state.get("resolution_progress", 0.0)


        score = min(max(score, 0.01), 0.99)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_name} error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    """Run inference on all 3 tasks."""
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Instantiate the environment directly (no HTTP, no Docker needed)
    env = SupportEnvironment()
    print(f"[DEBUG] Using local SupportEnvironment instance", flush=True)

    for task_name, task_config in TASKS.items():
        run_task(llm_client, env, task_name, task_config)


if __name__ == "__main__":
    main()
