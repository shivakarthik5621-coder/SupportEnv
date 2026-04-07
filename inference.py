"""
Inference Script — SupportEnv (Customer Support Ticket Resolution)
===================================
MANDATORY
- Before submitting, ensure the following variables are defined:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    IMAGE_NAME     The name of the local Docker image (if using from_docker_image).

STDOUT FORMAT
- [START] task=<task_name> env=<benchmark> model=<model_name>
- [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
- [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import re
import textwrap
from typing import Dict, List, Optional

import httpx
from openai import OpenAI


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy_token"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("SUPPORT_ENV_BENCHMARK", "support_env")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

# Task configs: name → (max_steps, max_tokens)
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
    # Truncate action for readability (remove newlines)
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
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
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
    # Build context
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
# Environment interaction via HTTP
# ---------------------------------------------------------------------------
class EnvHTTPClient:
    """Minimal async HTTP client for the SupportEnv server."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)

    async def reset(self, task: str) -> Dict:
        resp = await self._client.post("/reset", json={"task": task})
        resp.raise_for_status()
        return resp.json()

    async def step(self, command: str) -> Dict:
        resp = await self._client.post(
            "/step",
            json={"action": {"command": command}},
        )
        resp.raise_for_status()
        return resp.json()

    async def state(self) -> Dict:
        resp = await self._client.get("/state")
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Docker support
# ---------------------------------------------------------------------------
async def start_docker_container(image_name: str, port: int = 8000) -> str:
    """Start a Docker container and return the base URL."""
    import subprocess, time

    container_name = f"supportenv-inference-{port}"

    # Remove existing
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    # Start
    subprocess.run(
        [
            "docker", "run", "-d",
            "--name", container_name,
            "-p", f"{port}:8000",
            image_name,
        ],
        check=True,
        capture_output=True,
    )

    # Wait for health
    base_url = f"http://localhost:{port}"
    for i in range(30):
        try:
            async with httpx.AsyncClient() as http:
                resp = await http.get(f"{base_url}/health", timeout=2.0)
                if resp.status_code == 200:
                    print(f"[DEBUG] Container ready on {base_url}", flush=True)
                    return base_url
        except Exception:
            pass
        await asyncio.sleep(1)

    raise RuntimeError(f"Container {container_name} failed to become healthy")


async def stop_docker_container(port: int = 8000) -> None:
    """Stop and remove the Docker container."""
    import subprocess
    container_name = f"supportenv-inference-{port}"
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------
async def run_task(
    llm_client: OpenAI,
    env: EnvHTTPClient,
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
        result = await env.reset(task=task_name)
        obs = result["observation"]
        history: List[str] = []

        for step in range(1, task_config["max_steps"] + 1):
            if result.get("done", False):
                break

            # Get LLM command
            command = get_model_command(llm_client, task_config, obs, history)

            # Execute
            result = await env.step(command)
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
        # The grader embeds "Final Score: X.XX/1.00" in the action_result
        action_result = obs.get("action_result", "")
        match = re.search(r"Final Score:\s*([\d.]+)", action_result)
        if match:
            score = float(match.group(1))
        else:
            # Fallback: use resolution progress from state
            env_state = await env.state()
            score = env_state.get("resolution_progress", 0.0)

        # Clamp score
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_name} error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    """Run inference on all 3 tasks."""
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Determine environment URL
    base_url = ENV_BASE_URL
    using_docker = False

    if IMAGE_NAME:
        print(f"[DEBUG] Starting Docker container: {IMAGE_NAME}", flush=True)
        base_url = await start_docker_container(IMAGE_NAME)
        using_docker = True
    else:
        print(f"[DEBUG] Connecting to environment at: {base_url}", flush=True)

    env = EnvHTTPClient(base_url)

    try:
        for task_name, task_config in TASKS.items():
            await run_task(llm_client, env, task_name, task_config)
            # Small delay between tasks
            await asyncio.sleep(1)
    finally:
        await env.close()
        if using_docker:
            await stop_docker_container()


if __name__ == "__main__":
    asyncio.run(main())
