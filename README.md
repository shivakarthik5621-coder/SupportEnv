---
title: SupportEnv — Customer Support Ticket Resolution
emoji: 🎧
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
tags:
  - openenv
---

# 🎧 SupportEnv — Customer Support Ticket Resolution

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment where AI agents learn to resolve customer support tickets through investigation, action, and response.

**Why this matters:** Customer support is a $400B+ industry. Every company needs it, every customer experiences it, and there's massive demand for AI agents that can handle tickets effectively — investigating issues, taking corrective actions, and composing helpful responses. This environment provides a standardized benchmark for training and evaluating such agents.

## Environment Overview

The agent acts as a customer support agent. Each episode presents a **support ticket** from a frustrated customer. The agent must:

1. **🔍 Investigate** — Look up orders, customer profiles, and knowledge base articles
2. **⚡ Act** — Issue refunds, reset passwords, escalate issues, apply discounts
3. **✉️ Respond** — Compose a clear, empathetic, comprehensive reply
4. **✅ Close** — Submit the resolution

The environment simulates a complete company backend with order databases, customer profiles, knowledge base articles, and company policies.

## Action Space

The agent sends one text command per step:

```python
SupportAction(command="lookup_order ORD-1234")
```

| Command | Description | Example |
|---------|-------------|---------|
| `view_ticket` | View current ticket details | `view_ticket` |
| `lookup_order <id>` | Retrieve order details | `lookup_order ORD-1234` |
| `lookup_customer <id>` | View customer profile | `lookup_customer CUST-100` |
| `search_kb <query>` | Search knowledge base | `search_kb return policy` |
| `check_policy <name>` | Check company policy | `check_policy refund_policy` |
| `issue_refund <id> <amt>` | Process a refund | `issue_refund ORD-1234 35.99` |
| `apply_discount <id> <pct>` | Apply goodwill discount | `apply_discount CUST-100 15` |
| `send_replacement <id>` | Ship replacement item | `send_replacement ORD-1234` |
| `reset_password <id>` | Reset password | `reset_password CUST-100` |
| `escalate <dept> <reason>` | Escalate to specialist | `escalate billing duplicate_charge` |
| `update_status <id> <status>` | Update order status | `update_status ORD-1234 returning` |
| `add_note <text>` | Add internal note | `add_note Customer is VIP` |
| `send_response <msg>` | Send reply to customer | `send_response Dear customer...` |
| `close_ticket` | Close and finalize | `close_ticket` |

## Observation Space

Each step returns:

```python
SupportObservation(
    ticket_id="TKT-5001",           # Ticket identifier
    ticket_status="open",            # open | pending | resolved | closed
    customer_message="...",          # Original customer message
    action_result="...",             # Output from last command
    actions_taken=["..."],           # History of commands executed
    resolution_progress=0.45,        # Checklist completion (0.0–1.0)
    steps_taken=3,                   # Steps used so far
    max_steps=10,                    # Maximum steps allowed
    task_description="...",          # Task objective
    available_commands="...",        # Command reference
    last_action_error=None,          # Error if last action failed
)
```

## Tasks

### Task 1: `simple_inquiry` (Easy)
**Order Status Check** — Customer asks where their package is.
- Look up the order, find tracking info, respond with status
- **Optimal:** 3–4 steps | **Max:** 10 steps

### Task 2: `complaint_resolution` (Medium)
**Wrong Item + Refund** — Customer received wrong item, demands refund.
- Investigate order & customer history, issue refund, compose empathetic response with return instructions
- **Optimal:** 5–6 steps | **Max:** 15 steps

### Task 3: `complex_escalation` (Hard)
**Multi-Issue Resolution** — Customer has 3 simultaneous problems: damaged laptop, duplicate billing charge, and broken password reset.
- Investigate all issues, issue refund for damaged item, escalate billing to specialist, reset password, compose comprehensive response addressing all 3 issues
- **Optimal:** 8–10 steps | **Max:** 25 steps

## Reward Design

| Signal | Value |
|--------|-------|
| Correct investigation/action | +0.05 to +0.15 (via checklist progress) |
| Good response elements | +0.05 per required element |
| Wrong/forbidden action | −0.05 to −0.10 |
| Irrelevant command | 0.0 (wastes a step) |
| Good close (score ≥ 0.7) | +0.10 bonus |
| Premature close (score < 0.3) | −0.05 penalty |

**Scoring formula:**
```
final_score = (action_score × 0.40) + (response_score × 0.40) + (efficiency_score × 0.20) − penalties
```

## Setup Instructions

### Prerequisites
- Python 3.10+
- Docker (for containerized runs)
- `pip install openenv-core`

### Local Development

```bash
# Install dependencies
pip install -e .

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal, test:
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" \
  -d '{"action": {"command": "view_ticket"}}'
```

### Docker

```bash
# Build
docker build -t support-env .

# Run
docker run -p 8000:8000 support-env

# Test
curl -X POST http://localhost:8000/reset -d '{}'
```

### Run Inference

```bash
# Set environment variables
export HF_TOKEN="your_hf_token"
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

# Option 1: Connect to running server
export ENV_BASE_URL="http://localhost:8000"
python inference.py

# Option 2: Use Docker image
export IMAGE_NAME="support-env"
python inference.py
```

### Deploy to Hugging Face Spaces

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create support-env --type space --space-sdk docker
git clone https://huggingface.co/spaces/purple5621/support-env
# Copy all files into the repo and push
```

## Baseline Scores

| Task | Difficulty | Score | Steps |
|------|-----------|-------|-------|
| `simple_inquiry` | Easy | ~0.75–0.90 | 3–5 |
| `complaint_resolution` | Medium | ~0.55–0.75 | 6–10 |
| `complex_escalation` | Hard | ~0.35–0.55 | 12–20 |

*Scores measured with Qwen2.5-72B-Instruct via HuggingFace router.*

## Project Structure

```
├── openenv.yaml          # OpenEnv manifest
├── pyproject.toml         # Package config
├── Dockerfile             # Container image
├── README.md              # This file
├── inference.py           # Baseline inference script
├── __init__.py            # Package exports
├── models.py              # Pydantic Action/Observation/State
├── client.py              # SupportEnv HTTP client
└── server/
    ├── __init__.py
    ├── app.py             # FastAPI application
    ├── support_environment.py  # Core environment logic
    ├── scenarios.py       # Ticket scenarios & solution checklists
    ├── graders.py         # Deterministic graders
    └── backend.py         # Simulated company backend
```

## OpenEnv API

```python
import asyncio
from client import SupportEnv
from models import SupportAction

async def main():
    env = SupportEnv(base_url="http://localhost:8000")

    # Reset with a task
    result = await env.reset(task="simple_inquiry")
    print(result.observation.customer_message)

    # Take actions
    result = await env.step(SupportAction(command="lookup_order ORD-1001"))
    print(result.observation.action_result)
    print(f"Progress: {result.observation.resolution_progress:.0%}")

    # Send response and close
    result = await env.step(SupportAction(command="send_response Dear Alex, your order has shipped..."))
    result = await env.step(SupportAction(command="close_ticket"))
    print(f"Final reward: {result.reward}")

    await env.close()

asyncio.run(main())
```
