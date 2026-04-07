"""
SupportEnv — FastAPI Application

Exposes the Customer Support environment via HTTP endpoints:
  POST /reset   — reset the environment
  POST /step    — execute an action
  GET  /state   — get current state
  GET  /health  — health check
  GET  /schema  — action/observation schemas
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .support_environment import SupportEnvironment

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    task: Optional[str] = Field(
        default=None,
        description="Task to load: simple_inquiry, complaint_resolution, complex_escalation",
    )

    class Config:
        extra = "allow"


class StepRequest(BaseModel):
    action: Dict[str, Any] = Field(
        ..., description="Action dict with 'command' key"
    )
    timeout_s: Optional[float] = None

    class Config:
        extra = "allow"


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False


class HealthResponse(BaseModel):
    status: str = "healthy"


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SupportEnv — Customer Support Ticket Resolution",
    description=(
        "An OpenEnv environment where AI agents learn to resolve "
        "customer support tickets through investigation, action, and response."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single environment instance (one episode at a time)
env = SupportEnvironment()


@app.get("/health")
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy")


@app.post("/reset")
async def reset(request: ResetRequest = None) -> StepResponse:
    """Reset the environment and start a new episode."""
    if request is None:
        request = ResetRequest()

    result = env.reset(
        seed=request.seed,
        episode_id=request.episode_id,
        task=request.task,
    )
    return StepResponse(
        observation=result["observation"],
        reward=result.get("reward"),
        done=result.get("done", False),
    )


@app.post("/step")
async def step(request: StepRequest) -> StepResponse:
    """Execute one action in the environment."""
    result = env.step(request.action)
    return StepResponse(
        observation=result["observation"],
        reward=result.get("reward"),
        done=result.get("done", False),
    )


@app.get("/state")
async def state() -> Dict[str, Any]:
    """Return the current environment state."""
    return env.state


@app.get("/schema")
async def schema() -> Dict[str, Any]:
    """Return the action and observation JSON schemas."""
    import sys
    from pathlib import Path
    parent = str(Path(__file__).resolve().parent.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from models import SupportAction, SupportObservation, SupportState

    return {
        "action": SupportAction.model_json_schema(),
        "observation": SupportObservation.model_json_schema(),
        "state": SupportState.model_json_schema(),
    }


@app.get("/tasks")
async def tasks() -> Dict[str, Any]:
    """List available tasks."""
    from .scenarios import get_all_task_names, get_scenario

    result = {}
    for name in get_all_task_names():
        sc = get_scenario(name)
        result[name] = {
            "difficulty": sc.difficulty,
            "max_steps": sc.max_steps,
            "description": sc.task_description,
        }
    return {"tasks": result}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run the server directly."""
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
