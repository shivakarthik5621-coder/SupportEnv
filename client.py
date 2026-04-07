"""
SupportEnv — HTTP Client

Provides SupportEnv, a lightweight async/sync HTTP client for
interacting with the SupportEnv FastAPI server.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import httpx

from .models import SupportAction, SupportObservation, SupportState


class StepResult:
    """Result from a step or reset call."""

    def __init__(
        self,
        observation: SupportObservation,
        reward: Optional[float],
        done: bool,
    ) -> None:
        self.observation = observation
        self.reward = reward
        self.done = done


class SupportEnv:
    """
    HTTP client for the Customer Support environment.

    Usage (async):
        env = SupportEnv(base_url="http://localhost:8000")
        result = await env.reset(task="simple_inquiry")
        result = await env.step(SupportAction(command="lookup_order ORD-1001"))
        await env.close()

    Usage (sync via from_docker_image — compatible with sample inference):
        env = await SupportEnv.from_docker_image("support-env:latest")
        result = await env.reset()
        result = await env.step(SupportAction(command="view_ticket"))
        await env.close()
    """

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)

    async def reset(
        self,
        task: Optional[str] = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> StepResult:
        """Reset the environment."""
        payload: Dict[str, Any] = {}
        if task:
            payload["task"] = task
        if seed is not None:
            payload["seed"] = seed
        if episode_id:
            payload["episode_id"] = episode_id

        resp = await self._client.post("/reset", json=payload)
        resp.raise_for_status()
        data = resp.json()

        obs = SupportObservation(**data["observation"])
        return StepResult(
            observation=obs,
            reward=data.get("reward"),
            done=data.get("done", False),
        )

    async def step(self, action: SupportAction) -> StepResult:
        """Execute one step."""
        payload = {
            "action": action.model_dump(exclude_none=True),
        }
        resp = await self._client.post("/step", json=payload)
        resp.raise_for_status()
        data = resp.json()

        obs = SupportObservation(**data["observation"])
        return StepResult(
            observation=obs,
            reward=data.get("reward"),
            done=data.get("done", False),
        )

    async def state(self) -> SupportState:
        """Get current state."""
        resp = await self._client.get("/state")
        resp.raise_for_status()
        return SupportState(**resp.json())


    # -------------------------------------------------------------------
    # Docker image support (matches sample inference pattern)
    # -------------------------------------------------------------------

    @classmethod
    async def from_docker_image(
        cls,
        image_name: str,
        port: int = 8000,
        **kwargs,
    ) -> "SupportEnv":
        """
        Start a Docker container from the given image and return a client.

        This is a simplified version — in production you'd use openenv-core's
        container management. For the hackathon, we support both Docker and
        direct connection to HF Spaces.
        """
        import subprocess
        import time

        container_name = f"supportenv-{port}"

        # Stop any existing container with the same name
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
        )

        # Start the container
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
        client = cls(base_url=base_url)
        client._container_name = container_name

        for _ in range(30):
            try:
                async with httpx.AsyncClient() as http:
                    resp = await http.get(f"{base_url}/health", timeout=2.0)
                    if resp.status_code == 200:
                        return client
            except Exception:
                pass
            await asyncio.sleep(1)

        raise RuntimeError(f"Container {container_name} failed to start")

    async def close(self) -> None:
        """Close client and optionally stop container."""
        await self._client.aclose()
        container = getattr(self, "_container_name", None)
        if container:
            import subprocess
            subprocess.run(
                ["docker", "rm", "-f", container],
                capture_output=True,
            )
