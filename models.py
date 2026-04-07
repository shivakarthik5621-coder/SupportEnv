"""
SupportEnv — Pydantic Models

Typed Action, Observation, and State models for the Customer Support
Ticket Resolution environment. Extends OpenEnv base types.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Base types — mirrors openenv.core.env_server.types but self-contained
# so the environment works both standalone and inside the OpenEnv repo.
# ---------------------------------------------------------------------------
class Action(BaseModel):
    """Base action class (OpenEnv compatible)."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """Base observation class (OpenEnv compatible)."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    done: bool = Field(default=False)
    reward: Optional[float] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class State(BaseModel):
    """Base state class (OpenEnv compatible)."""
    model_config = ConfigDict(extra="allow", validate_assignment=True)
    episode_id: Optional[str] = Field(default=None)
    step_count: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# SupportEnv — typed models
# ---------------------------------------------------------------------------

class SupportAction(Action):
    """
    An action in the customer-support environment.

    The agent sends a single text command per step, e.g.
        "lookup_order ORD-1234"
        "issue_refund ORD-1234 35.99"
        "send_response Dear customer, ..."
    """
    command: str = Field(
        ...,
        description="Text command to execute (e.g. 'lookup_order ORD-1234')",
        min_length=1,
        max_length=4096,
    )


class SupportObservation(Observation):
    """
    Observation returned after each step.

    Contains the ticket context, the result of the last command,
    and progress indicators the agent can use to decide its next action.
    """
    ticket_id: str = Field(
        default="",
        description="Current ticket identifier",
    )
    ticket_status: str = Field(
        default="open",
        description="Ticket status: open | pending | resolved | closed",
    )
    customer_message: str = Field(
        default="",
        description="The original customer message / inquiry",
    )
    action_result: str = Field(
        default="",
        description="Output from the last command executed",
    )
    actions_taken: List[str] = Field(
        default_factory=list,
        description="History of commands executed so far in this episode",
    )
    resolution_progress: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of the resolution checklist completed (0.0–1.0)",
    )
    steps_taken: int = Field(
        default=0,
        ge=0,
        description="Number of steps taken so far",
    )
    max_steps: int = Field(
        default=10,
        ge=1,
        description="Maximum steps allowed for this task",
    )
    task_description: str = Field(
        default="",
        description="High-level description of the current task objective",
    )
    available_commands: str = Field(
        default="",
        description="Summary of commands the agent can use",
    )
    last_action_error: Optional[str] = Field(
        default=None,
        description="Error message if last action failed, else None",
    )


class SupportState(State):
    """
    Internal environment state (accessible via state() endpoint).
    """
    task_name: str = Field(default="", description="Active task name")
    ticket_id: str = Field(default="", description="Current ticket ID")
    resolution_progress: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Checklist completion fraction",
    )
    actions_taken: List[str] = Field(
        default_factory=list,
        description="Actions executed in this episode",
    )
    response_sent: bool = Field(
        default=False,
        description="Whether a customer response has been sent",
    )
    ticket_closed: bool = Field(
        default=False,
        description="Whether the ticket has been closed",
    )
