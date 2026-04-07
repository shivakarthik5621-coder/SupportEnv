"""
SupportEnv — Core Environment

Implements the OpenEnv Environment interface:
  reset(task=...) → initial observation
  step(action)    → observation with reward, done
  state           → current episode state
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Optional

from .backend import CompanyBackend, build_backend_for_task
from .graders import TaskGrader
from .scenarios import Scenario, get_all_task_names, get_scenario


# ---------------------------------------------------------------------------
# Commands specification (shown to the agent)
# ---------------------------------------------------------------------------

AVAILABLE_COMMANDS = """Available commands:
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
  send_response <message>             — Send reply to customer
  close_ticket                        — Close the ticket (ends episode)""".strip()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SupportEnvironment:
    """
    Customer Support Ticket Resolution Environment.

    The agent receives a customer ticket and must resolve it by:
    1. Investigating (lookups, KB search, policy checks)
    2. Taking actions (refunds, password resets, escalations)
    3. Composing a response
    4. Closing the ticket
    """

    def __init__(self) -> None:
        self._backend: Optional[CompanyBackend] = None
        self._scenario: Optional[Scenario] = None
        self._grader: Optional[TaskGrader] = None
        self._episode_id: str = ""
        self._step_count: int = 0
        self._actions_taken: List[str] = []
        self._prev_progress: float = 0.0
        self._done: bool = False
        self._task_name: str = ""
        self._last_error: Optional[str] = None

    # -------------------------------------------------------------------
    # OpenEnv interface
    # -------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Reset environment with a new ticket scenario."""
        task_name = task or kwargs.get("task_name", "simple_inquiry")
        if task_name not in get_all_task_names():
            task_name = "simple_inquiry"

        self._task_name = task_name
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._actions_taken = []
        self._prev_progress = 0.0
        self._done = False
        self._last_error = None

        # Load scenario and backend
        self._scenario = get_scenario(task_name)
        self._backend = build_backend_for_task(task_name)
        self._grader = TaskGrader(self._scenario)

        return self._build_observation(
            action_result=f"Ticket {self._scenario.ticket_id} opened. Read the customer message and resolve their issue.",
            reward=0.0,
        )

    def step(self, action_dict: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute one step: parse command, execute, compute reward."""
        if self._done:
            return self._build_observation(
                action_result="Episode is already done. Call reset() to start a new episode.",
                reward=0.0,
                done=True,
            )

        if self._scenario is None or self._backend is None:
            return self._build_observation(
                action_result="Environment not initialized. Call reset() first.",
                reward=0.0,
            )

        self._step_count += 1
        command = action_dict.get("command", "").strip()

        if not command:
            self._last_error = "Empty command"
            return self._build_observation(
                action_result="❌ Empty command. Please provide a valid command.",
                reward=0.0,
            )

        # Record the action
        self._actions_taken.append(command)

        # Parse and execute
        result = self._execute_command(command)

        # Compute progress and reward
        new_progress = self._grader.compute_progress(
            self._backend, self._actions_taken
        )
        reward = new_progress - self._prev_progress

        # Check for forbidden action penalty
        penalty = self._check_step_penalty(command)
        reward -= penalty

        self._prev_progress = new_progress

        # Check if episode is done
        if self._backend.ticket_closed or self._step_count >= self._scenario.max_steps:
            self._done = True

            # Final grading
            grade_result = self._grader.grade(
                self._backend,
                self._actions_taken,
                self._step_count,
            )
            final_score = grade_result["score"]

            # Add close bonus/penalty
            if self._backend.ticket_closed:
                if final_score >= 0.7:
                    reward += 0.10  # bonus for good resolution
                elif final_score < 0.3:
                    reward -= 0.05  # penalty for premature close
                result += f"\n\n📊 Final Score: {final_score:.2f}/1.00"
                result += f"\n  Actions: {grade_result['action_score']:.2f}"
                result += f"\n  Response: {grade_result['response_score']:.2f}"
                result += f"\n  Efficiency: {grade_result['efficiency_score']:.2f}"
            else:
                result += "\n\n⏰ Maximum steps reached. Episode ended."
                result += f"\n📊 Final Score: {final_score:.2f}/1.00"

        # Clamp reward
        reward = round(max(-1.0, min(1.0, reward)), 4)

        return self._build_observation(
            action_result=result,
            reward=reward,
            done=self._done,
        )

    @property
    def state(self) -> Dict[str, Any]:
        """Return current environment state."""
        return {
            "episode_id": self._episode_id,
            "step_count": self._step_count,
            "task_name": self._task_name,
            "ticket_id": self._scenario.ticket_id if self._scenario else "",
            "resolution_progress": self._prev_progress,
            "actions_taken": list(self._actions_taken),
            "response_sent": self._backend.response_sent is not None if self._backend else False,
            "ticket_closed": self._backend.ticket_closed if self._backend else False,
        }

    def get_final_score(self) -> float:
        """Get the final graded score for the episode (0.0–1.0)."""
        if not self._grader or not self._backend:
            return 0.0
        result = self._grader.grade(
            self._backend, self._actions_taken, self._step_count
        )
        return result["score"]

    # -------------------------------------------------------------------
    # Command parser and executor
    # -------------------------------------------------------------------

    def _execute_command(self, command: str) -> str:
        """Parse and execute a text command."""
        parts = self._parse_command(command)
        cmd = parts[0].lower() if parts else ""
        args = parts[1:] if len(parts) > 1 else []

        try:
            if cmd == "view_ticket":
                return self._cmd_view_ticket()
            elif cmd == "lookup_order":
                return self._cmd_lookup_order(args)
            elif cmd == "lookup_customer":
                return self._cmd_lookup_customer(args)
            elif cmd == "search_kb":
                return self._cmd_search_kb(args, command)
            elif cmd == "check_policy":
                return self._cmd_check_policy(args)
            elif cmd == "issue_refund":
                return self._cmd_issue_refund(args)
            elif cmd == "apply_discount":
                return self._cmd_apply_discount(args)
            elif cmd == "send_replacement":
                return self._cmd_send_replacement(args)
            elif cmd == "reset_password":
                return self._cmd_reset_password(args)
            elif cmd == "escalate":
                return self._cmd_escalate(args, command)
            elif cmd == "update_status":
                return self._cmd_update_status(args)
            elif cmd == "add_note":
                return self._cmd_add_note(args, command)
            elif cmd == "send_response":
                return self._cmd_send_response(args, command)
            elif cmd == "close_ticket":
                return self._cmd_close_ticket()
            else:
                self._last_error = f"Unknown command: {cmd}"
                return (
                    f"❌ Unknown command: '{cmd}'.\n\n{AVAILABLE_COMMANDS}"
                )
        except Exception as e:
            self._last_error = str(e)
            return f"❌ Error executing '{cmd}': {e}"

    def _parse_command(self, command: str) -> List[str]:
        """Parse a command string into parts, respecting quoted strings."""
        # Handle send_response specially — everything after the command is the message
        cmd_lower = command.strip().lower()
        if cmd_lower.startswith("send_response"):
            msg = command.strip()[len("send_response"):].strip()
            return ["send_response", msg] if msg else ["send_response"]

        if cmd_lower.startswith("add_note"):
            note = command.strip()[len("add_note"):].strip()
            return ["add_note", note] if note else ["add_note"]

        if cmd_lower.startswith("search_kb"):
            query = command.strip()[len("search_kb"):].strip()
            return ["search_kb", query] if query else ["search_kb"]

        if cmd_lower.startswith("escalate"):
            rest = command.strip()[len("escalate"):].strip()
            parts = rest.split(None, 1)
            return ["escalate"] + parts

        # Standard space-separated parsing
        return command.strip().split()

    # --- Individual command handlers ---

    def _cmd_view_ticket(self) -> str:
        return (
            f"🎫 Ticket: {self._scenario.ticket_id}\n"
            f"Customer: {self._scenario.customer_id}\n"
            f"Status: {'closed' if self._backend.ticket_closed else 'open'}\n\n"
            f"Customer Message:\n{self._scenario.customer_message}"
        )

    def _cmd_lookup_order(self, args: List[str]) -> str:
        if not args:
            return "❌ Usage: lookup_order <order_id>"
        return self._backend.lookup_order(args[0].upper())

    def _cmd_lookup_customer(self, args: List[str]) -> str:
        if not args:
            return "❌ Usage: lookup_customer <customer_id>"
        return self._backend.lookup_customer(args[0].upper())

    def _cmd_search_kb(self, args: List[str], raw: str) -> str:
        query = args[0] if args else ""
        if not query:
            return "❌ Usage: search_kb <query>"
        return self._backend.search_kb(query)

    def _cmd_check_policy(self, args: List[str]) -> str:
        if not args:
            return "❌ Usage: check_policy <policy_name>\nAvailable: refund_policy, shipping_policy, escalation_policy, discount_policy, warranty_policy"
        return self._backend.check_policy(args[0].lower())

    def _cmd_issue_refund(self, args: List[str]) -> str:
        if len(args) < 2:
            return "❌ Usage: issue_refund <order_id> <amount>"
        return self._backend.issue_refund(args[0].upper(), args[1])

    def _cmd_apply_discount(self, args: List[str]) -> str:
        if len(args) < 2:
            return "❌ Usage: apply_discount <customer_id> <percentage>"
        return self._backend.apply_discount(args[0].upper(), args[1])

    def _cmd_send_replacement(self, args: List[str]) -> str:
        if not args:
            return "❌ Usage: send_replacement <order_id>"
        return self._backend.send_replacement(args[0].upper())

    def _cmd_reset_password(self, args: List[str]) -> str:
        if not args:
            return "❌ Usage: reset_password <customer_id>"
        return self._backend.reset_password(args[0].upper())

    def _cmd_escalate(self, args: List[str], raw: str) -> str:
        if len(args) < 2:
            return "❌ Usage: escalate <department> <reason>\nDepartments: billing, technical, management, shipping, legal"
        return self._backend.escalate(args[0], args[1])

    def _cmd_update_status(self, args: List[str]) -> str:
        if len(args) < 2:
            return "❌ Usage: update_status <order_id> <new_status>"
        return self._backend.update_status(args[0].upper(), args[1])

    def _cmd_add_note(self, args: List[str], raw: str) -> str:
        note = args[0] if args else ""
        if not note:
            return "❌ Usage: add_note <note_text>"
        return self._backend.add_note(note)

    def _cmd_send_response(self, args: List[str], raw: str) -> str:
        message = args[0] if args else ""
        if not message:
            return "❌ Usage: send_response <message>"
        return self._backend.send_response(message)

    def _cmd_close_ticket(self) -> str:
        return self._backend.close_ticket()

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _check_step_penalty(self, command: str) -> float:
        """Check if this specific step triggers a forbidden-action penalty."""
        if not self._scenario:
            return 0.0

        cmd_lower = command.strip().lower()
        penalty = 0.0

        for forbidden in self._scenario.forbidden_actions:
            atype = forbidden.action_type.lower()
            if not cmd_lower.startswith(atype):
                continue

            # Check if specific params are required for the match
            if forbidden.params:
                params_match = all(
                    v.lower() in cmd_lower
                    for v in forbidden.params.values()
                )
                if params_match:
                    penalty += forbidden.penalty
            else:
                # No params required — any use of this action type is forbidden
                penalty += forbidden.penalty

        return penalty

    def _build_observation(
        self,
        action_result: str,
        reward: float,
        done: bool = False,
    ) -> Dict[str, Any]:
        """Build the observation dict returned to the client."""
        sc = self._scenario
        return {
            "observation": {
                "ticket_id": sc.ticket_id if sc else "",
                "ticket_status": "closed" if (self._backend and self._backend.ticket_closed) else "open",
                "customer_message": sc.customer_message if sc else "",
                "action_result": action_result,
                "actions_taken": list(self._actions_taken),
                "resolution_progress": self._prev_progress,
                "steps_taken": self._step_count,
                "max_steps": sc.max_steps if sc else 10,
                "task_description": sc.task_description if sc else "",
                "available_commands": AVAILABLE_COMMANDS,
                "last_action_error": self._last_error,
                "done": done,
                "reward": reward,
                "metadata": {},
            },
            "reward": reward,
            "done": done,
        }
