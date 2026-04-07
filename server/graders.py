"""
SupportEnv — Deterministic Graders

Evaluates agent performance on each task by checking:
1. Action correctness — did the agent take the required actions? (40%)
2. Response quality — does the response contain required information? (40%)
3. Efficiency — how many steps vs. optimal path? (20%)

All grading is deterministic — no LLM-based evaluation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .backend import CompanyBackend
from .scenarios import (
    ForbiddenAction,
    RequiredAction,
    RequiredResponseElement,
    Scenario,
)


# ---------------------------------------------------------------------------
# Main grader
# ---------------------------------------------------------------------------

class TaskGrader:
    """Deterministic grader for a customer-support scenario."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario

    def grade(
        self,
        backend: CompanyBackend,
        actions_taken: List[str],
        steps_taken: int,
    ) -> Dict:
        """
        Grade the agent's performance.

        Returns a dict with:
            score: float (0.0–1.0)
            action_score: float (0.0–1.0)
            response_score: float (0.0–1.0)
            efficiency_score: float (0.0–1.0)
            details: dict with breakdown
        """
        action_score, action_details = self._grade_actions(backend, actions_taken)
        response_score, response_details = self._grade_response(backend)
        efficiency_score = self._grade_efficiency(steps_taken)
        penalty = self._compute_penalties(backend, actions_taken)

        # Weighted combination
        raw_score = (
            action_score * 0.40
            + response_score * 0.40
            + efficiency_score * 0.20
        )
        final_score = max(0.0, min(1.0, raw_score - penalty))

        return {
            "score": round(final_score, 4),
            "action_score": round(action_score, 4),
            "response_score": round(response_score, 4),
            "efficiency_score": round(efficiency_score, 4),
            "penalty": round(penalty, 4),
            "details": {
                "actions": action_details,
                "response": response_details,
                "steps_taken": steps_taken,
                "optimal_steps": self.scenario.optimal_steps,
                "max_steps": self.scenario.max_steps,
            },
        }

    def compute_progress(
        self,
        backend: CompanyBackend,
        actions_taken: List[str],
    ) -> float:
        """
        Compute resolution progress (0.0–1.0) for intermediate rewards.
        Simpler than full grading — just checks action/response completeness.
        """
        action_score, _ = self._grade_actions(backend, actions_taken)
        response_score, _ = self._grade_response(backend)

        # 50/50 weight for progress (efficiency not counted here)
        progress = action_score * 0.5 + response_score * 0.5
        return round(min(1.0, max(0.0, progress)), 4)

    # -------------------------------------------------------------------
    # Action grading
    # -------------------------------------------------------------------

    def _grade_actions(
        self,
        backend: CompanyBackend,
        actions_taken: List[str],
    ) -> Tuple[float, List[Dict]]:
        """Check which required actions the agent completed."""
        details = []
        total_weight = 0.0
        earned_weight = 0.0

        # Check required actions
        for req in self.scenario.required_actions:
            total_weight += req.weight
            matched = self._check_action_match(req, backend, actions_taken)
            earned = req.weight if matched else 0.0
            earned_weight += earned
            details.append({
                "description": req.description,
                "required": True,
                "completed": matched,
                "weight": req.weight,
            })

        # Check bonus actions
        for bonus in self.scenario.bonus_actions:
            matched = self._check_action_match(bonus, backend, actions_taken)
            if matched:
                earned_weight += bonus.weight * 0.5  # bonuses count at half
            details.append({
                "description": bonus.description,
                "required": False,
                "completed": matched,
                "weight": bonus.weight * 0.5,
            })

        # Check that response was sent and ticket closed (if required)
        if self.scenario.must_send_response:
            total_weight += 1.0
            sent = backend.response_sent is not None and len(backend.response_sent) > 10
            earned_weight += 1.0 if sent else 0.0
            details.append({
                "description": "Sent customer response",
                "required": True,
                "completed": sent,
                "weight": 1.0,
            })

        if self.scenario.must_close_ticket:
            total_weight += 0.5
            closed = backend.ticket_closed
            earned_weight += 0.5 if closed else 0.0
            details.append({
                "description": "Closed the ticket",
                "required": True,
                "completed": closed,
                "weight": 0.5,
            })

        score = earned_weight / total_weight if total_weight > 0 else 0.0
        return min(1.0, score), details

    def _check_action_match(
        self,
        req: RequiredAction,
        backend: CompanyBackend,
        actions_taken: List[str],
    ) -> bool:
        """Check if a required action was performed."""
        atype = req.action_type

        if atype == "lookup_order":
            order_id = req.params.get("order_id", "")
            return any(
                f"lookup_order" in a and order_id.lower() in a.lower()
                for a in actions_taken
            )

        elif atype == "lookup_customer":
            cust_id = req.params.get("customer_id", "")
            return any(
                "lookup_customer" in a and cust_id.lower() in a.lower()
                for a in actions_taken
            )

        elif atype == "issue_refund":
            order_id = req.params.get("order_id", "")
            return order_id in backend.refunds_issued

        elif atype == "send_replacement":
            order_id = req.params.get("order_id", "")
            return order_id in backend.replacements_sent

        elif atype == "reset_password":
            cust_id = req.params.get("customer_id", "")
            return cust_id in backend.passwords_reset

        elif atype == "apply_discount":
            cust_id = req.params.get("customer_id", "")
            if cust_id:
                return cust_id in backend.discounts_applied
            return len(backend.discounts_applied) > 0

        elif atype == "escalate":
            dept = req.params.get("department", "")
            if dept:
                return any(e["department"] == dept for e in backend.escalations)
            return len(backend.escalations) > 0

        elif atype == "check_policy":
            return any("check_policy" in a for a in actions_taken)

        elif atype == "search_kb":
            return any("search_kb" in a for a in actions_taken)

        return False

    # -------------------------------------------------------------------
    # Response grading
    # -------------------------------------------------------------------

    def _grade_response(
        self,
        backend: CompanyBackend,
    ) -> Tuple[float, List[Dict]]:
        """Check if the response contains required information."""
        response = backend.response_sent or ""
        response_lower = response.lower()
        details = []
        total_weight = 0.0
        earned_weight = 0.0

        for elem in self.scenario.required_response_elements:
            total_weight += elem.weight

            if elem.case_sensitive:
                matched = any(kw in response for kw in elem.keywords)
            else:
                matched = any(kw.lower() in response_lower for kw in elem.keywords)

            earned_weight += elem.weight if matched else 0.0
            details.append({
                "description": elem.description,
                "matched": matched,
                "weight": elem.weight,
            })

        score = earned_weight / total_weight if total_weight > 0 else 0.0
        return min(1.0, score), details

    # -------------------------------------------------------------------
    # Efficiency grading
    # -------------------------------------------------------------------

    def _grade_efficiency(self, steps_taken: int) -> float:
        """Score based on how close to optimal the agent was."""
        optimal = self.scenario.optimal_steps
        max_steps = self.scenario.max_steps

        if steps_taken <= optimal:
            return 1.0

        # Linear decay from optimal to max_steps
        extra = steps_taken - optimal
        max_extra = max_steps - optimal
        if max_extra <= 0:
            return 1.0

        return max(0.0, 1.0 - (extra / max_extra))

    # -------------------------------------------------------------------
    # Penalty computation
    # -------------------------------------------------------------------

    def _compute_penalties(
        self,
        backend: CompanyBackend,
        actions_taken: List[str],
    ) -> float:
        """Compute total penalty for forbidden actions."""
        total_penalty = 0.0

        for forbidden in self.scenario.forbidden_actions:
            if self._check_forbidden_match(forbidden, backend, actions_taken):
                total_penalty += forbidden.penalty

        return total_penalty

    def _check_forbidden_match(
        self,
        forbidden: ForbiddenAction,
        backend: CompanyBackend,
        actions_taken: List[str],
    ) -> bool:
        """Check if a forbidden action was taken."""
        atype = forbidden.action_type
        params = forbidden.params

        if atype == "issue_refund":
            order_id = params.get("order_id", "")
            if order_id:
                return order_id in backend.refunds_issued
            return len(backend.refunds_issued) > 0

        elif atype == "escalate":
            dept = params.get("department", "")
            if dept:
                return any(e["department"] == dept for e in backend.escalations)
            return len(backend.escalations) > 0

        elif atype == "send_replacement":
            order_id = params.get("order_id", "")
            if order_id:
                return order_id in backend.replacements_sent
            return len(backend.replacements_sent) > 0

        # Generic check — look for the action type in actions taken
        return any(atype in a for a in actions_taken)
