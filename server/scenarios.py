"""
SupportEnv — Ticket Scenarios & Solution Checklists

Each scenario defines:
- The customer's message
- Which task it belongs to
- The expected solution (required actions, required response elements)
- Forbidden actions
- Scoring weights
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class RequiredAction:
    """An action the agent MUST perform to get credit."""
    action_type: str  # e.g. "lookup_order", "issue_refund"
    params: Dict[str, str] = field(default_factory=dict)  # e.g. {"order_id": "ORD-1234"}
    description: str = ""  # human-readable description
    weight: float = 1.0  # relative weight in scoring
    # For flexible matching — any of these alternative params also count
    alternative_params: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class RequiredResponseElement:
    """A keyword/phrase that MUST appear in the agent's response."""
    keywords: List[str]  # any of these keywords must appear (OR logic)
    description: str = ""  # what this element represents
    weight: float = 1.0
    case_sensitive: bool = False


@dataclass
class ForbiddenAction:
    """An action the agent must NOT perform."""
    action_type: str
    params: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    penalty: float = 0.10  # penalty applied if this action is taken


@dataclass
class Scenario:
    """Complete ticket scenario with grading criteria."""
    task_name: str
    ticket_id: str
    customer_id: str
    customer_message: str
    task_description: str
    difficulty: str  # easy, medium, hard
    max_steps: int
    optimal_steps: int

    # Grading criteria
    required_actions: List[RequiredAction] = field(default_factory=list)
    required_response_elements: List[RequiredResponseElement] = field(default_factory=list)
    forbidden_actions: List[ForbiddenAction] = field(default_factory=list)
    bonus_actions: List[RequiredAction] = field(default_factory=list)

    # Whether the agent must send a response and close the ticket
    must_send_response: bool = True
    must_close_ticket: bool = True


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

def get_scenario(task_name: str) -> Scenario:
    """Return the scenario for a given task name."""
    scenarios = {
        "simple_inquiry": _simple_inquiry(),
        "complaint_resolution": _complaint_resolution(),
        "complex_escalation": _complex_escalation(),
    }
    if task_name not in scenarios:
        raise ValueError(
            f"Unknown task: '{task_name}'. Available: {list(scenarios.keys())}"
        )
    return scenarios[task_name]


def get_all_task_names() -> List[str]:
    """Return all available task names."""
    return ["simple_inquiry", "complaint_resolution", "complex_escalation"]


# ---------------------------------------------------------------------------
# Task 1: Simple Inquiry (Easy)
# ---------------------------------------------------------------------------

def _simple_inquiry() -> Scenario:
    return Scenario(
        task_name="simple_inquiry",
        ticket_id="TKT-5001",
        customer_id="CUST-201",
        customer_message=(
            "Hi, I placed order #ORD-1001 five days ago and haven't received "
            "any shipping updates. Can you let me know where my package is? "
            "I ordered Wireless Bluetooth Headphones."
        ),
        task_description=(
            "The customer wants to know the status of their order. "
            "Look up the order, find the tracking information, and respond "
            "with the current status and estimated delivery date."
        ),
        difficulty="easy",
        max_steps=10,
        optimal_steps=3,

        required_actions=[
            RequiredAction(
                action_type="lookup_order",
                params={"order_id": "ORD-1001"},
                description="Look up order ORD-1001 to find status and tracking",
                weight=1.0,
            ),
        ],

        required_response_elements=[
            RequiredResponseElement(
                keywords=["TRK-88421-US", "TRK-88421", "tracking"],
                description="Mention the tracking number",
                weight=1.0,
            ),
            RequiredResponseElement(
                keywords=["shipped", "on its way", "on the way", "in transit"],
                description="Confirm the order has been shipped",
                weight=1.0,
            ),
            RequiredResponseElement(
                keywords=["3-5", "3–5", "3 to 5", "business days", "delivery", "arrive"],
                description="Mention estimated delivery timeframe",
                weight=1.0,
            ),
        ],

        bonus_actions=[
            RequiredAction(
                action_type="search_kb",
                params={},  # any KB search is fine
                description="Searched knowledge base for shipping info",
                weight=0.5,
            ),
        ],

        forbidden_actions=[
            ForbiddenAction(
                action_type="issue_refund",
                description="Do not refund — this is just a status inquiry",
                penalty=0.15,
            ),
            ForbiddenAction(
                action_type="escalate",
                description="Do not escalate a simple status check",
                penalty=0.10,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Task 2: Complaint Resolution (Medium)
# ---------------------------------------------------------------------------

def _complaint_resolution() -> Scenario:
    return Scenario(
        task_name="complaint_resolution",
        ticket_id="TKT-5002",
        customer_id="CUST-315",
        customer_message=(
            "I'm really frustrated. I ordered a blue XL t-shirt (Order #ORD-2567) "
            "but received a red medium instead. This has happened before with your "
            "company. I want a full refund immediately. The item cost $35.99 plus "
            "shipping, totaling $41.98."
        ),
        task_description=(
            "The customer received the wrong item and demands a refund. "
            "You need to: 1) investigate the order, 2) verify the customer's "
            "history, 3) issue an appropriate refund, 4) send an empathetic "
            "response with refund details and return instructions."
        ),
        difficulty="medium",
        max_steps=15,
        optimal_steps=5,

        required_actions=[
            RequiredAction(
                action_type="lookup_order",
                params={"order_id": "ORD-2567"},
                description="Look up order ORD-2567 to verify the claim",
                weight=1.0,
            ),
            RequiredAction(
                action_type="lookup_customer",
                params={"customer_id": "CUST-315"},
                description="Check customer history for repeat issues",
                weight=1.0,
            ),
            RequiredAction(
                action_type="issue_refund",
                params={"order_id": "ORD-2567"},
                description="Issue refund for the wrong item",
                weight=1.5,
            ),
        ],

        required_response_elements=[
            RequiredResponseElement(
                keywords=["sorry", "apologize", "apologies", "regret", "understand your frustration"],
                description="Apologize to the customer",
                weight=1.0,
            ),
            RequiredResponseElement(
                keywords=["refund", "refunded", "money back"],
                description="Confirm the refund",
                weight=1.0,
            ),
            RequiredResponseElement(
                keywords=["41.98", "35.99", "$41", "$35"],
                description="Mention the refund amount",
                weight=0.8,
            ),
            RequiredResponseElement(
                keywords=["return", "send back", "ship back", "return label"],
                description="Provide return instructions",
                weight=1.0,
            ),
            RequiredResponseElement(
                keywords=["again", "repeat", "previous", "before", "happened before", "second time"],
                description="Acknowledge this is a repeat issue",
                weight=0.8,
            ),
        ],

        bonus_actions=[
            RequiredAction(
                action_type="apply_discount",
                params={"customer_id": "CUST-315"},
                description="Offered goodwill discount for the inconvenience",
                weight=0.5,
            ),
            RequiredAction(
                action_type="check_policy",
                params={},
                description="Checked refund/return policy",
                weight=0.3,
            ),
        ],

        forbidden_actions=[
            ForbiddenAction(
                action_type="escalate",
                params={"department": "management"},
                description="No escalation needed — agent can handle this directly",
                penalty=0.05,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Task 3: Complex Escalation (Hard)
# ---------------------------------------------------------------------------

def _complex_escalation() -> Scenario:
    return Scenario(
        task_name="complex_escalation",
        ticket_id="TKT-5003",
        customer_id="CUST-450",
        customer_message=(
            "I've had THREE problems this month and I'm seriously considering "
            "canceling my account:\n\n"
            "1. Order #ORD-3100 arrived damaged — the shipping box was completely "
            "crushed and my laptop screen is cracked. This is a $1,299.99 laptop!\n\n"
            "2. I was charged TWICE for order #ORD-3205 — check my billing. "
            "I see two charges of $86.97 on my credit card statement from March 1st.\n\n"
            "3. I've been trying to reset my password for a week and the reset "
            "emails never arrive. I had to use a friend's computer to write this.\n\n"
            "I want to speak to a manager. I've been a loyal Platinum customer "
            "for almost 4 years and this is unacceptable."
        ),
        task_description=(
            "The customer has THREE separate issues that all need to be addressed:\n"
            "1. Damaged laptop (ORD-3100) — investigate, arrange refund or replacement\n"
            "2. Duplicate charge (ORD-3205) — investigate, escalate to billing\n"
            "3. Password reset not working — reset their password\n"
            "You must investigate all issues, take appropriate actions, compose a "
            "comprehensive response addressing all three problems, and handle the "
            "escalation request appropriately."
        ),
        difficulty="hard",
        max_steps=25,
        optimal_steps=9,

        required_actions=[
            RequiredAction(
                action_type="lookup_order",
                params={"order_id": "ORD-3100"},
                description="Investigate the damaged laptop order",
                weight=1.0,
            ),
            RequiredAction(
                action_type="lookup_order",
                params={"order_id": "ORD-3205"},
                description="Investigate the double-charged order",
                weight=1.0,
            ),
            RequiredAction(
                action_type="lookup_customer",
                params={"customer_id": "CUST-450"},
                description="Check customer profile and history",
                weight=1.0,
            ),
            RequiredAction(
                action_type="issue_refund",
                params={"order_id": "ORD-3100"},
                description="Issue refund for the damaged laptop",
                weight=1.5,
                alternative_params=[],
            ),
            RequiredAction(
                action_type="escalate",
                params={"department": "billing"},
                description="Escalate duplicate charge to billing department",
                weight=1.5,
            ),
            RequiredAction(
                action_type="reset_password",
                params={"customer_id": "CUST-450"},
                description="Reset the customer's password",
                weight=1.0,
            ),
        ],

        required_response_elements=[
            RequiredResponseElement(
                keywords=["sorry", "apologize", "apologies", "regret", "understand"],
                description="Express empathy and apology",
                weight=1.0,
            ),
            RequiredResponseElement(
                keywords=["laptop", "ORD-3100", "damaged", "cracked"],
                description="Address the damaged laptop issue",
                weight=1.0,
            ),
            RequiredResponseElement(
                keywords=["refund", "refunded", "$1,299", "$1299", "1299.99", "1379"],
                description="Confirm refund for damaged item",
                weight=1.0,
            ),
            RequiredResponseElement(
                keywords=["duplicate", "charged twice", "double charge", "billing", "ORD-3205"],
                description="Address the billing / duplicate charge issue",
                weight=1.0,
            ),
            RequiredResponseElement(
                keywords=["escalat", "billing team", "billing department", "specialist"],
                description="Confirm escalation of billing issue",
                weight=0.8,
            ),
            RequiredResponseElement(
                keywords=["password", "reset", "email", "link"],
                description="Address the password reset issue",
                weight=0.8,
            ),
            RequiredResponseElement(
                keywords=["loyal", "platinum", "valued", "years", "appreciate"],
                description="Acknowledge customer loyalty",
                weight=0.8,
            ),
        ],

        bonus_actions=[
            RequiredAction(
                action_type="apply_discount",
                params={"customer_id": "CUST-450"},
                description="Offered goodwill discount for the trouble",
                weight=0.5,
            ),
            RequiredAction(
                action_type="check_policy",
                params={},
                description="Checked warranty/damage policy",
                weight=0.3,
            ),
            RequiredAction(
                action_type="send_replacement",
                params={"order_id": "ORD-3100"},
                description="Sent replacement instead of refund (also valid)",
                weight=0.5,
            ),
        ],

        forbidden_actions=[
            ForbiddenAction(
                action_type="issue_refund",
                params={"order_id": "ORD-3205"},
                description="Do NOT refund ORD-3205 directly — billing dispute must be escalated",
                penalty=0.10,
            ),
        ],
    )
