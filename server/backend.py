"""
SupportEnv — Simulated Company Backend

Provides in-memory order database, customer database, knowledge base,
and company policies that the agent can query during ticket resolution.
Also handles action execution (refunds, password resets, etc.).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data models for the simulated company
# ---------------------------------------------------------------------------

@dataclass
class OrderItem:
    name: str
    sku: str
    price: float
    quantity: int = 1


@dataclass
class Order:
    order_id: str
    customer_id: str
    items: List[OrderItem]
    total: float
    status: str  # pending, shipped, delivered, returned, refunded
    tracking_number: Optional[str] = None
    order_date: str = ""
    delivery_date: Optional[str] = None
    shipping_method: str = "standard"
    refunded: bool = False
    refund_amount: float = 0.0
    notes: List[str] = field(default_factory=list)


@dataclass
class Customer:
    customer_id: str
    name: str
    email: str
    membership: str  # standard, silver, gold, platinum
    account_since: str
    previous_tickets: int = 0
    lifetime_value: float = 0.0
    password_reset_pending: bool = False
    discount_applied: float = 0.0
    notes: List[str] = field(default_factory=list)


@dataclass
class KBArticle:
    article_id: str
    title: str
    content: str
    tags: List[str]


@dataclass
class Policy:
    name: str
    title: str
    content: str


# ---------------------------------------------------------------------------
# Backend class — holds all data and processes actions
# ---------------------------------------------------------------------------

class CompanyBackend:
    """Simulated company backend with orders, customers, KB, and policies."""

    def __init__(self) -> None:
        self.orders: Dict[str, Order] = {}
        self.customers: Dict[str, Customer] = {}
        self.kb_articles: List[KBArticle] = []
        self.policies: Dict[str, Policy] = {}

        # Action log — records everything the agent did
        self.action_log: List[Dict[str, Any]] = []

        # Tracks what the agent has done (for grading)
        self.refunds_issued: Dict[str, float] = {}  # order_id -> amount
        self.replacements_sent: List[str] = []  # order_ids
        self.passwords_reset: List[str] = []  # customer_ids
        self.discounts_applied: Dict[str, float] = {}  # customer_id -> percent
        self.escalations: List[Dict[str, str]] = []  # [{department, reason}]
        self.status_updates: Dict[str, str] = {}  # order_id -> new_status
        self.internal_notes: List[str] = []
        self.response_sent: Optional[str] = None
        self.ticket_closed: bool = False

    def deep_copy(self) -> "CompanyBackend":
        """Return a deep copy for clean reset."""
        return copy.deepcopy(self)

    # -------------------------------------------------------------------
    # Query methods
    # -------------------------------------------------------------------

    def lookup_order(self, order_id: str) -> str:
        """Look up an order by ID. Returns formatted string."""
        order = self.orders.get(order_id)
        if not order:
            return f"❌ Order '{order_id}' not found. Available orders: {', '.join(self.orders.keys())}"

        items_str = "\n".join(
            f"    - {item.name} (SKU: {item.sku}) × {item.quantity} — ${item.price:.2f}"
            for item in order.items
        )
        result = (
            f"📦 Order Details — {order.order_id}\n"
            f"  Customer: {order.customer_id}\n"
            f"  Items:\n{items_str}\n"
            f"  Total: ${order.total:.2f}\n"
            f"  Status: {order.status}\n"
            f"  Order Date: {order.order_date}\n"
        )
        if order.tracking_number:
            result += f"  Tracking: {order.tracking_number}\n"
        if order.delivery_date:
            result += f"  Delivery Date: {order.delivery_date}\n"
        result += f"  Shipping: {order.shipping_method}\n"
        if order.refunded:
            result += f"  ⚠️ Refund issued: ${order.refund_amount:.2f}\n"
        if order.notes:
            result += f"  Notes: {'; '.join(order.notes)}\n"
        return result.strip()

    def lookup_customer(self, customer_id: str) -> str:
        """Look up a customer by ID."""
        cust = self.customers.get(customer_id)
        if not cust:
            return f"❌ Customer '{customer_id}' not found. Available: {', '.join(self.customers.keys())}"

        result = (
            f"👤 Customer Profile — {cust.customer_id}\n"
            f"  Name: {cust.name}\n"
            f"  Email: {cust.email}\n"
            f"  Membership: {cust.membership}\n"
            f"  Account Since: {cust.account_since}\n"
            f"  Previous Support Tickets: {cust.previous_tickets}\n"
            f"  Lifetime Value: ${cust.lifetime_value:.2f}\n"
        )
        if cust.password_reset_pending:
            result += "  ⚠️ Password reset already pending\n"
        if cust.discount_applied > 0:
            result += f"  💰 Active discount: {cust.discount_applied}%\n"
        if cust.notes:
            result += f"  Notes: {'; '.join(cust.notes)}\n"
        return result.strip()

    def search_kb(self, query: str) -> str:
        """Search knowledge base by keyword matching."""
        query_lower = query.lower()
        matches = []
        for article in self.kb_articles:
            score = 0
            if query_lower in article.title.lower():
                score += 3
            if query_lower in article.content.lower():
                score += 1
            if any(query_lower in tag.lower() for tag in article.tags):
                score += 2
            if score > 0:
                matches.append((score, article))

        matches.sort(key=lambda x: x[0], reverse=True)
        if not matches:
            return f"📚 No knowledge base articles found for '{query}'.\nAvailable topics: returns, shipping, refunds, passwords, loyalty, warranty, escalation"

        results = [f"📚 Knowledge Base — Search results for '{query}':\n"]
        for _, article in matches[:3]:
            results.append(f"  📄 {article.title}\n     {article.content}\n")
        return "\n".join(results).strip()

    def check_policy(self, policy_name: str) -> str:
        """Look up a specific company policy."""
        policy = self.policies.get(policy_name)
        if not policy:
            available = ", ".join(self.policies.keys())
            return f"❌ Policy '{policy_name}' not found.\nAvailable policies: {available}"

        return f"📋 Policy: {policy.title}\n{policy.content}"

    # -------------------------------------------------------------------
    # Action methods
    # -------------------------------------------------------------------

    def issue_refund(self, order_id: str, amount_str: str) -> str:
        """Issue a refund for an order."""
        order = self.orders.get(order_id)
        if not order:
            return f"❌ Cannot issue refund — order '{order_id}' not found."

        try:
            amount = float(amount_str.replace("$", "").replace(",", ""))
        except (ValueError, TypeError):
            return f"❌ Invalid refund amount: '{amount_str}'. Provide a numeric value."

        if amount <= 0:
            return f"❌ Refund amount must be positive. Got: {amount}"
        if amount > order.total:
            amount = order.total

        order.refunded = True
        order.refund_amount = amount
        order.status = "refunded"
        self.refunds_issued[order_id] = amount

        self.action_log.append({
            "type": "refund",
            "order_id": order_id,
            "amount": amount,
        })
        return f"✅ Refund of ${amount:.2f} issued for order {order_id}. Customer will receive the refund in 3–5 business days."

    def send_replacement(self, order_id: str) -> str:
        """Send a replacement for an order."""
        order = self.orders.get(order_id)
        if not order:
            return f"❌ Cannot send replacement — order '{order_id}' not found."

        order.status = "replacement_shipped"
        self.replacements_sent.append(order_id)

        self.action_log.append({
            "type": "replacement",
            "order_id": order_id,
        })
        items = ", ".join(f"{item.name}" for item in order.items)
        return f"✅ Replacement shipment initiated for order {order_id} ({items}). Estimated delivery: 3–5 business days."

    def reset_password(self, customer_id: str) -> str:
        """Reset a customer's password."""
        cust = self.customers.get(customer_id)
        if not cust:
            return f"❌ Cannot reset password — customer '{customer_id}' not found."

        cust.password_reset_pending = True
        self.passwords_reset.append(customer_id)

        self.action_log.append({
            "type": "password_reset",
            "customer_id": customer_id,
        })
        return f"✅ Password reset link sent to {cust.email}. The link is valid for 24 hours."

    def apply_discount(self, customer_id: str, percent_str: str) -> str:
        """Apply a goodwill discount to a customer's account."""
        cust = self.customers.get(customer_id)
        if not cust:
            return f"❌ Cannot apply discount — customer '{customer_id}' not found."

        try:
            percent = float(percent_str.replace("%", ""))
        except (ValueError, TypeError):
            return f"❌ Invalid discount percentage: '{percent_str}'."

        if percent <= 0 or percent > 50:
            return f"❌ Discount must be between 1% and 50%. Got: {percent}%"

        cust.discount_applied = percent
        self.discounts_applied[customer_id] = percent

        self.action_log.append({
            "type": "discount",
            "customer_id": customer_id,
            "percent": percent,
        })
        return f"✅ {percent}% goodwill discount applied to customer {customer_id} ({cust.name})'s account."

    def escalate(self, department: str, reason: str) -> str:
        """Escalate the ticket to a specialist department."""
        valid_departments = ["billing", "technical", "management", "shipping", "legal"]
        dept_lower = department.lower()
        if dept_lower not in valid_departments:
            return f"❌ Invalid department '{department}'. Valid: {', '.join(valid_departments)}"

        self.escalations.append({
            "department": dept_lower,
            "reason": reason,
        })

        self.action_log.append({
            "type": "escalation",
            "department": dept_lower,
            "reason": reason,
        })
        return f"✅ Ticket escalated to {dept_lower.title()} department. Reason: {reason}. A specialist will follow up within 24 hours."

    def update_status(self, order_id: str, new_status: str) -> str:
        """Update the status of an order."""
        order = self.orders.get(order_id)
        if not order:
            return f"❌ Cannot update status — order '{order_id}' not found."

        valid_statuses = ["pending", "processing", "shipped", "delivered", "returning", "returned", "cancelled"]
        if new_status.lower() not in valid_statuses:
            return f"❌ Invalid status '{new_status}'. Valid: {', '.join(valid_statuses)}"

        old_status = order.status
        order.status = new_status.lower()
        self.status_updates[order_id] = new_status.lower()

        self.action_log.append({
            "type": "status_update",
            "order_id": order_id,
            "old_status": old_status,
            "new_status": new_status.lower(),
        })
        return f"✅ Order {order_id} status updated: {old_status} → {new_status.lower()}"

    def add_note(self, note: str) -> str:
        """Add an internal note to the ticket."""
        self.internal_notes.append(note)
        self.action_log.append({"type": "note", "note": note})
        return f"📝 Internal note added: {note}"

    def send_response(self, message: str) -> str:
        """Send a response message to the customer."""
        if not message or len(message.strip()) < 10:
            return "❌ Response is too short. Please provide a meaningful reply to the customer."

        self.response_sent = message.strip()
        self.action_log.append({"type": "response", "message": message.strip()})
        return f"✅ Response sent to customer ({len(message.strip())} characters)."

    def close_ticket(self) -> str:
        """Close the current ticket."""
        self.ticket_closed = True
        self.action_log.append({"type": "close_ticket"})
        return "✅ Ticket closed."


# ---------------------------------------------------------------------------
# Factory: build populated backends for each scenario
# ---------------------------------------------------------------------------

def build_backend_for_task(task_name: str) -> CompanyBackend:
    """Create and populate a CompanyBackend for the given task."""
    backend = CompanyBackend()
    _populate_shared_policies(backend)
    _populate_shared_kb(backend)

    if task_name == "simple_inquiry":
        _populate_simple_inquiry(backend)
    elif task_name == "complaint_resolution":
        _populate_complaint_resolution(backend)
    elif task_name == "complex_escalation":
        _populate_complex_escalation(backend)
    else:
        raise ValueError(f"Unknown task: {task_name}")

    return backend


def _populate_shared_policies(b: CompanyBackend) -> None:
    b.policies["refund_policy"] = Policy(
        name="refund_policy",
        title="Refund & Returns Policy",
        content=(
            "• Full refund within 30 days of delivery for unused/defective items.\n"
            "• Partial refund (50%) for items returned within 31–60 days.\n"
            "• No refund after 60 days.\n"
            "• Wrong item shipped: full refund + free return shipping.\n"
            "• Damaged in transit: full refund or free replacement (customer's choice).\n"
            "• Refunds processed within 3–5 business days."
        ),
    )
    b.policies["shipping_policy"] = Policy(
        name="shipping_policy",
        title="Shipping Policy",
        content=(
            "• Standard shipping: 3–5 business days.\n"
            "• Express shipping: 1–2 business days.\n"
            "• Free shipping on orders over $50.\n"
            "• Tracking available for all shipments.\n"
            "• International shipping: 7–14 business days."
        ),
    )
    b.policies["escalation_policy"] = Policy(
        name="escalation_policy",
        title="Escalation Policy",
        content=(
            "• Escalate to Billing: duplicate charges, billing disputes, payment failures.\n"
            "• Escalate to Technical: account lockouts, system bugs, app issues.\n"
            "• Escalate to Management: VIP complaints, legal threats, 3+ repeat issues.\n"
            "• Escalate to Shipping: lost packages, carrier disputes.\n"
            "• Specialist follow-up within 24 hours."
        ),
    )
    b.policies["discount_policy"] = Policy(
        name="discount_policy",
        title="Goodwill Discount Policy",
        content=(
            "• Max 20% goodwill discount for first-time issues.\n"
            "• Max 30% for repeat issues.\n"
            "• VIP/Gold/Platinum members: up to 25% for first issue.\n"
            "• Discounts applied to next order automatically.\n"
            "• Manager approval required for discounts above 30%."
        ),
    )
    b.policies["warranty_policy"] = Policy(
        name="warranty_policy",
        title="Warranty & Damage Policy",
        content=(
            "• Electronics: 1-year manufacturer warranty.\n"
            "• Clothing & accessories: 90-day quality guarantee.\n"
            "• Damaged-in-transit claims must be filed within 48 hours of delivery.\n"
            "• Photos of damage required for claims over $100.\n"
            "• Replacement or refund at customer's choice for valid claims."
        ),
    )


def _populate_shared_kb(b: CompanyBackend) -> None:
    b.kb_articles = [
        KBArticle(
            article_id="KB-001",
            title="How to Process a Return",
            content="To begin a return: 1) Look up the order, 2) Verify within return window, 3) Issue return label, 4) Process refund once item received. For wrong-item cases, issue immediate refund.",
            tags=["return", "refund", "wrong item"],
        ),
        KBArticle(
            article_id="KB-002",
            title="Shipping & Tracking FAQ",
            content="Standard delivery is 3–5 business days. Express is 1–2 days. Tracking updates may take 24 hours after shipment. If no update after 48 hours, contact shipping department.",
            tags=["shipping", "tracking", "delivery"],
        ),
        KBArticle(
            article_id="KB-003",
            title="Password & Account Recovery",
            content="To reset a customer's password: use reset_password command. Reset link valid for 24 hours. If emails not arriving, check spam folder. For persistent issues, escalate to Technical.",
            tags=["password", "account", "reset", "login"],
        ),
        KBArticle(
            article_id="KB-004",
            title="Handling Damaged Items",
            content="For items damaged in transit: 1) Express sympathy, 2) Check warranty policy, 3) Offer replacement or refund (customer's choice), 4) No need for return of damaged goods under $50.",
            tags=["damage", "warranty", "broken", "transit"],
        ),
        KBArticle(
            article_id="KB-005",
            title="Dealing with Billing Disputes",
            content="Duplicate charges and billing discrepancies must be escalated to the Billing department. Do NOT attempt manual billing adjustments. Provide customer with escalation reference.",
            tags=["billing", "charge", "duplicate", "payment"],
        ),
        KBArticle(
            article_id="KB-006",
            title="Loyalty Program Benefits",
            content="Gold members: 10% ongoing discount, priority support. Platinum: 15% discount, free express shipping, dedicated account manager. Silver: 5% discount on orders over $100.",
            tags=["loyalty", "gold", "platinum", "membership", "VIP"],
        ),
    ]


def _populate_simple_inquiry(b: CompanyBackend) -> None:
    """Task 1: Simple order status inquiry."""
    b.orders["ORD-1001"] = Order(
        order_id="ORD-1001",
        customer_id="CUST-201",
        items=[
            OrderItem(name="Wireless Bluetooth Headphones", sku="WBH-200", price=49.99),
        ],
        total=54.98,
        status="shipped",
        tracking_number="TRK-88421-US",
        order_date="2024-03-10",
        delivery_date=None,
        shipping_method="standard",
    )
    b.customers["CUST-201"] = Customer(
        customer_id="CUST-201",
        name="Alex Chen",
        email="alex.chen@email.com",
        membership="silver",
        account_since="2023-05-20",
        previous_tickets=0,
        lifetime_value=320.00,
    )


def _populate_complaint_resolution(b: CompanyBackend) -> None:
    """Task 2: Wrong item complaint with refund."""
    b.orders["ORD-2567"] = Order(
        order_id="ORD-2567",
        customer_id="CUST-315",
        items=[
            OrderItem(name="Blue XL T-Shirt", sku="TSH-BLU-XL", price=35.99),
        ],
        total=41.98,
        status="delivered",
        tracking_number="TRK-77203-US",
        order_date="2024-03-05",
        delivery_date="2024-03-09",
        shipping_method="standard",
    )
    b.customers["CUST-315"] = Customer(
        customer_id="CUST-315",
        name="Maria Rodriguez",
        email="maria.rod@email.com",
        membership="gold",
        account_since="2022-11-10",
        previous_tickets=1,
        lifetime_value=890.50,
    )


def _populate_complex_escalation(b: CompanyBackend) -> None:
    """Task 3: Multiple issues requiring investigation and escalation."""
    # Damaged laptop order
    b.orders["ORD-3100"] = Order(
        order_id="ORD-3100",
        customer_id="CUST-450",
        items=[
            OrderItem(name='ProBook 15" Laptop', sku="LAP-PB15", price=1299.99),
            OrderItem(name="Laptop Sleeve Case", sku="SLV-15", price=29.99),
        ],
        total=1379.97,
        status="delivered",
        tracking_number="TRK-99101-US",
        order_date="2024-02-20",
        delivery_date="2024-02-25",
        shipping_method="express",
    )
    # Double-charged order
    b.orders["ORD-3205"] = Order(
        order_id="ORD-3205",
        customer_id="CUST-450",
        items=[
            OrderItem(name="USB-C Hub Adapter", sku="USB-HUB-7", price=45.99),
            OrderItem(name="Wireless Mouse", sku="WM-ERG-01", price=34.99),
        ],
        total=86.97,
        status="delivered",
        tracking_number="TRK-99205-US",
        order_date="2024-03-01",
        delivery_date="2024-03-05",
        shipping_method="standard",
        notes=["⚠️ BILLING ALERT: Duplicate charge detected — $86.97 charged twice on 2024-03-01."],
    )
    b.customers["CUST-450"] = Customer(
        customer_id="CUST-450",
        name="James Whitfield",
        email="j.whitfield@email.com",
        membership="platinum",
        account_since="2020-08-15",
        previous_tickets=3,
        lifetime_value=4280.00,
    )
