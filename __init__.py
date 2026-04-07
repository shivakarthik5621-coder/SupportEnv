"""
SupportEnv — Customer Support Ticket Resolution Environment

An OpenEnv environment where AI agents learn to resolve customer
support tickets through investigation, action, and response.
"""

from .models import SupportAction, SupportObservation, SupportState
from .client import SupportEnv

__all__ = [
    "SupportAction",
    "SupportObservation",
    "SupportState",
    "SupportEnv",
]
