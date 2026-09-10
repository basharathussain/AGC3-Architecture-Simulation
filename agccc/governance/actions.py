"""The decision space.

These are *structural* actions, not agent names. Selecting one produces a graph
transformation (see `strategies.adaptive`), which is the difference between
adaptation and "choose another agent" (Raw_planning §12 Guideline 3).
"""

from __future__ import annotations

from enum import Enum


class Action(str, Enum):
    ACCEPT = "ACCEPT"                    # release the artefact and finish
    RETRY_CODER = "RETRY_CODER"          # re-enter the Coder, then re-verify
    INVOKE_SECURITY = "INVOKE_SECURITY"  # splice the specialist in, then re-verify
    REPLAN = "REPLAN"                    # re-enter the Planner
    TERMINATE = "TERMINATE"              # give up, safely


ALL_ACTIONS: tuple[Action, ...] = tuple(Action)
