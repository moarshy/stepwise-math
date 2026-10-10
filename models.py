# SPDX-License-Identifier: BSD-3-Clause

"""
Data models for the Math Arena environment.

The model sees a math problem, may use an exact calculator, and submits one
final answer that is graded right (1) or wrong (0).
"""

from typing import Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class MathArenaAction(Action):
    """What the model sends: a calculator request or the final answer."""

    command: Literal["calc", "submit"] = Field(
        ..., description='"calc" evaluates an arithmetic expression; "submit" gives the final answer and ends the episode'
    )
    text: str = Field(
        ..., description='For calc: an expression using numbers, + - * / // % ** and brackets, e.g. "(3/4 + 2) * 5". For submit: an integer or reduced fraction, e.g. "42" or "7/12"'
    )


class MathArenaObservation(Observation):
    """What the environment sends back: the problem, a calculator result, or the grade."""

    message: str = Field(default="", description="The problem, the calculator result, or the grading feedback")
    moves_left: int = Field(default=0, description="Moves remaining before the episode ends with reward 0")
