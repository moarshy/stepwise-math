# SPDX-License-Identifier: BSD-3-Clause

"""
Math Arena environment.

reset(task_id) draws a fresh problem from that task's generator, step() answers
calculator requests, and a submit is graded against the exact answer.
"""

import ast
import operator
import random
from fractions import Fraction
from typing import Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MathArenaAction, MathArenaObservation
    from .generators import GENERATORS, fmt
except ImportError:
    from models import MathArenaAction, MathArenaObservation
    from server.generators import GENERATORS, fmt

DEFAULT_TASK = "counting-medium"
MAX_MOVES = 20
MAX_EXPONENT = 64  # big powers must be reduced by reasoning, not brute-forced in calc
MAX_DIGITS = 200

INSTRUCTIONS = (
    'Reply with one JSON object. Use {"command": "calc", "text": "<expression>"} for exact arithmetic '
    '(+ - * / // % ** and brackets; exponents up to 64), and {"command": "submit", "text": "<answer>"} '
    'once with your final answer as an integer or reduced fraction like 7/12.'
)

_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _eval(node) -> Fraction:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            if right.denominator != 1 or abs(right) > MAX_EXPONENT:
                raise ValueError(f"exponent must be an integer with absolute value at most {MAX_EXPONENT}")
            if left == 0 and right < 0:
                raise ValueError("division by zero")
        result = _BIN[type(node.op)](left, right)
        if len(str(abs(result.numerator))) > MAX_DIGITS:
            raise ValueError("result too large")
        return result
    raise ValueError("only numbers, + - * / // % ** and brackets are allowed")


def calculate(expr: str) -> str:
    """Exact calculator: fractions stay fractions, so 1/3 + 1/6 = 1/2."""
    try:
        result = _eval(ast.parse(expr.strip(), mode="eval").body)
        return f"{expr.strip()} = {fmt(result)}"
    except ZeroDivisionError:
        return "calculator error: division by zero"
    except (SyntaxError, ValueError, TypeError) as e:
        return f"calculator error: {e}"


def parse_answer(text: str) -> Optional[Fraction]:
    """Read "42", "-7/12", "0.375" as an exact number; anything else is unreadable."""
    try:
        return Fraction(text.strip().replace(" ", ""))
    except (ValueError, ZeroDivisionError):
        return None


class MathArenaEnvironment(Environment):
    """A math problem with an exact calculator and one graded final answer."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._answer = Fraction(0)

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None,
              task_id: Optional[str] = None, **kwargs) -> MathArenaObservation:
        """Draw a fresh problem for the task and ask it."""
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        generator = GENERATORS.get(task_id or DEFAULT_TASK, GENERATORS[DEFAULT_TASK])
        question, self._answer = generator(random.Random(seed))
        return MathArenaObservation(
            message=f"{question}\n\n{INSTRUCTIONS}",
            moves_left=MAX_MOVES, done=False, reward=0.0,
        )

    def step(self, action: MathArenaAction) -> MathArenaObservation:  # type: ignore[override]
        """Answer a calculator request, or grade the final answer."""
        self._state.step_count += 1
        moves_left = MAX_MOVES - self._state.step_count

        if action.command == "submit":
            correct = parse_answer(action.text) == self._answer
            return MathArenaObservation(
                message="Correct." if correct else "Incorrect.",
                moves_left=moves_left, done=True, reward=1.0 if correct else 0.0,
            )

        if moves_left <= 0:
            return MathArenaObservation(message="Out of moves.", moves_left=0, done=True, reward=0.0)
        return MathArenaObservation(message=calculate(action.text), moves_left=moves_left, done=False, reward=0.0)

    @property
    def state(self) -> State:
        return self._state
