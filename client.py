# SPDX-License-Identifier: BSD-3-Clause

"""Math Arena Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import MathArenaAction, MathArenaObservation


class MathArenaEnv(
    EnvClient[MathArenaAction, MathArenaObservation, State]
):
    """
    Client for the Math Arena Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with MathArenaEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(MathArenaAction(command="calc", text="12*24"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect (.sync() for sync use)
        >>> client = MathArenaEnv.from_docker_image("math_arena-env:latest").sync()
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(MathArenaAction(command="submit", text="42"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: MathArenaAction) -> Dict:
        """
        Convert MathArenaAction to JSON payload for step message.

        Args:
            action: MathArenaAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "command": action.command,
            "text": action.text,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MathArenaObservation]:
        """
        Parse server response into StepResult[MathArenaObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with MathArenaObservation
        """
        obs_data = payload.get("observation", {})
        observation = MathArenaObservation(
            message=obs_data.get("message", ""),
            moves_left=obs_data.get("moves_left", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=payload.get("metadata", obs_data.get("metadata", {})),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=payload.get("metadata"),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
