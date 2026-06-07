from typing import Any, Dict, Optional

from dingir.agents.guards import Guard, GuardError


class MaxIterationsReached(GuardError):
    """Exception raised when an agent exceeds the maximum allowed iterations."""

    pass


class IterationGuard(Guard):
    """Guard to track and limit agent iterations.

    Usage::

        agent.respond(msg, on_step_callback=IterationGuard(max_iterations=5))
    """

    def __init__(self, max_iterations: int, log: bool = True):
        super().__init__(log=log)
        self.max_iterations = max_iterations
        self.iterations = 0

    def check_tool_args(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> None:
        """Delegate validation to check_step using the active agent if available."""
        from dingir.agents.guards import _active_agent
        agent = _active_agent.get()
        self.check_step(agent)

    def check_step(self, agent: Optional[Any] = None) -> None:
        """Increment the iteration count and raise if the limit is exceeded."""
        if (
            agent is not None
            and hasattr(agent, "memory")
            and agent.memory.last
        ):
            turns = sum(
                1 for m in agent.memory.messages if m.role == "assistant"
            )
            self.iterations = turns

            # Determine if we are at the start of an iteration (before model
            # call) or in the middle (after assistant responds, before tools).
            if agent.memory.last.role == "assistant":
                limit_exceeded = turns > self.max_iterations
            else:
                limit_exceeded = turns >= self.max_iterations
        else:
            self.iterations += 1
            limit_exceeded = self.iterations > self.max_iterations

        if limit_exceeded:
            raise MaxIterationsReached(
                f"Agent stopped: maximum iteration limit of "
                f"{self.max_iterations} reached."
            )
