from typing import Any

from dingir.agents.guards import Guard, GuardError


class MaxIterationsReached(GuardError):
    """Exception raised when an agent exceeds the maximum allowed iterations."""
    pass


class IterationGuard(Guard):
    """Guard to track and limit iterations, usable directly or as a step callback."""

    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations
        self.iterations = 0

    def check(self) -> None:
        """Increment the iteration count and raise MaxIterationsReached if the limit is exceeded."""
        self.iterations += 1
        if self.iterations > self.max_iterations:
            raise MaxIterationsReached(
                f"Agent stopped: maximum iteration limit of {self.max_iterations} reached."
            )

    def __call__(self, chat: Any = None) -> None:
        """Allows using the guard as a callback function."""
        self.check()
