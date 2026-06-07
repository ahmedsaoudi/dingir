from typing import Any, Optional

from dingir.agents.guards import Guard, GuardError, log_guard_trigger


class MaxIterationsReached(GuardError):
    """Exception raised when an agent exceeds the maximum allowed iterations."""
    pass


class IterationGuard(Guard):
    """Guard to track and limit iterations, usable directly or as a step callback."""

    def __init__(self, max_iterations: int, log: bool = True):
        super().__init__(log=log)
        self.max_iterations = max_iterations
        self.iterations = 0

    def check(self, agent: Optional[Any] = None) -> None:
        """Increment the iteration count and raise MaxIterationsReached if the limit is exceeded."""
        self.iterations += 1
        if self.iterations > self.max_iterations:
            err = MaxIterationsReached(
                f"Agent stopped: maximum iteration limit of {self.max_iterations} reached."
            )
            log_guard_trigger(self, err, agent=agent)
            raise err

    def __call__(self, agent: Any = None) -> None:
        """Allows using the guard as a callback function."""
        self.check(agent=agent)
