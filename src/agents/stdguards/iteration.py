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
        if agent is not None and hasattr(agent, "memory") and agent.memory.last:
            turns = sum(1 for m in agent.memory.messages if m.role == "assistant")
            self.iterations = turns
            
            # Determine if we are at the start of an iteration (before model call)
            # or in the middle of an iteration (after assistant responds, before tools run)
            if agent.memory.last.role == "assistant":
                # Middle of iteration: assistant has responded. Limit is exceeded if turns > max_iterations.
                limit_exceeded = turns > self.max_iterations
            else:
                # Start of iteration: checking completed turns. Limit is exceeded if turns >= max_iterations.
                limit_exceeded = turns >= self.max_iterations
        else:
            self.iterations += 1
            limit_exceeded = self.iterations > self.max_iterations

        if limit_exceeded:
            err = MaxIterationsReached(
                f"Agent stopped: maximum iteration limit of {self.max_iterations} reached."
            )
            log_guard_trigger(self, str(err), agent=agent, status="failed")
            raise err

    def __call__(self, agent: Any = None) -> None:
        """Allows using the guard as a callback function."""
        self.check(agent=agent)
