from typing import Any, Callable

from dingir.agents.guards import Guard


class IfGuard(Guard):
    """Conditional guard that only runs an inner guard if a condition is met.

    The inner guard is invoked via its full ``__call__`` path so that
    its own logging and dispatch logic are preserved.

    Example::

        guard = IfGuard(
            condition=lambda agent: "help" in (
                agent.memory.last.content
                if agent and agent.memory.last else ""
            ),
            guard=IterationGuard(5),
        )
    """

    def __init__(
        self,
        condition: Callable[[Any], bool],
        guard: "Guard",
        log: bool = True,
    ):
        super().__init__(log=log)
        self.condition = condition
        self.inner_guard = guard

    def check_step(self, agent: Any = None) -> None:
        """Evaluate the condition and delegate to the inner guard if met."""
        if self.condition(agent):
            # Invoke via __call__ so the inner guard's auto-logging fires.
            self.inner_guard(agent)
