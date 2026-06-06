from typing import Any, Callable

from dingir.agents.guards import Guard


class IfGuard(Guard):
    """Conditional guard that only runs an inner guard if a condition is met.

    Example:
         # Stop agent after 5 iterations only if the last message contains "help"
         guard = IfGuard(
             condition=lambda agent: "help" in (agent.memory.last.content if agent and agent.memory.last else ""),
             guard=IterationGuard(5)
         )
    """

    def __init__(self, condition: Callable[[Any], bool], guard: Callable[[Any], None]):
        self.condition = condition
        self.guard = guard

    def __call__(self, agent: Any = None) -> None:
        if self.condition(agent):
            self.guard(agent)
