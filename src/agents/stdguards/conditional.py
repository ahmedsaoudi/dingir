from typing import Any, Callable

from dingir.agents.guards import Guard


class IfGuard(Guard):
    """Conditional guard that only runs an inner guard if a condition is met.

    Example:
        # Stop agent after 5 iterations only if the last message contains "help"
        guard = IfGuard(
            condition=lambda chat: "help" in (chat.last_message.content if chat and chat.last_message else ""),
            guard=IterationGuard(5)
        )
    """

    def __init__(self, condition: Callable[[Any], bool], guard: Callable[[Any], None]):
        self.condition = condition
        self.guard = guard

    def __call__(self, chat: Any = None) -> None:
        if self.condition(chat):
            self.guard(chat)
