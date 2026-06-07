from typing import Any, Dict

from dingir.agents.guards import (
    Guard,
    GuardError,
    get_approval_handler,
    log_guard_trigger,
)


class SecureAction(Guard):
    """Security middleware for human-in-the-loop approval.

    Works in all 3 modes by implementing just ``check_tool_args``:

    1. Step callback::

        on_step_callback=SecureAction(require_approval=True)

    2. Decorator::

        @SecureAction(require_approval=True)
        def dangerous_tool(cmd: str): ...

    3. guard_tool::

        guard_tool(dangerous_tool, SecureAction(require_approval=True))
    """

    def __init__(self, require_approval: bool = True, log: bool = True):
        super().__init__(log=log)
        self.require_approval = require_approval

    def check_tool_args(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> None:
        """Request human approval for the tool call."""
        if self.require_approval:
            handler = get_approval_handler()
            prompt = (
                f"Authorization requested for tool execution: '{tool_name}'"
            )
            approved = handler(prompt, payload=arguments)
            if approved:
                log_guard_trigger(
                    self,
                    f"Authorization approved for tool execution: "
                    f"'{tool_name}'",
                    tool_name=tool_name,
                    arguments=arguments,
                    status="approved",
                )
            else:
                raise GuardError(
                    "EXECUTION BLOCKED: Operation rejected by safety "
                    "supervisor operator."
                )
