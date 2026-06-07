import functools
import inspect
import json
from typing import Any, Callable, Dict

from dingir.agents.guards import (
    Guard,
    GuardError,
    get_approval_handler,
    log_guard_trigger,
)


class SecureAction(Guard):
    """Security middleware for type verification and HITL approval.

    Works in all 3 modes:

    1. Step callback::

        on_step_callback=SecureAction(require_approval=True)

    2. Decorator::

        @SecureAction(require_approval=True)
        def dangerous_tool(cmd: str): ...

    3. guard_tool::

        guard_tool(dangerous_tool, SecureAction(require_approval=True))

    In decorator mode, also performs type enforcement against
    parameter annotations.
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

    def wrap_tool(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator mode: type enforcement + approval check."""

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            sig = inspect.signature(func)

            # Unpack JSON argument strings automatically
            purged_kwargs: Dict[str, Any] = {}
            for k, v in kwargs.items():
                if isinstance(v, str) and (
                    v.startswith("{") or v.startswith("[")
                ):
                    try:
                        purged_kwargs[k] = json.loads(v)
                    except Exception:
                        purged_kwargs[k] = v
                else:
                    purged_kwargs[k] = v

            bound = sig.bind(*args, **purged_kwargs)
            bound.apply_defaults()

            # 1. Type enforcement guardrail
            for name, value in bound.arguments.items():
                annotation = sig.parameters[name].annotation
                if (
                    annotation != inspect.Parameter.empty
                    and not isinstance(value, annotation)
                ):
                    raise GuardError(
                        f"SECURITY FAULT: Param '{name}' must match "
                        f"type {annotation.__name__}."
                    )

            # 2. Approval check (delegates to check_tool_args)
            self.check_tool_args(func.__name__, dict(bound.arguments))

            return func(*args, **purged_kwargs)

        wrapper._dingir_guards = getattr(func, "_dingir_guards", []) + [self]
        return wrapper
