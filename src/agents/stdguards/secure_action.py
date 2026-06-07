import functools
import inspect
import json
from typing import Any, Callable

from dingir.agents.guards import Guard, GuardError, get_approval_handler, log_guard_trigger


class SecureAction(Guard):
    """Security middleware gate for strict type verification and HITL approval.

    Can be used in two ways:
    1. As a step-level guard:
       `on_step_callback=SecureAction(require_approval=True)`
    2. As a tool decorator:
       `@SecureAction(require_approval=True)`
    """

    def __init__(
        self,
        func_or_require_approval: Any = True,
        require_approval: bool = True,
        log: bool = True,
    ):
        super().__init__(log=log)
        if callable(func_or_require_approval):
            self.require_approval = require_approval
            self.wrapped = self._wrap(func_or_require_approval)
        else:
            self.wrapped = None
            self.require_approval = func_or_require_approval

    def __call__(self, *args, **kwargs) -> Any:
        if self.wrapped is not None:
            # Case 1: Decorated function is being invoked
            return self.wrapped(*args, **kwargs)

        # Check if used as a step guard (called with Agent object)
        if len(args) == 1 and hasattr(args[0], "memory"):
            agent = args[0]
            if agent.memory.last and agent.memory.last.tool_calls:
                for tc in agent.memory.last.tool_calls:
                    if self.require_approval:
                        handler = get_approval_handler()
                        prompt = f"Authorization requested for tool execution: '{tc.get('name')}'"
                        payload = tc.get("arguments")
                        approved = handler(prompt, payload=payload)
                        if approved:
                            log_guard_trigger(
                                self,
                                f"Authorization approved for tool execution: '{tc.get('name')}'",
                                agent=agent,
                                tool_name=tc.get("name"),
                                arguments=payload,
                                status="approved",
                            )
                        else:
                            err = GuardError(
                                "EXECUTION BLOCKED: Operation rejected by safety supervisor operator."
                            )
                            log_guard_trigger(
                                self,
                                str(err),
                                agent=agent,
                                tool_name=tc.get("name"),
                                arguments=payload,
                                status="rejected",
                            )
                            raise err
            return None

        # Case 2: Decorator with args, Python passes the function to wrap
        func = args[0]
        return self._wrap(func)

    def _wrap(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)

            # Unpack JSON arguments strings automatically if pushed from tool executors
            purged_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, str) and (
                    v.startswith("{") or v.startswith("[")
                ):
                    try:
                        purged_kwargs[k] = json.loads(v)
                    except:
                        purged_kwargs[k] = v
                else:
                    purged_kwargs[k] = v

            bound_args = sig.bind(*args, **purged_kwargs)
            bound_args.apply_defaults()

            # 1. Type Enforcement Guardrail Pass
            for name, value in bound_args.arguments.items():
                annotation = sig.parameters[name].annotation
                if annotation != inspect.Parameter.empty and not isinstance(
                    value, annotation
                ):
                    err_msg = f"SECURITY FAULT: Param '{name}' must match type {annotation.__name__}."
                    log_guard_trigger(
                        self,
                        err_msg,
                        tool_name=func.__name__,
                        arguments=dict(bound_args.arguments),
                        status="failed",
                    )
                    return err_msg

            # 2. Human-In-The-Loop Approval Pass
            if self.require_approval:
                handler = get_approval_handler()
                prompt = f"Authorization requested for tool operation: '{func.__name__}'"
                payload = dict(bound_args.arguments)
                approved = handler(prompt, payload=payload)
                if approved:
                    log_guard_trigger(
                        self,
                        f"Authorization approved for tool operation: '{func.__name__}'",
                        tool_name=func.__name__,
                        arguments=payload,
                        status="approved",
                    )
                else:
                    err_msg = "EXECUTION BLOCKED: Operation rejected by safety supervisor operator."
                    log_guard_trigger(
                        self,
                        err_msg,
                        tool_name=func.__name__,
                        arguments=payload,
                        status="rejected",
                    )
                    return err_msg

            return func(*args, **purged_kwargs)

        return wrapper
