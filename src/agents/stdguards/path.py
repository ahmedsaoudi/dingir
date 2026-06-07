import functools
import inspect
import json
import os
from typing import Any, Dict, List, Optional

from dingir.agents.guards import Guard, GuardError, log_guard_trigger


class PathGuard(Guard):
    """Security guard that restricts path parameters to allowed directories.

    Can be used in three ways:
    1. As a step-level guard:
       `on_step_callback=PathGuard(allowed_dirs=["./sandbox"])`
    2. As a tool decorator:
       `@PathGuard(allowed_dirs=["./sandbox"])`
    3. As a tool guard via `guard_tool`:
       `guard_tool(write_file, PathGuard(allowed_dirs=["./sandbox"]))`
    """

    def __init__(
        self,
        allowed_dirs: List[str],
        param_name: str = "filepath",
        log: bool = True,
    ):
        super().__init__(log=log)
        self.allowed_dirs = [os.path.abspath(d) for d in allowed_dirs]
        self.param_name = param_name
        self.wrapped = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.wrapped is not None:
            # Case 1: Decorated function is being invoked
            sig = inspect.signature(self.wrapped)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            self._check_args(
                bound.arguments,
                tool_name=getattr(self.wrapped, "__name__", None),
            )
            return self.wrapped(*args, **kwargs)

        # Case 2: Check if used as a step guard (called with Agent object)
        if len(args) == 1 and hasattr(args[0], "memory"):
            agent = args[0]
            if agent.memory.last and agent.memory.last.tool_calls:
                for tc in agent.memory.last.tool_calls:
                    tc_args = tc.get("arguments") or {}
                    if isinstance(tc_args, str):
                        try:
                            tc_args = json.loads(tc_args)
                        except Exception:
                            pass
                    if isinstance(tc_args, dict):
                        self._check_args(
                            tc_args, agent=agent, tool_name=tc.get("name")
                        )
            return None

        # Case 3: Check if used within guard_tool (called with bound arguments dict)
        if len(args) == 1 and isinstance(args[0], dict):
            self._check_args(args[0])
            return None

        # Case 4: Decorator wrapping a function
        func = args[0]
        self.wrapped = func
        functools.update_wrapper(self, func)
        return self

    def _check_args(
        self,
        args: Dict[str, Any],
        agent: Optional[Any] = None,
        tool_name: Optional[str] = None,
    ) -> None:
        filepath = args.get(self.param_name)
        if filepath is not None:
            abs_path = os.path.abspath(filepath)
            if not any(
                abs_path.startswith(allowed) for allowed in self.allowed_dirs
            ):
                err = GuardError(
                    f"Access Denied: Path '{filepath}' is outside permitted directories: {self.allowed_dirs}"
                )
                log_guard_trigger(
                    self, err, agent=agent, tool_name=tool_name, arguments=args
                )
                raise err
