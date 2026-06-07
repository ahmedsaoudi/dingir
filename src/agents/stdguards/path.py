import os
from typing import Any, Dict, List

from dingir.agents.guards import Guard, GuardError


class PathGuard(Guard):
    """Security guard that restricts path parameters to allowed directories.

    Works in all 3 modes by implementing just ``check_tool_args``:

    1. Step callback::

        on_step_callback=PathGuard(allowed_dirs=["./sandbox"])

    2. Decorator::

        @PathGuard(allowed_dirs=["./sandbox"])
        def write_file(filepath: str, content: str): ...

    3. guard_tool::

        guard_tool(write_file, PathGuard(allowed_dirs=["./sandbox"]))
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

    def check_tool_args(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> None:
        """Reject paths that fall outside the allowed directories."""
        filepath = arguments.get(self.param_name)
        if filepath is not None:
            abs_path = os.path.abspath(filepath)
            if not any(
                abs_path.startswith(allowed) for allowed in self.allowed_dirs
            ):
                raise GuardError(
                    f"Access Denied: Path '{filepath}' is outside "
                    f"permitted directories: {self.allowed_dirs}"
                )
