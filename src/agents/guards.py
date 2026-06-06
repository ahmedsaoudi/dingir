from typing import Any, Callable, Dict
import functools
import sys


class GuardError(Exception):
    """Base exception class for all guard-related failures."""

    pass


class Guard:
    """Base class for all agent guards.

    To write a custom guard, inherit from this class and implement the
    `__call__` method. The method should accept the `agent` context and raise
    a `GuardError` (or a subclass of it) if the guard check fails.
    """

    def __call__(self, agent: Any = None) -> None:
        """Execute the guard check on the current agent state."""
        raise NotImplementedError(
            "Custom guards must implement __call__(self, agent)"
        )


def default_approval_handler(prompt: str, payload: Any = None) -> bool:
    """Standard CLI-based approval handler. Fails securely if stdin is not a TTY."""
    if not (sys.stdin and sys.stdin.isatty()):
        raise GuardError(
            f"Approval requested but environment is non-interactive: {prompt}"
        )
    print(f"\n⚠️  [GUARD APPROVAL REQUESTED]: {prompt}")
    if payload:
        print(f"   Payload: {payload}")
    try:
        confirm = (
            input("👉 Authorize action implementation turn? (y/N): ")
            .strip()
            .lower()
        )
        return confirm in ("y", "yes")
    except EOFError:
        raise GuardError(f"Approval requested but stdin reached EOF: {prompt}")


_approval_handler: Callable[[str, Any], bool] = default_approval_handler


def set_approval_handler(handler: Callable[[str, Any], bool]) -> None:
    """Sets the global approval handler used by interactive guards."""
    global _approval_handler
    _approval_handler = handler


def get_approval_handler() -> Callable[[str, Any], bool]:
    """Gets the currently active approval handler."""
    return _approval_handler


def guard_tool(
    tool: Callable[..., Any], *guards: Callable[[Dict[str, Any]], None]
) -> Callable[..., Any]:
    """Wraps a tool callable with a list of tool-level guards.

    Each guard is called with a dictionary of the bound tool arguments
    before the underlying tool is executed.
    """
    import inspect

    @functools.wraps(tool)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        sig = inspect.signature(tool)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Execute guards in order
        for guard in guards:
            guard(bound.arguments)

        return tool(*args, **kwargs)

    return wrapper
