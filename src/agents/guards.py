from typing import Any, Callable, Dict, Optional
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

    log: bool = True

    def __init__(self, log: bool = True) -> None:
        self.log = log

    def get_details(self) -> Dict[str, Any]:
        """Return a dictionary of the guard's configuration/constraints."""
        details = {}
        for k, v in self.__dict__.items():
            if not k.startswith("_") and not callable(v) and k != "wrapped":
                details[k] = v
        return details

    def __call__(self, agent: Any = None) -> None:
        """Execute the guard check on the current agent state."""
        raise NotImplementedError(
            "Custom guards must implement __call__(self, agent)"
        )


def _find_active_agent() -> Optional[Any]:
    import sys
    frame = sys._getframe(1)
    while frame:
        self_obj = frame.f_locals.get("self")
        if self_obj and self_obj.__class__.__name__ == "Agent":
            return self_obj
        frame = frame.f_back
    return None


def log_guard_trigger(
    guard: Any,
    error: Exception,
    agent: Optional[Any] = None,
    tool_name: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None,
) -> None:
    if not getattr(guard, "log", True):
        return
    if agent is None:
        agent = _find_active_agent()
    if agent is not None and hasattr(agent, "log") and agent.log is not None:
        if isinstance(guard, Guard):
            constraints = guard.get_details()
            guard_type = guard.__class__.__name__
        else:
            constraints = {}
            guard_type = getattr(guard, "__name__", str(guard))

        context = {
            "guard_type": guard_type,
            "constraints": constraints,
            "error": str(error),
        }
        if tool_name:
            context["applied_to_tool"] = tool_name
        if arguments:
            context["tool_arguments"] = arguments
        if hasattr(agent, "__name__"):
            context["applied_to_agent"] = agent.__name__

        agent.log.record(
            entry_type="guard_trigger",
            content=context,
            agent_name=getattr(agent, "__name__", ""),
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
    tool: Callable[..., Any],
    *guards: Callable[[Dict[str, Any]], None],
    log: bool = True,
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
            try:
                guard(bound.arguments)
            except GuardError as e:
                if log:
                    log_guard_trigger(
                        guard,
                        e,
                        tool_name=tool.__name__,
                        arguments=bound.arguments,
                    )
                raise

        return tool(*args, **kwargs)

    return wrapper
