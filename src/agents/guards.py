from typing import Any, Callable, Dict, Optional
import contextvars
import functools
import inspect
import json
import sys


# Context variable set by Agent._execute_tool_sync so that guards
# firing inside guard_tool wrappers can discover the active agent
# without stack introspection.
_active_agent: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "_active_agent", default=None
)


class GuardError(Exception):
    """Base exception class for all guard-related failures."""

    pass


class Guard:
    """Base class for all agent guards.

    Subclasses implement domain logic via two hooks:

    - ``check_step(agent)``: Called during the agent loop. Override for
      step-level guards like iteration limits. Receives the Agent instance.

    - ``check_tool_args(tool_name, arguments)``: Called to validate tool
      arguments. Override for argument-validation guards like path
      restrictions.

    A guard works in all 3 usage modes automatically:

    1. Step callback:   ``agent.respond(msg, on_step_callback=MyGuard())``
    2. Decorator:       ``@MyGuard()``
    3. guard_tool:      ``guard_tool(my_func, MyGuard())``

    For argument-validation guards, implement only ``check_tool_args``.
    The default ``check_step`` automatically extracts pending tool calls
    from agent memory and delegates to ``check_tool_args``, so all 3
    modes work with a single method override.

    Logging is automatic — just raise ``GuardError`` and the framework
    handles the rest.
    """

    log: bool = True

    def __init__(self, log: bool = True) -> None:
        self.log = log

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        original_init = cls.__dict__.get("__init__")
        if original_init is not None:

            @functools.wraps(original_init)
            def checked_init(self: Any, *args: Any, **kwargs: Any) -> None:
                original_init(self, *args, **kwargs)
                if not hasattr(self, "log"):
                    raise TypeError(
                        f"{cls.__name__}.__init__ must call super().__init__(). "
                        f"Missing 'log' attribute."
                    )

            cls.__init__ = checked_init  # type: ignore[attr-defined]

    def get_details(self) -> Dict[str, Any]:
        """Return a dictionary of the guard's configuration/constraints."""
        details = {}
        for k, v in self.__dict__.items():
            if not k.startswith("_") and not callable(v):
                details[k] = v
        return details

    # --------------------------------------------------------------------- #
    # Override hooks — subclasses implement these
    # --------------------------------------------------------------------- #

    def check_step(self, agent: Any) -> None:
        """Called as ``on_step_callback``.  Override for step-level guards.

        Default implementation: extracts pending tool calls from the
        agent's memory and calls ``check_tool_args`` for each.  This
        means argument-validation guards only need to implement
        ``check_tool_args`` to work in all 3 modes.

        Raise ``GuardError`` to halt execution.
        """
        if (
            agent is not None
            and hasattr(agent, "memory")
            and agent.memory.last
            and agent.memory.last.tool_calls
        ):
            for tc in agent.memory.last.tool_calls:
                tc_args = tc.get("arguments") or {}
                if isinstance(tc_args, str):
                    try:
                        tc_args = json.loads(tc_args)
                    except Exception:
                        pass
                if isinstance(tc_args, dict):
                    tool_name = tc.get("name", "")
                    try:
                        self.check_tool_args(tool_name, tc_args)
                    except GuardError as e:
                        if self.log and not getattr(
                            e, "_guard_logged", False
                        ):
                            log_guard_trigger(
                                self,
                                str(e),
                                agent=agent,
                                tool_name=tool_name,
                                arguments=tc_args,
                                status="failed",
                            )
                            e._guard_logged = True  # type: ignore[attr-defined]
                        raise

    def check_tool_args(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> None:
        """Called to validate tool arguments before execution.

        Override for argument-validation guards.  Receives the tool name
        and a dict of bound arguments.  Raise ``GuardError`` to block
        the call.
        """
        pass

    def wrap_tool(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Called when used as a ``@decorator``.  Returns a wrapped callable.

        Default implementation: binds arguments and calls
        ``check_tool_args``.  Override for guards that need access to
        the function signature (e.g. type enforcement).
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            try:
                self.check_tool_args(func.__name__, bound.arguments)
            except GuardError as e:
                if self.log and not getattr(e, "_guard_logged", False):
                    log_guard_trigger(
                        self,
                        str(e),
                        tool_name=func.__name__,
                        arguments=bound.arguments,
                        status="failed",
                    )
                    e._guard_logged = True  # type: ignore[attr-defined]
                raise
            return func(*args, **kwargs)

        wrapper._dingir_guards = getattr(func, "_dingir_guards", []) + [self]
        return wrapper

    # --------------------------------------------------------------------- #
    # Dispatcher — routes to the correct hook automatically
    # --------------------------------------------------------------------- #

    def __call__(self, func_or_agent: Any = None) -> Any:
        """Smart dispatcher that routes to the appropriate handler.

        - If called with an Agent (has ``.memory``), routes to ``check_step``.
        - If called with a callable, routes to ``wrap_tool`` (decorator mode).
        - If called with ``None``, routes to ``check_step(None)``.
        """
        # Step callback mode: called with an Agent instance
        if func_or_agent is not None and hasattr(func_or_agent, "memory"):
            try:
                self.check_step(func_or_agent)
            except GuardError as e:
                if self.log and not getattr(e, "_guard_logged", False):
                    log_guard_trigger(
                        self,
                        str(e),
                        agent=func_or_agent,
                        status="failed",
                    )
                    e._guard_logged = True  # type: ignore[attr-defined]
                raise
            return None

        # Decorator mode: called with a function to wrap
        if callable(func_or_agent):
            return self.wrap_tool(func_or_agent)

        # Step callback with no agent context
        if func_or_agent is None:
            try:
                self.check_step(None)
            except GuardError as e:
                if self.log and not getattr(e, "_guard_logged", False):
                    log_guard_trigger(self, str(e), status="failed")
                    e._guard_logged = True  # type: ignore[attr-defined]
                raise
            return None

        return None


def log_guard_trigger(
    guard: Any,
    message: str,
    agent: Optional[Any] = None,
    tool_name: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None,
    status: str = "failed",
) -> None:
    """Record a guard event to the agent's log.

    Called automatically by the ``Guard`` base class.  Guard authors
    should not normally need to call this directly — just raise
    ``GuardError`` and the framework handles logging.  Direct calls
    are still supported for custom statuses (e.g. ``status='approved'``).
    """
    if not getattr(guard, "log", True):
        return
    if agent is None:
        agent = _active_agent.get()
    if agent is not None and hasattr(agent, "log") and agent.log is not None:
        if isinstance(guard, Guard):
            constraints = guard.get_details()
            guard_type = guard.__class__.__name__
        else:
            constraints = {}
            guard_type = getattr(guard, "__name__", str(guard))

        context: Dict[str, Any] = {
            "guard_type": guard_type,
            "constraints": constraints,
            "error": message,
            "status": status,
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
    print(f"\n[GUARD APPROVAL REQUESTED]: {prompt}")
    if payload:
        print(f"Payload: {payload}")
    try:
        confirm = (
            input("Authorize action implementation turn? (y/N): ")
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
    *guards: "Guard",
    log: bool = True,
) -> Callable[..., Any]:
    """Wraps a tool callable with a list of guards.

    Each Guard's ``check_tool_args`` method is called with the tool name
    and bound arguments before the underlying tool is executed.
    Plain callables are still supported for backward compatibility.
    """

    @functools.wraps(tool)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        sig = inspect.signature(tool)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        for guard in guards:
            if isinstance(guard, Guard):
                try:
                    guard.check_tool_args(tool.__name__, bound.arguments)
                except GuardError as e:
                    if guard.log and not getattr(
                        e, "_guard_logged", False
                    ):
                        log_guard_trigger(
                            guard,
                            str(e),
                            tool_name=tool.__name__,
                            arguments=bound.arguments,
                            status="failed",
                        )
                        e._guard_logged = True  # type: ignore[attr-defined]
                    raise
            else:
                # Legacy: plain callable guard
                try:
                    guard(bound.arguments)
                except GuardError as e:
                    if log and not getattr(e, "_guard_logged", False):
                        log_guard_trigger(
                            guard,
                            str(e),
                            tool_name=tool.__name__,
                            arguments=bound.arguments,
                            status="failed",
                        )
                        e._guard_logged = True  # type: ignore[attr-defined]
                    raise

        return tool(*args, **kwargs)

    wrapper._dingir_guards = getattr(tool, "_dingir_guards", []) + list(guards)
    return wrapper
