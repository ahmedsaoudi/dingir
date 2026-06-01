import functools
import inspect
import json
from typing import Callable


def secure_action(require_approval: bool = True):
    """Django-grade security middleware gate for strict type verification and HITL approval."""

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)

            # Unpack JSON arguments strings automatically if pushed from tool executors
            purged_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, str) and (v.startswith("{") or v.startswith("[")):
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
                    return f"SECURITY FAULT: Param '{name}' must match type {annotation.__name__}."

            # 2. Human-In-The-Loop Approval Pass
            if require_approval:
                print(
                    f"\n⚠️  [GUARD BLOCK]: Authorization requested for operation '{func.__name__}'."
                )
                print(f"   Payload: {dict(bound_args.arguments)}")
                confirm = (
                    input("👉 Authorize action implementation turn? (y/N): ")
                    .strip()
                    .lower()
                )
                if confirm not in ("y", "yes"):
                    return "EXECUTION BLOCKED: Operation rejected by safety supervisor operator."

            return func(*args, **purged_kwargs)

        return wrapper

    return decorator
