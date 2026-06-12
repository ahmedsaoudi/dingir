import inspect
import json
from typing import Any, Dict, List, Optional


def create_agent(
    instruction: str,
    system_prompt: str,
    tools: Any,
    name: Optional[str] = None,
) -> str:
    """Creates a subagent on the fly and executes it on the specified instruction/mission.

    Args:
        instruction: The task, mission, or prompt the subagent should execute.
        system_prompt: The system prompt or instructions for the subagent.
        tools: A list of tool names (or a comma-separated string) to give the subagent access to. Use 'all' to grant all available tools.Can be empty.
        name: An optional name for the subagent (defaults to a generated name).
    """
    from dingir.agents.guards import _active_agent
    from dingir.agents.core import Agent

    parent = _active_agent.get()
    if not parent:
        return "ERROR: No active parent agent found. 'create_agent' must be executed within an active agent context."

    # Parse tools argument
    if not tools:
        tools_list = []
    elif isinstance(tools, str):
        if tools.strip() in ("", "[]"):
            tools_list = []
        else:
            try:
                parsed = json.loads(tools)
                if isinstance(parsed, list):
                    tools_list = parsed
                else:
                    tools_list = [
                        t.strip() for t in tools.split(",") if t.strip()
                    ]
            except Exception:
                tools_list = [t.strip() for t in tools.split(",") if t.strip()]
    elif isinstance(tools, list):
        tools_list = tools
    else:
        return "ERROR: 'tools' must be a list of tool names, a comma-separated string, or empty/None."

    # Get available tools map
    tools_map = get_all_tools_map()
    subagent_tools = []

    for t_name in tools_list:
        if t_name == "all":
            for f in tools_map.values():
                if f not in subagent_tools:
                    subagent_tools.append(f)
            continue
        if t_name in tools_map:
            subagent_tools.append(tools_map[t_name])
        else:
            return f"ERROR: Tool '{t_name}' not found. Available tools: {', '.join(tools_map.keys())}"

    subagent_name = name if name and name.strip() else None

    try:
        # Create and invoke subagent
        subagent = Agent(
            model=parent.model,
            system=system_prompt,
            name=subagent_name,
            tools=subagent_tools,
            guards=parent.guards,
        )
        result = subagent(instruction)
        # Merge subagent's execution logs into the parent log
        parent.log.merge(subagent.log)
        return result
    except Exception as e:
        return f"ERROR during subagent execution: {str(e)}"


def list_available_tools() -> str:
    """Lists all the available tools in the system, showing their names, signatures, and descriptions."""
    tools_map = get_all_tools_map()
    if not tools_map:
        return "No tools available."

    lines = []
    for name, func in sorted(tools_map.items()):
        try:
            sig = inspect.signature(func)
        except Exception:
            sig = "(...)"
        doc = inspect.getdoc(func) or "No description available."
        doc_cleaned = "\n  ".join(
            [line.strip() for line in doc.splitlines() if line.strip()]
        )
        lines.append(f"- {name}{sig}:\n  {doc_cleaned}")

    return "\n\n".join(lines)


def register_custom_tool(code: str, name: str) -> str:
    """Dynamically compiles and registers a new Python function as a tool for the current agent.

    The function must be defined with standard type hints and a docstring explaining its behavior and parameters.

    Args:
        code: The complete Python code defining the function.
        name: The name of the function to register.
    """
    from dingir.agents.guards import _active_agent

    active = _active_agent.get()
    if not active:
        return "ERROR: No active agent context found to register the tool."

    namespace = {}
    try:
        # Compile and execute code in a clean namespace
        exec(code, namespace)
    except Exception as e:
        return f"ERROR: Failed to compile/execute code: {str(e)}"

    func = namespace.get(name)
    if not func:
        return f"ERROR: Function '{name}' not found in the provided code namespace."

    if not callable(func):
        return f"ERROR: '{name}' is not callable."

    # Check if a tool with the same name already exists
    existing = next(
        (t for t in active.tools if getattr(t, "__name__", None) == name), None
    )
    if existing:
        active.tools.remove(existing)

    active.tools.append(func)
    return f"Successfully registered custom tool '{name}'."


def get_subagent_log(subagent_name: str) -> str:
    """Retrieves the execution log/history of a previously created subagent by its name.

    This is useful for debugging failures or inspecting intermediate steps taken by the subagent.

    Args:
        subagent_name: The name of the subagent whose log to retrieve.
    """
    from dingir.agents.guards import _active_agent

    active = _active_agent.get()
    if not active:
        return "ERROR: No active agent context found."

    for entry in reversed(active.log._entries):
        if (
            entry.entry_type == "subagent_log"
            and entry.agent_name == subagent_name
        ):
            child_log_dict = entry.content
            entries = child_log_dict.get("entries", [])
            output_lines = []
            output_lines.append(f"=== Subagent Log for '{subagent_name}' ===")
            output_lines.append(
                f"System Prompt: {child_log_dict.get('system_prompt')}"
            )
            output_lines.append(f"Model ID: {child_log_dict.get('model_id')}")
            output_lines.append("Execution Steps:")
            for item in entries:
                etype = item.get("entry_type")
                content = item.get("content")
                if etype == "message":
                    role = content.get("role")
                    text = content.get("content")
                    tool_calls = content.get("tool_calls")
                    if tool_calls:
                        output_lines.append(
                            f"  [{role}]: (called tools: {', '.join(tc.get('name') for tc in tool_calls)})"
                        )
                    else:
                        output_lines.append(f"  [{role}]: {text}")
                elif etype == "tool_call":
                    output_lines.append(
                        f"  [tool call]: {content.get('name')} with {content.get('arguments')}"
                    )
                elif etype == "tool_result":
                    output_lines.append(
                        f"  [tool result]: {content.get('name')} -> {content.get('output')}"
                    )
                elif etype == "exception":
                    output_lines.append(
                        f"  [exception]: {content.get('exception_type')}: {content.get('message')}"
                    )
            return "\n".join(output_lines)

    return f"ERROR: Subagent log for '{subagent_name}' not found."


def list_subagents() -> str:
    """Lists all subagents that have been created and executed in the current agent session, including their names and execution status."""
    from dingir.agents.guards import _active_agent

    active = _active_agent.get()
    if not active:
        return "ERROR: No active agent context found."

    subagents = []
    for entry in active.log._entries:
        if entry.entry_type == "subagent_log":
            name = entry.agent_name
            system_prompt = entry.content.get("system_prompt", "")
            system_summary = (
                system_prompt[:60] + "..."
                if len(system_prompt) > 60
                else system_prompt
            )
            subagents.append(f"- {name} (System: {system_summary})")

    if not subagents:
        return "No subagents have been executed in this session."
    return "\n".join(subagents)


def unregister_tool(name: str) -> str:
    """Removes a tool from the current agent's active tools list.

    Args:
        name: The name of the tool to remove.
    """
    from dingir.agents.guards import _active_agent

    active = _active_agent.get()
    if not active:
        return "ERROR: No active agent context found."

    existing = next(
        (t for t in active.tools if getattr(t, "__name__", None) == name), None
    )
    if existing:
        active.tools.remove(existing)
        return f"Successfully unregistered tool '{name}'."
    return f"ERROR: Tool '{name}' not found in the active tools list."


def get_all_tools_map() -> Dict[str, Any]:
    from dingir.agents import stdtools

    tools_map = {}
    for name in getattr(stdtools, "__all__", []):
        try:
            tool_func = getattr(stdtools, name)
            if callable(tool_func):
                tools_map[name] = tool_func
        except AttributeError:
            pass

    from dingir.agents.guards import _active_agent

    active = _active_agent.get()
    if active:
        for tool_func in active.tools:
            if hasattr(tool_func, "__name__"):
                tools_map[tool_func.__name__] = tool_func
            elif hasattr(tool_func, "__class__") and hasattr(
                tool_func.__class__, "__name__"
            ):
                name = getattr(
                    tool_func, "__name__", tool_func.__class__.__name__
                )
                tools_map[name] = tool_func

    return tools_map
