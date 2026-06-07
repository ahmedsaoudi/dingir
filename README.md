# Dingir Framework

> The Django-grade stateless multi-agent orchestration and contextual data store engine.

Dingir is a highly structured, stateless orchestration library designed to build resilient, interchangeable multi-agent topologies. It treats agents, tools, guards, and data sources as interchangeable callables, allowing developers to compose dynamic workflows with built-in human-in-the-loop (HITL) safety guardrails and complete trace logging.

---

## Key Concepts

- **Interchangeability**: Subagents, callables, and tools share a unified signature. An agent can call another agent exactly like a tool.
- **Stateless Orchestration**: Orchestration loops are stateless, relying on explicit conversation memory and logs for tracing.
- **Guardrails**: Guards are execution boundaries (e.g., path restrictions, iteration limits, human approval) that can be applied seamlessly as step callbacks, decorators, or tool wraps.
- **Structured Logs**: Logging records all activities (system prompts, configurations, tool executions, guard status) recursively for subagents and primary controllers.

---

## Code Example: Orchestration & Subagents

Here is how to set up a primary orchestrator that delegates tasks to specialized subagents:

```python
from dingir import ModelConfig
from dingir.agents import Agent
from dingir.agents.stdtools import read_file, write_file, replace_lines
from dingir.agents.llms import OpenAICompatible
from dingir.agents.guards import guard_tool
from dingir.agents.stdguards import IterationGuard, SecureAction

def main():
    model = OpenAICompatible(
        config=ModelConfig(temperature=0.7, max_tokens=1024),
        id="gemma-4-12b",
        base_url="http://127.0.0.1:1234/v1",
        api_key="none",
    )

    # Secure file access tool wrapped with HITL approval
    secure_read = guard_tool(read_file, SecureAction(require_approval=True))

    # Subagent 1: Specialized in reading codebase (guarded at the agent level)
    code_reader = Agent(
        model=model,
        system="You are a specialized code reader subagent. Inspect and return file contents.",
        tools=[secure_read],
        guards=[PathGuard(allowed_dirs=["./sandbox"])]  # Protects the entire subagent and all tools it uses!
    )
    code_reader.__name__ = "code_reader"
    code_reader.description = "Inspects file contents. Accepts a filename."
    code_reader.__doc__ = code_reader.description

    # Subagent 2: Specialized in modifying codebase
    code_writer = Agent(
        model=model,
        system="You are a specialized code writer subagent. Write and edit files.",
        tools=[write_file, replace_lines],
    )
    code_writer.__name__ = "code_writer"
    code_writer.description = "Writes/modifies file contents."
    code_writer.__doc__ = code_writer.description

    # Orchestrator coordinating subagents
    orchestrator = Agent(
        model=model,
        system=(
            "Coordinate fixing bugs in code.py. "
            "1. Delegate reading code to `code_reader`.\n"
            "2. Delegate writing/fixing code to `code_writer`."
        ),
        tools=[code_reader, code_writer],
    )
    orchestrator.__name__ = "orchestrator"

    # Execute orchestrator loop
    orchestrator.respond(
        "fix any code bugs in 'code.py'",
        on_step_callback=[IterationGuard(max_iterations=4)]
    )

if __name__ == "__main__":
    main()
```

---

## Community Proposal: Real-Time Chat & Agentic Harnesses

To support high-concurrency real-time environments (like WebSockets, server-sent events, and live agent workspaces), we propose upgrading the execution loop to support async-await and real-time streaming tokens.

### Phase A: Async Model Drivers & Streaming Support

The model drivers will implement async client wrappers yielding token deltas:

```python
class BaseLLM(ABC):
    @abstractmethod
    async def request_stream(
        self,
        system: Optional[str],
        messages: List[Dict[str, Any]],
        tools: List[Any],
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yields chunks containing partial content tokens or tool call deltas."""
        pass
```

### Phase B: Asynchronous Agent Loop (`asyncio`)

Convert the core execution loop to support async tool resolution and token callbacks:

```python
class Agent:
    async def respond_async(
        self,
        message: Optional[str] = None,
        on_token_callback: Optional[Callable[[str], None]] = None,
        on_call_callback: Optional[Callable[[dict], None]] = None,
    ):
        # Async execution of tools and streaming token generation
        ...
```

### Phase C: Unified Execution Events

Following Dingir's core design principle of **interchangeability**, all callables (functions, subagents, MCP tools, and RAG sources) are treated as unified execution units. They report events through a single set of paired hooks:

| Hook Event | Trigger Point | Payload Description |
| :--- | :--- | :--- |
| `on_token` | Each time a new token is generated | `token: str` |
| `on_thought` | During reasoning channel output | `thought: str` |
| `on_call_start` | Before executing any callable (agent, function, MCP, etc.) | `name: str`, `arguments: dict`, `type: str` ("agent", "function", "mcp"), `guards: list` |
| `on_call_end` | After the callable returns | `name: str`, `output: str`, `type: str`, `logs: Optional[dict]` (if type is "agent") |

Harnesses or frontends can inspect `type` and `logs` in `on_call_end` to render nested UI layouts (like sub-agent reasoning loops) dynamically.
