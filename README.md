# Dingir: Django-Grade Multi-Agent Orchestration Framework

**Dingir** is a stateless multi-agent orchestration engine and contextual data store designed with Django-grade simplicity and production-ready resilience. Build sophisticated agentic workflows with seamless LLM integration, built-in guardrails, and comprehensive observability.

## ✨ Key Features

- **Stateless Agent Architecture**: Agents maintain isolation with deterministic execution—perfect for distributed and serverless environments
- **Multi-LLM Support**: Unified interface for OpenAI, Google Gemini, Ollama, HuggingFace, and more
- **Guard Rails**: Built-in safety mechanisms and custom validation guards for controlled agent behavior
- **Comprehensive Logging**: Detailed execution logs with memory management and structured event tracking
- **Tool Integration**: Seamlessly bind Python functions and other agents as callable tools
- **RAG Support**: Optional Retrieval-Augmented Generation with ChromaDB integration
- **Config Composition**: Unified `ModelConfig` for symmetric parameter handling across all backends

## 🚀 Quick Start

### Installation

```bash
pip install dingir
```

For RAG capabilities:
```bash
pip install dingir[rag]
```

For local model support:
```bash
pip install dingir[local]
```

### Your First Agent

Here's a complete example that creates a simple agent using OpenAI:

```python
from dingir import ModelConfig, Memory, Message
from dingir.agents.core import Agent
from dingir.agents.llms.openai import OpenAIDriver

# 1. Set up your model configuration
config = ModelConfig(
    temperature=0.7,
    max_tokens=1024,
    top_p=0.9
)

# 2. Create a driver for your LLM
driver = OpenAIDriver(
    api_key="your-openai-key",
    model="gpt-4",
    config=config
)

# 3. Define your system prompt
system_prompt = """You are a helpful assistant that answers questions clearly and concisely.
Always provide accurate information and ask for clarification when needed."""

# 4. Create an agent
agent = Agent(
    model=driver,
    system=system_prompt,
    name="my_assistant",
    description="A helpful question-answering assistant"
)

# 5. Execute the agent
response = agent("What is the capital of France?")
print(response)  # Output: "The capital of France is Paris..."

# 6. Access execution logs and memory
print(agent.log)      # Detailed execution log
print(agent.memory)   # Message history
```

## 📚 Core Concepts

### Agent

The `Agent` class is the core orchestration boundary:

```python
Agent(
    model=driver,              # LLM driver instance
    system=str,                # System prompt
    name=str,                  # Optional agent name
    description=str,           # Optional description
    tools=list,                # Optional list of callable tools
    guards=list                # Optional list of guard functions
)
```

**Key Methods:**
- `respond(message)`: Process a message through the agent's decision loop
- `__call__(instruction)`: Execute agent with fresh memory (for subagent mode)

### Memory & Logging

Track all agent activity:

```python
# Access conversation memory
agent.memory.messages          # List of Message objects
agent.memory.last              # Last message in conversation

# Detailed execution logs
agent.log.entries              # All logged events
agent.log.to_dict()            # Serialize to dictionary
agent.log.merge(other_log)     # Merge logs from subagents
```

### ModelConfig

Unified configuration across all LLM backends:

```python
config = ModelConfig(
    temperature=0.5,           # Creativity (0-1)
    max_tokens=2048,           # Response length limit
    top_p=0.95,                # Nucleus sampling
    top_k=50,                  # Top-k sampling
    presence_penalty=0.0,      # Repeat penalty
    frequency_penalty=0.0,     # Frequency penalty
    seed=42,                   # Reproducibility
    timeout=30.0,              # Request timeout
    max_retries=3              # Retry attempts
)

# Compose multiple configs (rightmost wins)
merged = ModelConfig.merge([base_config, override_config])
```

## 🛠️ Tools & Function Binding

Agents can use Python functions as tools:

```python
def search_web(query: str) -> str:
    """Search the web for information."""
    # Implementation here
    return f"Results for '{query}'"

def get_weather(location: str) -> str:
    """Get current weather for a location."""
    # Implementation here
    return f"Weather in {location}: 72°F"

agent = Agent(
    model=driver,
    system="You are a helpful assistant with access to tools.",
    tools=[search_web, get_weather]
)

# Agent will now intelligently choose to call these tools when needed
response = agent("What's the weather in Paris and any recent news there?")
```

## 🔒 Guards & Safety

Add validation guards to control agent behavior:

```python
from dingir.agents.guards import Guard

def safety_check(agent: Agent):
    """Validate agent state before each step."""
    if len(agent.memory.messages) > 100:
        raise GuardError("Conversation exceeded maximum length")

def rate_limiter(agent: Agent):
    """Enforce rate limits."""
    # Custom rate limiting logic
    pass

agent = Agent(
    model=driver,
    system=system_prompt,
    guards=[safety_check, rate_limiter]
)
```

## 🔗 Multi-Agent Workflows

Create sophisticated workflows with subagent delegation:

```python
# Create specialized agents
researcher = Agent(
    model=driver,
    system="You are a research expert. Analyze and synthesize information.",
    name="researcher"
)

writer = Agent(
    model=driver,
    system="You are a professional writer. Create clear, engaging content.",
    name="writer",
    tools=[researcher]  # Researcher becomes a callable tool
)

# Orchestrate the workflow
result = writer("Write a detailed article about machine learning trends")
# Writer will delegate research tasks to the researcher agent
```

## 📊 Observability

Access comprehensive execution data:

```python
# Get structured logs
log_data = agent.log.to_dict()

# Example log structure:
{
    "agent_name": "my_assistant",
    "model_id": "gpt-4",
    "system_prompt": "...",
    "entries": [
        {
            "type": "message",
            "role": "user",
            "content": "What is AI?"
        },
        {
            "type": "tool_call",
            "name": "search_web",
            "arguments": {"query": "artificial intelligence"}
        },
        {
            "type": "tool_result",
            "name": "search_web",
            "output": "..."
        }
    ]
}

# Monitor execution at runtime
def on_step(agent):
    print(f"Step completed. Messages: {len(agent.memory.messages)}")

agent.respond("Your query", on_step_callback=on_step)
```

## 🔌 Supported LLM Backends

Dingir provides drivers for:

- **OpenAI**: GPT-4, GPT-3.5, and other OpenAI models
- **Google Gemini**: Gemini Pro and advanced models
- **Ollama**: Local and remote Ollama instances
- **HuggingFace**: Inference API and local transformers
- **Custom Drivers**: Easily implement your own LLM driver

All drivers use the same unified `ModelConfig` for parameter handling.

## 🧠 RAG Integration

Enable Retrieval-Augmented Generation for knowledge-grounded responses:

```python
from dingir.rag.stores import ChromaStore

# Set up RAG
rag_store = ChromaStore(collection_name="my_docs")
rag_store.add_documents([
    {"content": "Document 1", "metadata": {"source": "doc1"}},
    {"content": "Document 2", "metadata": {"source": "doc2"}}
])

# Agent can now retrieve relevant context
agent = Agent(
    model=driver,
    system="Use provided context to answer questions accurately.",
    tools=[rag_store.retrieve]  # Add retrieval as a tool
)
```

## 📝 Examples

The `tests/` directory contains comprehensive examples:

- **test_config_composition.py**: Working with `ModelConfig` composition
- **test_stdtools.py**: Using standard tools and guards
- **drivers/**: Examples for each supported LLM backend

Run tests:
```bash
python tests/run_tests.py
```

## 🏗️ Architecture

```
dingir/
├── agents/
│   ├── core.py          # Agent orchestration engine
│   ├── memory.py        # Conversation memory management
│   ├── log.py           # Execution logging
│   ├── guards.py        # Safety & validation framework
│   ├── llms/            # LLM driver implementations
│   ├── stdtools/        # Standard library tools
│   └── stdguards/       # Standard library guards
├── config.py            # Unified ModelConfig
└── rag/                 # RAG implementation
    └── stores/          # Vector store backends
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

Dingir is open source and available under the MIT License.

## 🤔 FAQ

**Q: Is Dingir stateless?**  
A: Yes! Each agent execution is isolated and deterministic. Perfect for serverless and distributed deployments.

**Q: Can I use multiple LLM backends?**  
A: Absolutely! Create different agents with different drivers and orchestrate them together.

**Q: How do I handle errors and retries?**  
A: Use `ModelConfig` to set `max_retries` and `timeout`, and implement custom guards for validation.

**Q: What's the difference between tools and guards?**  
A: **Tools** are functions agents can call to take action. **Guards** are callbacks that validate agent state before and after each step.

---

**Ready to build intelligent agents?** Check out the [examples](tests/) directory to get started!
