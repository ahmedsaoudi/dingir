# Logging Implementation Todo List

This todo list details the individual steps required to implement the logging proposal to make logs the ultimate source of truth for debugging complex agentic workflows.

---

## 📋 Task Checklist

### 1. Log Structure & Entry Types (`src/agents/log.py`)
- [x] Add `"raw_api_call"` to `ENTRY_TYPES` set.
- [x] Update `Log.to_dict()` and `LogEntry.to_dict()` to ensure safe serialization of `raw_api_call` structures.
- [x] Add support in `_format_log` for outputting `[RAW API CALL]` entries in a clean, human-readable indented format.
- [x] Ensure that existing log parser fallbacks are backward-compatible.

### 2. Context Propagation & Telemetry Collection (`src/agents/core.py`)
- [x] Initialize step-level tracking fields in `Agent.__init__` and `Agent.__call__`:
  - `self._current_step_guards = []`
  - `self._current_step_exceptions = []`
- [x] Implement `Agent.record_guard_encounter(guard, status, tool_name=None, arguments=None, error=None)` to record details of guard executions.
- [x] Implement `Agent.record_exception_encounter(exception, context)` to record details of runtime exceptions.
- [x] Wrap `respond()` loop body and `stream()` generator in context sets/resets for the `_active_agent` context variable.
- [x] Store a reference to the assistant `LogEntry` when created in the loop.
- [x] Update and enrich the assistant `LogEntry.metadata` at the end of each turn iteration with the accumulated step telemetry.

### 3. Guard Interception Hooks (`src/agents/guards.py`)
- [x] Update `Guard.__call__` to record step-level guard status (`started`, `passed`, `failed`) into the active agent's trackers.
- [x] Update `Guard.wrap_tool` wrapper to record tool-level guard status into the active agent's trackers.

### 4. Base Driver Logging Helper (`src/agents/llms/base.py`)
- [x] Implement `BaseLLM._log_raw_api_call(request_payload, response_payload)` to fetch the active agent and record a `"raw_api_call"` entry.
- [x] Implement a robust JSON-safe serializer helper to convert arbitrary response objects (like Pydantic models or client-specific response objects) into primitive python dictionaries/strings without throwing exceptions.

### 5. Driver Telemetry Integration (`src/agents/llms/*`)
- [x] **OpenAI Driver (`openai_driver.py`)**:
  - [x] Instrument `execute()` to log raw params and the Pydantic-dumped chat completion response.
  - [x] Instrument `execute_stream()` to log raw params and the final accumulated message result.
- [x] **Gemini Driver (`gemini_driver.py`)**:
  - [x] Instrument `execute()` to log request parameters and the generate content response.
  - [x] Instrument `execute_stream()` to log request parameters and the final accumulated response.
- [x] **HuggingFace API Driver (`hf_driver.py`)**:
  - [x] Instrument `execute()` and `execute_stream()` to log request dictionaries and responses.
- [x] **HuggingFace Local Driver (`hf_local.py`)**:
  - [x] Instrument `execute()` and `execute_stream()` to log the generated Jinja prompt, input kwargs, outputs, and parsed results.
- [x] **Ollama Driver (`ollama_driver.py`)**:
  - [x] Instrument `execute()` and `execute_stream()` to log the `chat_kwargs` and response dictionaries.
