import re
import json
import inspect
from abc import ABC, abstractmethod
from typing import Generator
from typing import Any, Dict, List, Optional

from dingir.config import ModelConfig


class BaseLLM(ABC):
    """
    The BaseLLM acts as the core orchestration engine for all language model providers.
    It implements the Template Method design pattern to centralize boilerplate logic,
    ensuring all drivers (OpenAI, Gemini, HuggingFace, etc.) behave identically.
    
    Architecture Highlights:
    1. Unified Configuration: Maps a single `ModelConfig` instance to the generic 
       API kwargs (like `temperature`, `max_tokens`) so drivers don't have to guess.
    2. Tool Serialization: Provides a standard engine to parse raw Python functions
       and their docstrings into industry-standard JSON Schemas.
    3. Tool Fallbacks (`use_native_tools=False`): Automatically handles older or 
       quirky API servers that crash when seeing 'tool' roles in message history. 
       It rewrites past tool calls as standard text and injects strict JSON schemas 
       into the system prompt to guide local models.
    4. Reasoning Extraction: Automatically extracts `<think>` or `<|think|>` tags 
       from generic models that don't natively expose a separate reasoning field.
    
    Child classes only need to implement the `execute()` method to handle the 
    actual API network call and return a standard dictionary of the result.
    """
    def __init__(self, id: str, config: Any):
        self.id = id
        if isinstance(config, list):
            self.config = ModelConfig.merge(config)
        elif isinstance(config, ModelConfig):
            self.config = config
        else:
            raise TypeError(
                "Configuration must be a single ModelConfig or a list of ModelConfigs, not a dictionary or other type."
            )

    def request(
        self,
        system: Optional[str],
        messages: List[Dict[str, Any]],
        tools: List[Any],
    ) -> Dict[str, Any]:
        """
        Standardized template method for LLM requests.
        This handles the entire lifecycle of a request from formatting to execution
        and post-processing, allowing drivers to focus purely on the network call.
        """
        # Determine if this specific driver execution requires manual tool fallbacks
        use_native_tools = getattr(self, "use_native_tools", True)
        
        # If the local API server crashes on native tools, we inject the JSON schema
        # directly into the System Prompt so the model still knows how to use them.
        if tools and not use_native_tools:
            system = self._format_fallback_prompt(system, tools, messages)
            
        # Unify the message history (e.g. converting past tool calls to text if fallback is enabled)
        formatted_messages = self._format_messages(system, messages)
        
        # Map the user's unified ModelConfig into generic API kwargs (temperature, max_tokens, etc.)
        kwargs = self._map_config()
        
        # [HOOK]: The child driver executes the actual API call
        try:
            response = self.execute(formatted_messages, tools, **kwargs)
        except Exception as e:
            if tools and use_native_tools and self._is_tool_unsupported_error(e):
                self.use_native_tools = False
                system = self._format_fallback_prompt(system, tools, messages)
                formatted_messages = self._format_messages(system, messages)
                response = self.execute(formatted_messages, tools, **kwargs)
            else:
                raise e
        
        # [POST-PROCESSING]: Standardize deepseek/gemini style <think> blocks
        content = response.get("content", "")
        reasoning = response.get("reasoning_content")
        
        if not reasoning and content:
            parsed_content, parsed_reasoning = self._parse_reasoning(content)
            if parsed_reasoning:
                response["content"] = parsed_content
                response["reasoning_content"] = parsed_reasoning

        # [POST-PROCESSING]: Extract tool calls from text when the driver
        # returned none. This catches models that emit tool calls as text
        # (Hermes XML, JSON blocks, etc.) regardless of which driver is used.
        if tools and not response.get("tool_calls"):
            content = response.get("content", "")
            if content:
                parsed_calls, cleaned = self._parse_tool_calls_from_text(content)
                if parsed_calls:
                    response["tool_calls"] = parsed_calls
                    response["content"] = cleaned

        return response

    def request_stream(
        self,
        system: Optional[str],
        messages: List[Dict[str, Any]],
        tools: List[Any],
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Streaming variant of request(). Yields delta dicts from the driver,
        applying the same pre-processing and post-processing as request().

        Chunk protocol:
          {"type": "content_delta", "content": "..."}
          {"type": "reasoning_delta", "content": "..."}
          {"type": "done", "content": "...", "tool_calls": ..., "reasoning_content": ...}
        """
        use_native_tools = getattr(self, "use_native_tools", True)

        if tools and not use_native_tools:
            system = self._format_fallback_prompt(system, tools, messages)

        formatted_messages = self._format_messages(system, messages)
        kwargs = self._map_config()

        try:
            stream = self.execute_stream(formatted_messages, tools, **kwargs)
        except Exception as e:
            if tools and use_native_tools and self._is_tool_unsupported_error(e):
                self.use_native_tools = False
                system = self._format_fallback_prompt(system, tools, messages)
                formatted_messages = self._format_messages(system, messages)
                stream = self.execute_stream(formatted_messages, tools, **kwargs)
            else:
                raise e

        for chunk in stream:
            if chunk["type"] == "done":
                # Post-process reasoning extraction on the final assembled result
                content = chunk.get("content", "")
                reasoning = chunk.get("reasoning_content")
                if not reasoning and content:
                    parsed_content, parsed_reasoning = self._parse_reasoning(content)
                    if parsed_reasoning:
                        chunk["content"] = parsed_content
                        chunk["reasoning_content"] = parsed_reasoning
                # Post-process tool-call extraction from text
                if tools and not chunk.get("tool_calls"):
                    stream_content = chunk.get("content", "")
                    if stream_content:
                        parsed_calls, cleaned = self._parse_tool_calls_from_text(stream_content)
                        if parsed_calls:
                            chunk["tool_calls"] = parsed_calls
                            chunk["content"] = cleaned
            yield chunk

    def _is_tool_unsupported_error(self, e: Exception) -> bool:
        """Helper to check if an exception was caused by a lack of tool calling support."""
        err_msg = str(e).lower()
        tool_keywords = ["tool", "tools", "function", "functions"]
        parameter_keywords = [
            "unrecognized", "unexpected", "invalid", "not supported", 
            "validation", "extra fields", "cannot", "unknown", "400", "422",
            "not implemented", "bad request", "schema"
        ]
        
        has_tool = any(tk in err_msg for tk in tool_keywords)
        has_param = any(pk in err_msg for pk in parameter_keywords)
        
        if has_tool and has_param:
            return True
            
        # Inspect HTTP status code if present
        status_code = getattr(e, "status_code", None)
        if status_code is None:
            # Check response attribute
            response = getattr(e, "response", None)
            if response is not None:
                status_code = getattr(response, "status_code", None)
        if status_code in (400, 422) and has_tool:
            return True
            
        return False

    @abstractmethod
    def execute(
        self,
        formatted_messages: List[Dict[str, Any]],
        tools: List[Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Child classes must implement the API call and response extraction."""
        pass

    def execute_stream(
        self,
        formatted_messages: List[Dict[str, Any]],
        tools: List[Any],
        **kwargs: Any,
    ) -> Generator[Dict[str, Any], None, None]:
        """Streaming variant of execute(). Yields delta dicts.

        Child classes should override this to use their provider's native
        streaming API. The default implementation falls back to execute()
        and yields the full response as a single chunk.
        """
        result = self.execute(formatted_messages, tools, **kwargs)
        content = result.get("content", "")
        if content:
            yield {"type": "content_delta", "content": content}
        yield {
            "type": "done",
            "content": content,
            "tool_calls": result.get("tool_calls"),
            "reasoning_content": result.get("reasoning_content"),
        }

    def _format_messages(
        self, system: Optional[str], messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Standard translation of system and messages into a unified sequence."""
        formatted = []
        if system:
            formatted.append({"role": "system", "content": system})
        for m in messages:
            formatted.append(
                {"role": m["role"], "content": m.get("content", "")}
            )
            # tool calls are left intact for specific drivers to format
            if "tool_calls" in m:
                formatted[-1]["tool_calls"] = m["tool_calls"]
            if "tool_call_id" in m:
                formatted[-1]["tool_call_id"] = m["tool_call_id"]
                formatted[-1]["name"] = m.get("name")
        return formatted

    def _map_config(self) -> Dict[str, Any]:
        """Maps standard ModelConfig fields to generic kwargs."""
        kwargs = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.top_p is not None:
            kwargs["top_p"] = self.config.top_p
        if self.config.top_k is not None:
            kwargs["top_k"] = self.config.top_k
        if self.config.stop_sequences:
            kwargs["stop"] = self.config.stop_sequences
        if self.config.seed is not None:
            kwargs["seed"] = self.config.seed
        if self.config.response_format is not None:
            kwargs["response_format"] = self.config.response_format
        if self.config.presence_penalty != 0.0:
            kwargs["presence_penalty"] = self.config.presence_penalty
        if self.config.frequency_penalty != 0.0:
            kwargs["frequency_penalty"] = self.config.frequency_penalty
        if self.config.timeout is not None:
            kwargs["timeout"] = self.config.timeout
        if self.config.min_p is not None:
            kwargs["min_p"] = self.config.min_p
        if self.config.repeat_penalty is not None:
            kwargs["repetition_penalty"] = self.config.repeat_penalty
        if self.config.logit_bias is not None:
            kwargs["logit_bias"] = self.config.logit_bias

        return kwargs

    def _parse_reasoning(self, content: str) -> tuple[str, Optional[str]]:
        """Extracts <think> blocks and returns (cleaned_content, reasoning)."""
        match = re.search(
            r"<(?:think|\|think\|)>(.*?)</(?:think|\|think\|)>",
            content,
            re.DOTALL,
        )
        if match:
            reasoning = match.group(1).strip()
            cleaned_content = re.sub(
                r"<(?:think|\|think\|)>.*?</(?:think|\|think\|)>\s*",
                "",
                content,
                flags=re.DOTALL,
            ).strip()
            return cleaned_content, reasoning
        return content, None

    def _parse_tool_calls_from_text(
        self, content: str
    ) -> tuple[list[dict[str, Any]], str]:
        """Extract tool calls embedded as text in the model's response.

        This is the universal fallback for models that emit tool calls as text
        instead of through the API's native tool-calling mechanism.  It runs as
        post-processing in ``request()`` / ``request_stream()`` so **every
        driver** benefits automatically.

        Supported formats (tried in order):

        1. **Hermes / Qwen XML** – ``<tool_call>`` blocks containing either
           Qwen-style ``<function=…><parameter=…>`` tags *or* a JSON body.
        2. **JSON code fences** – tool-call dicts inside ````json … ```` or
           ````tool_call … ```` Markdown fences.
        3. **Bare JSON objects** – a top-level ``{"name": …, "arguments": …}``
           dict (or ``{"function": {"name": …}}`` wrapper).

        Returns:
            (tool_calls, cleaned_content) where *tool_calls* is a list of
            normalised ``{id, name, arguments}`` dicts (or an empty list if
            nothing was found) and *cleaned_content* is the original text with
            the matched regions stripped out.
        """
        tool_calls: list[dict[str, Any]] = []
        cleaned = content

        # ── 1. Hermes / Qwen  <tool_call> blocks ──────────────────────────
        tc_blocks = re.findall(
            r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL
        )
        if tc_blocks:
            for block in tc_blocks:
                parsed = self._parse_single_tool_block(block)
                if parsed:
                    tool_calls.append(parsed)
            if tool_calls:
                cleaned = re.sub(
                    r"<tool_call>.*?</tool_call>",
                    "",
                    cleaned,
                    flags=re.DOTALL,
                ).strip()
                return tool_calls, cleaned

        # ── 2. Markdown JSON / tool_call code fences ──────────────────────
        fence_blocks = re.findall(
            r"```(?:json|tool_call)\s*\n(.*?)```", content, re.DOTALL
        )
        for block in fence_blocks:
            parsed = self._parse_single_tool_block(block)
            if parsed:
                tool_calls.append(parsed)
        if tool_calls:
            cleaned = re.sub(
                r"```(?:json|tool_call)\s*\n.*?```",
                "",
                cleaned,
                flags=re.DOTALL,
            ).strip()
            return tool_calls, cleaned

        # ── 3. Bare JSON object with name + arguments ─────────────────────
        bare = re.search(
            r'\{\s*"(?:name|function)"\s*:', content, re.DOTALL
        )
        if bare:
            # Find the matching closing brace
            start = bare.start()
            depth = 0
            end = start
            for i, ch in enumerate(content[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            candidate = content[start:end]
            parsed = self._parse_single_tool_block(candidate)
            if parsed:
                tool_calls.append(parsed)
                cleaned = (content[:start] + content[end:]).strip()
                return tool_calls, cleaned

        return tool_calls, cleaned

    def _parse_single_tool_block(self, block: str) -> Optional[dict[str, Any]]:
        """Try to parse a single text block into a normalised tool-call dict.

        Handles:
        - JSON bodies  (``{"name": …, "arguments": …}``)
        - Qwen XML     (``<function=name><parameter=k>v</parameter></function>``)
        """
        block = block.strip()

        # Attempt JSON first
        try:
            data = json.loads(block)
            if isinstance(data, dict):
                return self._normalize_tool_call(data)
        except (json.JSONDecodeError, ValueError):
            pass

        # Attempt Qwen-style XML  <function=name> … </function>
        func_match = re.search(r"<function=([^>]+)>", block)
        if func_match:
            func_name = func_match.group(1).strip()
            params: dict[str, str] = {}
            for pm in re.finditer(
                r"<parameter=([^>]+)>(.*?)</parameter>", block, re.DOTALL
            ):
                params[pm.group(1).strip()] = pm.group(2).strip()
            return self._normalize_tool_call(
                {"name": func_name, "arguments": params}
            )

        return None

    def _normalize_tool_call(self, data: Any) -> Optional[dict[str, Any]]:
        """Normalise a parsed dict into the standard ``{id, name, arguments}``
        format consumed by the agent loop."""
        if not isinstance(data, dict):
            return None

        name = data.get("name")
        arguments = None
        if not name:
            # Some models nest under a "function" key
            func = data.get("function", {})
            name = func.get("name")
            arguments = func.get("arguments")
        else:
            arguments = data.get("arguments", data.get("parameters", {}))

        if not name:
            return None

        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        elif arguments is None:
            arguments = "{}"

        return {
            "id": data.get("id", "text_parsed_call"),
            "name": name,
            "arguments": arguments,
        }

    def _format_fallback_tools(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Formats tool calls as standard assistant/user text for models lacking native tool support."""
        formatted = []
        for m in messages:
            role = m["role"]
            content = m.get("content") or ""

            if role == "assistant" and m.get("tool_calls"):
                calls_text = []
                for tc in m["tool_calls"]:
                    calls_text.append(
                        f"[Called tool '{tc.get('name')}' with arguments: {tc.get('arguments')}]"
                    )
                calls_summary = "\n".join(calls_text)
                content = (
                    f"{content}\n{calls_summary}" if content else calls_summary
                )

            if role == "tool":
                role = "user"
                content = f"[Tool Result for '{m.get('name')}']:\n{content}"

            msg = {"role": role, "content": content}
            formatted.append(msg)
        return formatted

    def _convert_to_json_schema(self, func) -> Dict[str, Any]:
        """
        Introspects a raw Python callable, parsing its type hints and docstring,
        and dynamically translates it into a strict OpenAI-style JSON Schema.
        This provides the model with exact parameter requirements and descriptions.
        """
        if isinstance(func, dict):
            return func
        sig = inspect.signature(func)
        docstring = inspect.getdoc(func) or ""

        param_docs = {}
        if docstring:
            matches = re.finditer(
                r"(?:- )?(?::param )?([a-zA-Z0-9_]+)(?: \(.*?\))?[:\-]\s*(.*?)(?=\n\s*(?:[a-zA-Z0-9_]+[:\-]|:param|\Z))",
                docstring,
                re.DOTALL,
            )
            for match in matches:
                param_docs[match.group(1)] = match.group(2).strip()

        properties = {}
        required = []
        for name, param in sig.parameters.items():
            if name in ["top_k"]:
                continue
            p_type = "string"
            if param.annotation in (int, float):
                p_type = "number"
            elif param.annotation == bool:
                p_type = "boolean"

            desc = (
                param_docs.get(name) or f"The argument target value for {name}."
            )
            desc = re.sub(r"\s+", " ", desc).strip()

            properties[name] = {
                "type": p_type,
                "description": desc,
            }
            if param.default == inspect.Parameter.empty:
                required.append(name)

        main_desc = re.split(
            r"\n\s*(?:Args|Arguments|Parameters|:param)\s*:?", docstring
        )[0].strip()
        if not main_desc:
            main_desc = f"Execute function {func.__name__}"

        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": main_desc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def _get_serialized_tools(
        self, tools: List[Any], messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generates the final list of JSON Schemas for all tools.
        Crucially, this method inspects the conversation history (`messages`) for any
        hallucinated tool calls (tools the model invented) and automatically injects
        dummy JSON schemas for them to prevent strict API gateways from crashing the request.
        """
        serialized = []
        existing_names = set()
        for t in tools:
            t_unwrapped = t
            while hasattr(t_unwrapped, "__wrapped__"):
                t_unwrapped = t_unwrapped.__wrapped__
            if hasattr(t_unwrapped, "_dingir_schema"):
                schema = t_unwrapped._dingir_schema
                existing_names.add(schema.get("function", {}).get("name", ""))
                serialized.append(schema)
            else:
                try:
                    schema = self._convert_to_json_schema(t_unwrapped)
                    existing_names.add(
                        schema.get("function", {}).get("name", "")
                    )
                    serialized.append(schema)
                except Exception:
                    pass

        for m in messages:
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    name = tc.get("name")
                    if name and name not in existing_names:
                        existing_names.add(name)
                        serialized.append(
                            {
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "description": "Hallucinated or unavailable tool helper.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {},
                                    },
                                },
                            }
                        )
        return serialized

    def _format_fallback_prompt(
        self,
        system: Optional[str],
        tools: List[Any],
        messages: List[Dict[str, Any]],
    ) -> str:
        """
        Injects a strict instruction block and the JSON schemas directly into the
        system prompt. This is ONLY triggered if `use_native_tools=False`, giving
        local models a manual map of how to execute tools without relying on their backend.
        """
        if not tools:
            return system or ""

        tool_schemas = self._get_serialized_tools(tools, messages)

        tool_descriptions = []
        for schema in tool_schemas:
            func = schema.get("function", {})
            tool_descriptions.append(
                {
                    "name": func.get("name"),
                    "description": func.get("description"),
                    "parameters": func.get("parameters"),
                }
            )

        tool_descriptions_json = json.dumps(tool_descriptions, indent=2)

        tools_instruction_block = (
            "\n\n"
            "You can do function calling with the following functions:\n\n"
            f"{tool_descriptions_json}\n\n"
            "TOOL EXECUTION GUIDELINES:\n"
            "1. Analyze the user query and select the most appropriate "
            "tool from the toolset above. If no tool is needed or "
            "suitable, respond directly using your general knowledge.\n"
            "2. When executing a tool call, ensure you strictly adhere "
            "to the types, descriptions, and required constraints "
            "defined in the parameter schema.\n"
            "3. All arguments must be passed as a valid JSON object "
            "matching the defined parameter structure.\n"
            "4. Always execute the tool through the provider's native "
            "tool-calling interface. Do not simulate or mock tool "
            "responses in your text content."
        )
        return f"{system or ''}{tools_instruction_block}"
