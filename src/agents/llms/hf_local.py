import os
import json
from typing import Any, Dict, Generator, List, Optional, Tuple

from dingir.agents.llms.base import BaseLLM
from dingir.config import ModelConfig


class HuggingFaceLocal(BaseLLM):
    """Local Hugging Face Transformers driver.

    Runs models directly on the local machine using the ``transformers``
    library, with automatic GPU detection and lazy-loaded pipelines.
    This driver is designed for maximum versatility — it can drive any
    model architecture that ``transformers.pipeline`` supports, from
    causal-LM text generation to sentence embeddings.

    Supported tasks:
        * ``text-generation`` (default) — chat / completion via
          ``transformers.pipeline("text-generation", ...)``.
        * ``feature-extraction`` — sentence embeddings via a raw
          ``AutoModel`` + ``AutoTokenizer`` with mean-pooling.

    Tool calling:
        When the loaded tokenizer's ``chat_template`` references a
        ``tools`` Jinja variable (e.g. Hermes, Qwen, Mistral tool-use
        templates), the driver will pass tools natively through
        ``apply_chat_template``. Otherwise it falls back to the base
        class's text-injection strategy (``_format_fallback_prompt`` and
        ``_parse_tool_calls_from_text`` are both handled by BaseLLM's
        ``request()`` lifecycle — this driver does not duplicate them).

    Args:
        id: A HuggingFace model identifier (e.g. ``"Qwen/Qwen2.5-Coder-7B"``).
        config: A ``ModelConfig`` controlling temperature, max_tokens, etc.
        task: The pipeline task. Defaults to ``"text-generation"``.
        api_key: Optional HuggingFace token for gated models. Falls back
            to the ``HF_TOKEN`` environment variable.
        torch_dtype: Override the torch dtype (e.g. ``torch.bfloat16``).
            When ``None`` (the default), the driver auto-selects
            ``float16`` on CUDA and ``float32`` on CPU.
        device_map: Override the device map string passed to the pipeline.
            When ``None``, auto-selects ``"auto"`` on CUDA and ``None``
            (CPU) otherwise.
    """

    def __init__(
        self,
        id: str,
        config: ModelConfig,
        task: str = "text-generation",
        api_key: Optional[str] = None,
        torch_dtype: Any = None,
        device_map: Optional[str] = None,
    ):
        super().__init__(id, config)
        self.task = task
        self.api_key = api_key or os.environ.get("HF_TOKEN")
        self._torch_dtype_override = torch_dtype
        self._device_map_override = device_map

        # Lazily initialised on first call
        self._pipeline = None       # text-generation pipeline
        self._model = None          # raw model for embeddings
        self._tokenizer = None      # raw tokenizer for embeddings

        # Will be set after introspecting the chat template
        self.use_native_tools = False

    # ── Lazy Loading ──────────────────────────────────────────────────

    def _resolve_dtype_and_device(self) -> Tuple[Any, Optional[str]]:
        """Resolve torch dtype and device_map from overrides or GPU availability."""
        import torch

        has_cuda = torch.cuda.is_available()

        if self._torch_dtype_override is not None:
            dtype = self._torch_dtype_override
        else:
            dtype = torch.float16 if has_cuda else torch.float32

        if self._device_map_override is not None:
            device_map = self._device_map_override
        else:
            device_map = "auto" if has_cuda else None

        return dtype, device_map

    def _lazy_load(self) -> None:
        """Import torch/transformers and build the pipeline on first use.

        This avoids penalising import time for users who construct the
        driver instance but don't call it immediately (common in config
        composition).
        """
        import transformers

        dtype, device_map = self._resolve_dtype_and_device()

        self._pipeline = transformers.pipeline(
            self.task,
            model=self.id,
            device_map=device_map,
            torch_dtype=dtype,
            token=self.api_key,
        )

        # After loading, introspect the chat template for native tool support
        if self._supports_native_tools():
            self.use_native_tools = True

    def _lazy_load_embeddings(self) -> None:
        """Load a raw model + tokenizer for embedding/feature-extraction."""
        import transformers

        dtype, device_map = self._resolve_dtype_and_device()

        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.id, token=self.api_key
        )
        self._model = transformers.AutoModel.from_pretrained(
            self.id,
            device_map=device_map,
            torch_dtype=dtype,
            token=self.api_key,
        )
        self._model.eval()

    def _ensure_pipeline(self) -> None:
        """Ensure the text-generation pipeline is loaded."""
        if self._pipeline is None:
            self._lazy_load()

    # ── Tool-Support Introspection ────────────────────────────────────

    def _supports_native_tools(self) -> bool:
        """Check whether the tokenizer's chat template supports tools.

        Returns ``True`` if:
        - The ``chat_template`` is a dict (template variants, one of
          which is typically a tool-use template), **or**
        - The ``chat_template`` string contains the Jinja variable
          ``tools`` (used by Hermes / Qwen / Mistral tool templates).
        """
        if self._pipeline is None:
            return False

        tokenizer = self._pipeline.tokenizer
        template = getattr(tokenizer, "chat_template", None)
        if template is None:
            return False

        if isinstance(template, dict):
            return True

        if isinstance(template, str) and "tools" in template:
            return True

        return False

    # ── Shared Helpers ────────────────────────────────────────────────

    def _build_prompt(
        self,
        formatted_messages: List[Dict[str, Any]],
        tools: List[Any],
    ) -> str:
        """Apply the tokenizer's chat template to produce a prompt string.

        When native tool support is active, the serialized tool schemas
        are passed into the template so tool-aware templates (Hermes,
        Qwen, etc.) can render them into the prompt structure.
        """
        template_kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }

        if tools and self.use_native_tools:
            serialized_tools = self._get_serialized_tools(tools, formatted_messages)
            template_kwargs["tools"] = serialized_tools

        return self._pipeline.tokenizer.apply_chat_template(
            formatted_messages, **template_kwargs
        )

    def _build_gen_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        """Translate BaseLLM's generic kwargs into pipeline generation parameters.

        This is the local equivalent of what the Ollama driver does when
        remapping ``max_tokens`` → ``num_predict``. The base class's
        ``_map_config()`` has already produced the generic kwargs; this
        method translates them into ``transformers.pipeline`` semantics
        (e.g. ``max_tokens`` → ``max_new_tokens``).
        """
        temperature = kwargs.get("temperature", 0.0)
        if temperature <= 0:
            temperature = 0.01

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "pad_token_id": self._pipeline.tokenizer.eos_token_id,
            "temperature": temperature,
            "do_sample": temperature > 0.01,
        }

        if kwargs.get("top_p") is not None:
            gen_kwargs["top_p"] = kwargs["top_p"]
        if kwargs.get("top_k") is not None:
            gen_kwargs["top_k"] = kwargs["top_k"]
        if kwargs.get("repetition_penalty") is not None:
            gen_kwargs["repetition_penalty"] = kwargs["repetition_penalty"]
        if kwargs.get("stop"):
            gen_kwargs["stop_strings"] = kwargs["stop"]
        if kwargs.get("seed") is not None:
            import torch
            torch.manual_seed(kwargs["seed"])

        return gen_kwargs

    def _normalize_native_tool_calls(
        self, generated_text: str, tools: List[Any]
    ) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        """Parse native tool calls from the tokenizer's response parser.

        Only invoked when ``use_native_tools`` is True and the tokenizer
        exposes a ``parse_response`` method. The base class's
        ``_parse_tool_calls_from_text()`` handles the fallback case
        (Hermes XML, JSON fences, etc.) — this method is exclusively
        for structured native-template parsing.

        Returns:
            (content, tool_calls) where tool_calls may be ``None``.
        """
        if not (tools and self.use_native_tools):
            return generated_text, None

        tokenizer = self._pipeline.tokenizer
        if not hasattr(tokenizer, "parse_response"):
            return generated_text, None

        parsed = tokenizer.parse_response(generated_text)
        content = parsed.get("content", generated_text)
        raw_calls = parsed.get("tool_calls", [])

        if not raw_calls:
            return content, None

        tc_out = []
        for tc in raw_calls:
            arguments = tc.get("arguments", tc.get("parameters", {}))
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments)
            tc_out.append({
                "id": "hf_local_call",
                "name": tc.get("name"),
                "arguments": arguments,
            })

        return content, tc_out

    # ── Core Execution ────────────────────────────────────────────────

    def execute(
        self,
        formatted_messages: List[Dict[str, Any]],
        tools: List[Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run a local text-generation pipeline call.

        The method applies the tokenizer's chat template to convert the
        message list into a prompt string, then feeds it through the
        pipeline. If the model supports native tools, the generated text
        is parsed back through the tokenizer's ``parse_response`` helper
        to extract structured tool calls.

        Note: Fallback tool-schema injection into the system prompt and
        text-based tool-call extraction are both handled by the base
        class's ``request()`` lifecycle — this method only needs to
        handle native-template tool parsing.
        """
        self._ensure_pipeline()

        if not self.use_native_tools and tools:
            formatted_messages = self._format_fallback_tools(formatted_messages)

        prompt = self._build_prompt(formatted_messages, tools)
        gen_kwargs = self._build_gen_kwargs(**kwargs)

        # Execute the pipeline
        raw_output = self._pipeline(prompt, **gen_kwargs)

        # The pipeline returns a list of dicts; extract the generated text
        generated_text = raw_output[0]["generated_text"]

        # Strip the prompt prefix to isolate the assistant's response
        if isinstance(generated_text, str) and generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]

        # Parse native tool calls (base class handles text-based fallback)
        content, tc_out = self._normalize_native_tool_calls(generated_text, tools)

        response = {"content": content or "", "tool_calls": tc_out}

        self._log_raw_api_call(
            {"model": self.id, "prompt_length": len(prompt), **gen_kwargs},
            response,
        )

        return response

    def execute_stream(
        self,
        formatted_messages: List[Dict[str, Any]],
        tools: List[Any],
        **kwargs: Any,
    ) -> Generator[Dict[str, Any], None, None]:
        """Streaming variant using the ``TextIteratorStreamer``.

        Falls back to the base class's default ``execute_stream()``
        (which wraps ``execute()``) when the streamer is unavailable.
        """
        try:
            from transformers import TextIteratorStreamer
            import threading
        except ImportError:
            # Graceful degradation: base class falls back to execute()
            yield from super().execute_stream(formatted_messages, tools, **kwargs)
            return

        self._ensure_pipeline()

        if not self.use_native_tools and tools:
            formatted_messages = self._format_fallback_tools(formatted_messages)

        prompt = self._build_prompt(formatted_messages, tools)
        gen_kwargs = self._build_gen_kwargs(**kwargs)

        # Tokenize the prompt for the streamer path
        tokenizer = self._pipeline.tokenizer
        model = self._pipeline.model

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_length = inputs["input_ids"].shape[1]

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = {
            **inputs,
            **gen_kwargs,
            "streamer": streamer,
        }

        # Run generation in a background thread
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        full_content = ""
        for text_chunk in streamer:
            if text_chunk:
                full_content += text_chunk
                yield {"type": "content_delta", "content": text_chunk}

        thread.join()

        # Parse native tool calls from assembled output
        final_content, tc_out = self._normalize_native_tool_calls(full_content, tools)

        self._log_raw_api_call(
            {"model": self.id, "prompt_length": prompt_length, **gen_kwargs},
            {"content": final_content, "tool_calls": tc_out},
        )

        yield {
            "type": "done",
            "content": final_content,
            "tool_calls": tc_out,
            "reasoning_content": None,
        }

    # ── Embeddings ────────────────────────────────────────────────────

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using mean-pooling over token representations.

        Loads a raw ``AutoModel`` + ``AutoTokenizer`` on first call
        (separate from the text-generation pipeline) and applies
        attention-mask-weighted mean pooling to produce a single
        embedding vector per input text.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of embedding vectors, one per input text.
        """
        import torch

        if self._model is None or self._tokenizer is None:
            self._lazy_load_embeddings()

        encoded = self._tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )

        # Move inputs to same device as model
        device = self._model.device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)

        # outputs[0] = last_hidden_state  (batch, seq_len, hidden_dim)
        token_embeddings = outputs[0]

        # Mean pooling: weight by attention mask to ignore padding
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        sentence_embeddings = sum_embeddings / sum_mask

        return sentence_embeddings.cpu().numpy().tolist()
