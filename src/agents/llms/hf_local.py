import os
from typing import Any, Dict, List, Optional

from dingir.agents.llms.base import BaseLLM
from dingir.config import ModelConfig


class HuggingFaceLocal(BaseLLM):
    """Driver for downloading and executing Hugging Face models
    completely locally on your machine (supports both CPU and CUDA).
    """

    def __init__(
        self,
        id: str,
        config: ModelConfig,
        task: str = "text-generation",
        api_key: Optional[str] = None,
        native_tools: Optional[bool] = None,
    ):
        super().__init__(id, config)
        self.task = task
        self.api_key = api_key or os.environ.get("HF_TOKEN")
        self.use_native_tools = native_tools if native_tools is not None else False
        self._pipeline = None
        self._tokenizer = None
        self._model = None

    def _lazy_load(self):
        """Downloads weights to cache directory and builds pipelines only when called."""
        if self._pipeline is not None or self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer, pipeline

        # Set automatic hardware device placement structures
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.task == "text-generation":
            self._pipeline = pipeline(
                "text-generation",
                model=self.id,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16
                if device == "cuda"
                else torch.float32,
                token=self.api_key,
            )
        elif self.task == "feature-extraction":
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.id, token=self.api_key
            )
            self._model = AutoModel.from_pretrained(
                self.id, token=self.api_key
            ).to(device)

    def execute(
        self,
        formatted_messages: List[Dict[str, Any]],
        tools: List[Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        self._lazy_load()

        if not self.use_native_tools and tools:
            formatted_messages = self._format_fallback_tools(formatted_messages)

        # Apply the model's native chat template
        template_kwargs = {
            "conversation": formatted_messages,
            "tokenize": False,
            "add_generation_prompt": True
        }
        if tools:
            template_kwargs["tools"] = self._get_serialized_tools(tools, formatted_messages)

        prompt = self._pipeline.tokenizer.apply_chat_template(**template_kwargs)

        pipeline_kwargs = {
            "max_new_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "pad_token_id": self._pipeline.tokenizer.eos_token_id,
        }
        
        temperature = kwargs.get("temperature", self.config.temperature)
        if temperature > 0:
            pipeline_kwargs["temperature"] = temperature
            pipeline_kwargs["do_sample"] = True
        else:
            pipeline_kwargs["temperature"] = 0.01
            pipeline_kwargs["do_sample"] = False
            
        if "repetition_penalty" in kwargs:
            pipeline_kwargs["repetition_penalty"] = kwargs["repetition_penalty"]
        if "top_p" in kwargs:
            pipeline_kwargs["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            pipeline_kwargs["top_k"] = kwargs["top_k"]
        if "stop" in kwargs and kwargs["stop"]:
            pipeline_kwargs["stop_strings"] = kwargs["stop"]
            pipeline_kwargs["tokenizer"] = self._pipeline.tokenizer

        if "seed" in kwargs and kwargs["seed"] is not None:
            import torch

            torch.manual_seed(kwargs["seed"])

        outputs = self._pipeline(prompt, **pipeline_kwargs)

        # Extract only the newly generated token content substring
        generated_text = outputs[0]["generated_text"][len(prompt) :]
        return {"content": generated_text.strip(), "tool_calls": None}

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._lazy_load()
        import torch

        inputs = self._tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        # Ensure tensor variables match model placement execution devices
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            model_output = self._model(**inputs)
            # Perform mean pooling to generate high-quality text embedding vectors
            token_embeddings = model_output[0]
            input_mask_expanded = (
                inputs["attention_mask"]
                .unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
            )
            sum_embeddings = torch.sum(
                token_embeddings * input_mask_expanded, 1
            )
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = (sum_embeddings / sum_mask).cpu().numpy().tolist()

        return embeddings
