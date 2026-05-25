import os
from typing import List, Dict, Any, Optional
from dingir.agents.llms.base import BaseLLM
from dingir.config import ModelConfig

class HuggingFaceLocal(BaseLLM):
    """
    Django-grade driver for downloading and executing Hugging Face models 
    completely locally on your machine (supports both CPU and CUDA).
    """
    def __init__(self, id: str, config: ModelConfig, task: str = "text-generation"):
        super().__init__(id, config)
        self.task = task
        self._pipeline = None
        self._tokenizer = None
        self._model = None

    def _lazy_load(self):
        """Downloads weights to cache directory and builds pipelines only when called."""
        if self._pipeline is not None or self._model is not None:
            return

        import torch
        from transformers import pipeline, AutoTokenizer, AutoModel

        # Set automatic hardware device placement structures
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.task == "text-generation":
            self._pipeline = pipeline(
                "text-generation",
                model=self.id,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
        elif self.task == "feature-extraction":
            self._tokenizer = AutoTokenizer.from_pretrained(self.id)
            self._model = AutoModel.from_pretrained(self.id).to(device)

    def request(self, system: Optional[str], messages: List[Dict[str, Any]], tools: List[Any]) -> Dict[str, Any]:
        self._lazy_load()
        
        # Format conversation into standard template syntax
        formatted_messages = []
        if system:
            formatted_messages.append({"role": "system", "content": system})
        for m in messages:
            formatted_messages.append({"role": m["role"], "content": m["content"]})

        # Apply the model's native chat template
        prompt = self._pipeline.tokenizer.apply_chat_template(
            formatted_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        outputs = self._pipeline(
            prompt,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature if self.config.temperature > 0 else 0.01,
            do_sample=self.config.temperature > 0,
            pad_token_id=self._pipeline.tokenizer.eos_token_id
        )

        # Extract only the newly generated token content substring
        generated_text = outputs[0]["generated_text"][len(prompt):]
        return {"content": generated_text.strip(), "tool_calls": None}

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._lazy_load()
        import torch
        
        inputs = self._tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        # Ensure tensor variables match model placement execution devices
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            model_output = self._model(**inputs)
            # Perform mean pooling to generate high-quality text embedding vectors
            token_embeddings = model_output[0]
            input_mask_expanded = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = (sum_embeddings / sum_mask).cpu().numpy().tolist()
            
        return embeddings