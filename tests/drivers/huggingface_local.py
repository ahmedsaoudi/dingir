import os
import json
import pytest
from unittest.mock import MagicMock, patch

from dingir.config import ModelConfig
from dingir.agents.llms.hf_local import HuggingFaceLocal


class TestHuggingFaceLocalDriver:
    @patch("dingir.agents.llms.hf_local.os.environ.get")
    def test_lazy_load_text_generation(self, mock_env_get):
        mock_env_get.return_value = "local-hf-token"
        config = ModelConfig(max_tokens=50)
        driver = HuggingFaceLocal("Qwen/Qwen2.5-Coder-7B", config, task="text-generation")
        
        assert driver.task == "text-generation"
        assert driver.api_key == "local-hf-token"
        
        # Mock torch and transformers imports
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        
        mock_pipeline = MagicMock()
        mock_transformers = MagicMock()
        
        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "transformers": mock_transformers
        }):
            mock_transformers.pipeline = mock_pipeline
            driver._lazy_load()
            
            mock_pipeline.assert_called_once_with(
                "text-generation",
                model="Qwen/Qwen2.5-Coder-7B",
                device_map=None,
                torch_dtype=mock_torch.float32,
                token="local-hf-token"
            )

    @patch("dingir.agents.llms.hf_local.os.environ.get")
    def test_supports_native_tools_detection(self, mock_env_get):
        driver = HuggingFaceLocal("Qwen/Qwen-7B", ModelConfig())
        
        # Case 1: Template not loaded yet
        assert driver._supports_native_tools() is False
        
        # Case 2: No chat_template attribute
        driver._pipeline = MagicMock()
        driver._pipeline.tokenizer = MagicMock(spec=[])
        assert driver._supports_native_tools() is False
        
        # Case 3: Template contains 'tools' variable
        driver._pipeline.tokenizer.chat_template = "This is a jinja template containing tools references."
        assert driver._supports_native_tools() is True
        
        # Case 4: Template dict variant
        driver._pipeline.tokenizer.chat_template = {"tool_use": "variant template"}
        assert driver._supports_native_tools() is True

    @patch("dingir.agents.llms.hf_local.os.environ.get")
    def test_execute_and_normalize(self, mock_env_get):
        driver = HuggingFaceLocal("Hermes-Model", ModelConfig(temperature=0.0, max_tokens=100))
        
        mock_pipeline = MagicMock()
        driver._pipeline = mock_pipeline
        driver.use_native_tools = True
        
        # Mock applying template and executing
        mock_pipeline.tokenizer.apply_chat_template.return_value = "<prompt>"
        mock_pipeline.tokenizer.eos_token_id = 50256
        
        # Mock generated text outputs
        mock_pipeline.return_value = [
            {"generated_text": "<prompt>Generated assistant response"}
        ]
        
        # Mock tokenizer response parser
        mock_pipeline.tokenizer.parse_response.return_value = {
            "content": "Final content",
            "tool_calls": [
                {
                    "name": "web_search",
                    "parameters": {"query": "news"}
                }
            ]
        }
        
        res = driver.execute(
            [{"role": "user", "content": "news"}],
            tools=[]
        )
        
        assert res["content"] == "Final content"
        assert res["tool_calls"] == [
            {"id": "hf_local_call", "name": "web_search", "arguments": '{"query": "news"}'}
        ]
        
        # Verify pipeline call parameters
        mock_pipeline.assert_called_once_with(
            "<prompt>",
            max_new_tokens=100,
            pad_token_id=50256,
            temperature=0.01,
            do_sample=False
        )

    @patch("dingir.agents.llms.hf_local.os.environ.get")
    def test_embeddings(self, mock_env_get):
        driver = HuggingFaceLocal("BAAI/bge-large", ModelConfig(), task="feature-extraction")
        
        mock_torch = MagicMock()
        mock_transformers = MagicMock()
        
        # Setup model mock
        mock_model = MagicMock()
        mock_model.device = "cpu"
        
        # Mock attention mask expansion and math operations
        mock_token_embeddings = MagicMock()
        mock_token_embeddings.size.return_value = (1, 5, 128)
        
        mock_sum_embeddings = MagicMock()
        mock_sum_mask = MagicMock()
        mock_final_embeddings = MagicMock()
        mock_final_embeddings.cpu.return_value.numpy.return_value.tolist.return_value = [[0.1] * 128]
        
        # Override basic operators on MagicMocks for embedding math
        mock_torch.sum.return_value = mock_sum_embeddings
        mock_torch.clamp.return_value = mock_sum_mask
        mock_sum_embeddings.__truediv__.return_value = mock_final_embeddings
        
        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "transformers": mock_transformers
        }):
            driver._tokenizer = MagicMock()
            driver._tokenizer.return_value = {
                "attention_mask": MagicMock()
            }
            driver._model = mock_model
            
            mock_model.return_value = [mock_token_embeddings]
            
            embeddings = driver.embed(["text"])
            assert embeddings == [[0.1] * 128]
