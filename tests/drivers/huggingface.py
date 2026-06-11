import os
import pytest
from unittest.mock import MagicMock, patch

from dingir.config import ModelConfig
from dingir.agents.llms.hf_driver import HuggingFace


class TestHuggingFaceDriver:
    @patch("dingir.agents.llms.hf_driver.InferenceClient")
    @patch.dict(os.environ, {"HF_TOKEN": "fallback-token"})
    def test_init_and_execute(self, mock_client_cls):
        config = ModelConfig(temperature=0.0)
        driver = HuggingFace("meta-llama/Llama-3-8B-Instruct", config)
        
        mock_client_cls.assert_called_once_with(
            model="meta-llama/Llama-3-8B-Instruct",
            token="fallback-token"
        )
        
        mock_client = mock_client_cls.return_value
        mock_choice = MagicMock()
        mock_choice.message.content = "HF reply"
        mock_tool_call = MagicMock()
        mock_tool_call.id = "hf-call"
        mock_tool_call.function.name = "my_tool"
        mock_tool_call.function.arguments = '{"param": 1}'
        mock_choice.message.tool_calls = [mock_tool_call]
        mock_client.chat_completion.return_value.choices = [mock_choice]
        
        def my_tool(param: int):
            pass

        res = driver.execute(
            [{"role": "user", "content": "test"}],
            tools=[my_tool],
            temperature=0.0
        )
        
        assert res["content"] == "HF reply"
        assert res["tool_calls"] == [
            {"id": "hf-call", "name": "my_tool", "arguments": '{"param": 1}'}
        ]
        
        mock_client.chat_completion.assert_called_once()
        args, kwargs = mock_client.chat_completion.call_args
        assert kwargs["temperature"] == 0.01  # temperature <= 0 converted to 0.01
        assert "tools" in kwargs
