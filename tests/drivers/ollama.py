import pytest
from unittest.mock import MagicMock, patch

from dingir.config import ModelConfig
from dingir.agents.llms.ollama_driver import Ollama


class TestOllamaDriver:
    @patch("dingir.agents.llms.ollama_driver.ollama")
    def test_init_and_execute(self, mock_ollama):
        config = ModelConfig(temperature=0.8, max_tokens=200)
        driver = Ollama("llama3", config, base_url="http://localhost:11434")
        
        mock_ollama.Client.assert_called_once_with(host="http://localhost:11434")
        
        mock_client = mock_ollama.Client.return_value
        mock_client.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "Ollama response",
                "tool_calls": [
                    {
                        "function": {
                            "name": "calculator",
                            "arguments": {"expr": "1+1"}
                        }
                    }
                ]
            }
        }
        
        def calculator(expr: str):
            pass
            
        res = driver.execute(
            [{"role": "user", "content": "calc"}],
            tools=[calculator],
            temperature=0.8,
            max_tokens=200
        )
        
        assert res["content"] == "Ollama response"
        assert res["tool_calls"] == [
            {"id": "ollama_call", "name": "calculator", "arguments": {"expr": "1+1"}}
        ]
        
        # Check arguments sent to ollama chat endpoint
        mock_client.chat.assert_called_once()
        args, kwargs = mock_client.chat.call_args
        assert kwargs["model"] == "llama3"
        assert kwargs["options"] == {"temperature": 0.8, "num_predict": 200}
        assert "tools" in kwargs
