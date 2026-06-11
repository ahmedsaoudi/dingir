import pytest
from unittest.mock import MagicMock, patch

from dingir.config import ModelConfig
from dingir.agents.llms.gemini_driver import Gemini


class TestGeminiDriver:
    @patch("dingir.agents.llms.gemini_driver.Client")
    def test_init_and_kwargs_mapping(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        config = ModelConfig(temperature=0.5, max_tokens=150, stop_sequences=["\n"])
        driver = Gemini("gemini-2.5", config, api_key="gemini-key")
        
        mock_client_cls.assert_called_once_with(api_key="gemini-key")
        
        # Mock response
        mock_response = MagicMock()
        mock_response.text = "Gemini content response"
        mock_response.function_calls = None
        mock_client.models.generate_content.return_value = mock_response
        
        messages = [
            {"role": "system", "content": "be a coder"},
            {"role": "user", "content": "code hello world"}
        ]
        
        res = driver.execute(messages, tools=[], temperature=0.5, max_tokens=150, stop=["\n"])
        assert res["content"] == "Gemini content response"
        
        # Verify generate_content call mapping
        args, kwargs = mock_client.models.generate_content.call_args
        assert kwargs["model"] == "gemini-2.5"
        
        gen_config = kwargs["config"]
        assert gen_config.temperature == 0.5
        assert gen_config.max_output_tokens == 150
        assert gen_config.stop_sequences == ["\n"]
        assert gen_config.system_instruction == "be a coder"

    @patch("dingir.agents.llms.gemini_driver.Client")
    def test_tool_calling(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        driver = Gemini("gemini-2.5", ModelConfig())
        
        mock_response = MagicMock()
        mock_response.text = ""
        mock_call = MagicMock()
        mock_call.name = "fetch_webpage"
        mock_call.args = {"url": "http://test.com"}
        mock_response.function_calls = [mock_call]
        mock_client.models.generate_content.return_value = mock_response
        
        def fetch_webpage(url: str):
            """Fetch webpage."""
            pass

        res = driver.execute(
            [{"role": "user", "content": "get page"}],
            tools=[fetch_webpage]
        )
        
        assert res["tool_calls"] == [
            {"id": "gemini_call", "name": "fetch_webpage", "arguments": {"url": "http://test.com"}}
        ]
        
        # Verify tools configuration passed to API
        args, kwargs = mock_client.models.generate_content.call_args
        gen_config = kwargs["config"]
        assert len(gen_config.tools) == 1
        decl = gen_config.tools[0].function_declarations[0]
        assert decl.name == "fetch_webpage"
