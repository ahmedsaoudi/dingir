import pytest
from unittest.mock import MagicMock, patch

from dingir.config import ModelConfig
from dingir.agents.llms.openai_driver import OpenAI


class TestOpenAIDriver:
    @patch("dingir.agents.llms.openai_driver.OpenAIClient")
    def test_init_params(self, mock_client_cls):
        config = ModelConfig(max_retries=4, timeout=30.0)
        driver = OpenAI("gpt-4", config, api_key="test-api-key", base_url="https://api.openai.com/v1")
        
        mock_client_cls.assert_called_once_with(
            api_key="test-api-key",
            base_url="https://api.openai.com/v1",
            max_retries=4,
            timeout=30.0
        )
        assert driver.use_native_tools is True

    @patch("dingir.agents.llms.openai_driver.OpenAIClient")
    def test_payload_cleaning_and_formatting(self, mock_client_cls):
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance
        
        # Mock completions return value
        mock_choice = MagicMock()
        mock_choice.message.content = "Sure, I can search that for you."
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call-123"
        mock_tool_call.function.name = "web_search"
        mock_tool_call.function.arguments = '{"query": "news"}'
        mock_choice.message.tool_calls = [mock_tool_call]
        mock_instance.chat.completions.create.return_value.choices = [mock_choice]
        
        config = ModelConfig(temperature=0.7)
        driver = OpenAI("gpt-4", config, api_key="dummy")
        
        # Input messages containing non-standard keys like name=None, tool_calls=None, and tool_calls in flat format
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "search news", "tool_calls": None, "tool_call_id": None, "name": None},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call-123", "name": "web_search", "arguments": '{"query": "news"}'}], "tool_call_id": None, "name": None},
            {"role": "tool", "content": "some news result", "tool_calls": None, "tool_call_id": "call-123", "name": "web_search"}
        ]
        
        def mock_tool():
            """Search the web."""
            pass

        res = driver.execute(messages, tools=[mock_tool])
        
        # Verify result format
        assert res["content"] == "Sure, I can search that for you."
        assert res["tool_calls"] == [
            {"id": "call-123", "name": "web_search", "arguments": '{"query": "news"}'}
        ]
        
        # Verify the messages sent to the API are cleaned of extra keys and formatted to standard nested tool calls
        args, kwargs = mock_instance.chat.completions.create.call_args
        sent_messages = kwargs["messages"]
        
        # System message
        assert sent_messages[0] == {"role": "system", "content": "You are a helpful assistant."}
        # User message (keys like tool_calls/name/tool_call_id are stripped)
        assert sent_messages[1] == {"role": "user", "content": "search news"}
        # Assistant message (tool_calls are transformed from flat to standard nested layout)
        assert sent_messages[2] == {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-123",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query": "news"}'}
                }
            ]
        }
        # Tool message (extra keys like name and tool_calls are stripped, tool_call_id is preserved)
        assert sent_messages[3] == {
            "role": "tool",
            "content": "some news result",
            "tool_call_id": "call-123"
        }

    @patch("dingir.agents.llms.openai_driver.OpenAIClient")
    def test_fallback_native_tools_disabled(self, mock_client_cls):
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance
        
        mock_choice = MagicMock()
        mock_choice.message.content = "Direct text reply"
        mock_choice.message.tool_calls = None
        mock_instance.chat.completions.create.return_value.choices = [mock_choice]
        
        driver = OpenAI("gpt-4", ModelConfig(), api_key="dummy")
        driver.use_native_tools = False
        
        def dummy_tool():
            pass

        messages = [{"role": "user", "content": "hi"}]
        driver.execute(messages, tools=[dummy_tool])
        
        # Verify no tools parameter is sent to completions API
        args, kwargs = mock_instance.chat.completions.create.call_args
        assert "tools" not in kwargs

    @patch("dingir.agents.llms.openai_driver.OpenAIClient")
    def test_embeddings(self, mock_client_cls):
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance
        
        mock_instance.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3])
        ]
        
        driver = OpenAI("text-embedding-ada-002", ModelConfig(), api_key="dummy")
        embeddings = driver.embed(["hello"])
        assert embeddings == [[0.1, 0.2, 0.3]]
        mock_instance.embeddings.create.assert_called_once_with(
            input=["hello"],
            model="text-embedding-ada-002"
        )
