import pytest
from dingir.config import ModelConfig
from dingir.agents.llms.base import BaseLLM
from dingir.agents.llms.openai_driver import OpenAI


class DummyLLM(BaseLLM):
    def request(self, system, messages, tools):
        return {"content": "dummy response"}


def test_explicit_args_tracking():
    # Only temperature explicitly set
    c1 = ModelConfig(temperature=0.7)
    assert c1._explicitly_set == {"temperature"}
    assert c1.temperature == 0.7
    assert c1.max_tokens == 1024  # default value

    # Multiple fields set
    c2 = ModelConfig(
        max_tokens=2048,
        seed=42,
        stop_sequences=["\n"],
        min_p=0.05,
        repeat_penalty=1.1,
    )
    assert c2._explicitly_set == {
        "max_tokens",
        "seed",
        "stop_sequences",
        "min_p",
        "repeat_penalty",
    }
    assert c2.max_tokens == 2048
    assert c2.seed == 42
    assert c2.stop_sequences == ["\n"]
    assert c2.min_p == 0.05
    assert c2.repeat_penalty == 1.1
    assert c2.temperature == 0.0  # default value

    # No fields explicitly set (rely on defaults)
    c3 = ModelConfig()
    assert c3._explicitly_set == set()


def test_config_merge_rightmost_precedence():
    c_common = ModelConfig(
        max_tokens=2048, temperature=0.2, presence_penalty=0.5
    )
    c_specific = ModelConfig(
        temperature=0.7, timeout=30.0, min_p=0.1, repeat_penalty=1.2
    )

    # Merge [c_common, c_specific]
    # Rightmost (c_specific) should override temperature (0.2 -> 0.7)
    # Non-overlapping fields (max_tokens=2048, presence_penalty=0.5 from c_common; timeout=30.0, min_p=0.1, repeat_penalty=1.2 from c_specific) should combine
    # All other fields should be their default values
    merged = ModelConfig.merge([c_common, c_specific])

    assert merged.temperature == 0.7
    assert merged.max_tokens == 2048
    assert merged.presence_penalty == 0.5
    assert merged.timeout == 30.0
    assert merged.min_p == 0.1
    assert merged.repeat_penalty == 1.2
    assert merged.top_p is None  # default
    assert merged.stop_sequences == []  # default


def test_config_merge_none_ignored():
    c1 = ModelConfig(temperature=0.7)
    merged = ModelConfig.merge([c1, None])
    assert merged.temperature == 0.7


def test_invalid_config_type_raises_error():
    # ModelConfig.merge should reject non-ModelConfig types (such as dicts)
    with pytest.raises(
        TypeError, match="Expected ModelConfig instance, got dict"
    ):
        ModelConfig.merge([ModelConfig(temperature=0.7), {"temperature": 0.5}])

    # BaseLLM should reject dictionary configs
    with pytest.raises(
        TypeError,
        match="Configuration must be a single ModelConfig or a list of ModelConfigs",
    ):
        DummyLLM("dummy-id", {"temperature": 0.5})

    # BaseLLM should reject list containing dictionary configs
    with pytest.raises(
        TypeError, match="Expected ModelConfig instance, got dict"
    ):
        DummyLLM(
            "dummy-id", [ModelConfig(temperature=0.7), {"temperature": 0.5}]
        )


def test_driver_composite_config():
    # Verify that a driver (e.g. OpenAI) successfully accepts and resolves composite configurations
    c_common = ModelConfig(max_tokens=4096, temperature=0.1)
    c_specific = ModelConfig(temperature=0.9, top_p=0.95)

    model = OpenAI("gpt-4o", config=[c_common, c_specific], api_key="dummy-key")
    assert model.id == "gpt-4o"
    assert model.config.max_tokens == 4096
    assert model.config.temperature == 0.9
    assert model.config.top_p == 0.95
    assert model.config.presence_penalty == 0.0  # default


def test_openai_driver_parameter_conversion():
    from unittest.mock import MagicMock, patch

    # 1. Test client instantiation with max_retries and timeout
    c = ModelConfig(max_retries=5, timeout=120.0)
    with patch(
        "dingir.agents.llms.openai_driver.OpenAIClient"
    ) as mock_client_cls:
        model = OpenAI("gpt-4o", config=c, api_key="dummy-key")
        mock_client_cls.assert_called_once_with(
            api_key="dummy-key", max_retries=5, timeout=120.0
        )

    # 2. Test request parameters mapping
    c_request = ModelConfig(
        temperature=0.8,
        max_tokens=500,
        top_p=0.9,
        stop_sequences=["\n", "STOP"],
        presence_penalty=1.5,
        frequency_penalty=-1.0,
        seed=123,
        response_format={"type": "json_object"},
        logit_bias={"50256": -100.0},
        timeout=15.0,
    )
    with patch(
        "dingir.agents.llms.openai_driver.OpenAIClient"
    ) as mock_client_cls:
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance
        model = OpenAI("gpt-4o", config=c_request, api_key="dummy-key")

        # Mock create response
        mock_choice = MagicMock()
        mock_choice.message.content = "hello"
        mock_choice.message.tool_calls = None
        mock_instance.chat.completions.create.return_value.choices = [
            mock_choice
        ]

        res = model.request(
            system="sys", messages=[{"role": "user", "content": "hi"}], tools=[]
        )
        assert res["content"] == "hello"

        # Verify all mapped arguments in completions.create
        mock_instance.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
            temperature=0.8,
            max_tokens=500,
            top_p=0.9,
            stop=["\n", "STOP"],
            presence_penalty=1.5,
            frequency_penalty=-1.0,
            seed=123,
            response_format={"type": "json_object"},
            logit_bias={"50256": -100.0},
            timeout=15.0,
        )


def test_openai_compatible_driver():
    from unittest.mock import MagicMock, patch
    from dingir.agents.llms.openai_compatible import OpenAICompatible
    import os

    # 1. Test parameter passing and env var fallbacks
    c = ModelConfig(max_retries=3, timeout=60.0)

    # Fallback to OPENAI_COMPATIBLE_API_KEY and OPENAI_COMPATIBLE_BASE_URL
    with patch.dict(
        os.environ,
        {
            "OPENAI_COMPATIBLE_API_KEY": "compat-key",
            "OPENAI_COMPATIBLE_BASE_URL": "https://api.compat.com/v1",
        },
    ):
        with patch(
            "dingir.agents.llms.openai_compatible.OpenAIClient"
        ) as mock_client_cls:
            model = OpenAICompatible("deepseek-chat", config=c)
            mock_client_cls.assert_called_once_with(
                api_key="compat-key",
                base_url="https://api.compat.com/v1",
                max_retries=3,
                timeout=60.0,
            )

    # Fallback to OPENAI_API_KEY if compat key is not present
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "openai-key",
            "OPENAI_COMPATIBLE_BASE_URL": "https://api.compat.com/v1",
        },
        clear=True,
    ):
        with patch(
            "dingir.agents.llms.openai_compatible.OpenAIClient"
        ) as mock_client_cls:
            model = OpenAICompatible("deepseek-chat", config=c)
            mock_client_cls.assert_called_once_with(
                api_key="openai-key",
                base_url="https://api.compat.com/v1",
                max_retries=3,
                timeout=60.0,
            )

    # Direct constructor arguments override env variables
    with patch.dict(
        os.environ,
        {
            "OPENAI_COMPATIBLE_API_KEY": "compat-key",
            "OPENAI_COMPATIBLE_BASE_URL": "https://api.compat.com/v1",
        },
    ):
        with patch(
            "dingir.agents.llms.openai_compatible.OpenAIClient"
        ) as mock_client_cls:
            model = OpenAICompatible(
                "deepseek-chat",
                config=c,
                api_key="override-key",
                base_url="https://override.com/v1",
            )
            mock_client_cls.assert_called_once_with(
                api_key="override-key",
                base_url="https://override.com/v1",
                max_retries=3,
                timeout=60.0,
            )
