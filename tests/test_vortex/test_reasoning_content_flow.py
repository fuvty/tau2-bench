from types import SimpleNamespace

from tau2.data_model.message import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.user.user_simulator_base import UserState
from tau2.utils import llm_utils
from tau2.utils.display import MarkdownDisplay


def test_to_litellm_messages_only_serializes_assistant_reasoning_content():
    messages = [
        SystemMessage(role="system", content="system"),
        UserMessage(
            role="user",
            content="user visible",
            reasoning_content="user hidden",
        ),
        AssistantMessage(
            role="assistant",
            content=None,
            reasoning_content="assistant hidden",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="get_time",
                    arguments={},
                )
            ],
        ),
    ]

    serialized = llm_utils.to_litellm_messages(messages)

    assert serialized[1] == {"role": "user", "content": "user visible"}
    assert serialized[2]["role"] == "assistant"
    assert serialized[2]["reasoning_content"] == "assistant hidden"
    assert serialized[2]["tool_calls"][0]["function"]["name"] == "get_time"


def test_generate_preserves_reasoning_content_from_provider(monkeypatch):
    class FakeResponse(dict):
        def __init__(self):
            super().__init__({"usage": None})
            self.choices = [
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        role="assistant",
                        content="visible answer",
                        reasoning_content="hidden answer reasoning",
                        tool_calls=None,
                    ),
                )
            ]

        def to_dict(self):
            return {"choices": [{"message": {"reasoning_content": "hidden"}}]}

    monkeypatch.setattr(llm_utils, "completion", lambda **_: FakeResponse())
    monkeypatch.setattr(llm_utils, "get_response_cost", lambda _: 0.0)
    monkeypatch.setattr(
        llm_utils,
        "get_response_usage",
        lambda _: {"completion_tokens": 1, "prompt_tokens": 1},
    )

    response = llm_utils.generate(
        model="openai/Qwen/Qwen3-32B",
        messages=[UserMessage(role="user", content="hello")],
    )

    assert response.content == "visible answer"
    assert response.reasoning_content == "hidden answer reasoning"


def test_user_state_flip_roles_preserves_same_role_reasoning_only():
    state = UserState(
        system_messages=[],
        messages=[
            UserMessage(
                role="user",
                content=None,
                reasoning_content="user tool reasoning",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="lookup",
                        arguments={"id": 1},
                        requestor="user",
                    )
                ],
            ),
            ToolMessage(
                role="tool",
                id="call_1",
                content="tool result",
                requestor="user",
            ),
            AssistantMessage(
                role="assistant",
                content="agent visible text",
                reasoning_content="agent hidden reasoning",
            ),
        ],
    )

    flipped = state.flip_roles()

    assert isinstance(flipped[0], AssistantMessage)
    assert flipped[0].reasoning_content == "user tool reasoning"
    assert flipped[0].tool_calls is not None

    assert isinstance(flipped[1], ToolMessage)
    assert flipped[1].content == "tool result"

    assert isinstance(flipped[2], UserMessage)
    assert flipped[2].content == "agent visible text"
    assert flipped[2].reasoning_content is None


def test_markdown_display_hides_reasoning_by_default_and_shows_it_on_request():
    message = AssistantMessage(
        role="assistant",
        content="visible answer",
        reasoning_content="hidden reasoning",
    )

    hidden = MarkdownDisplay.display_messages([message])
    shown = MarkdownDisplay.display_messages([message], show_reasoning_content=True)

    assert "hidden reasoning" not in hidden
    assert "**Reasoning**" in shown
    assert "hidden reasoning" in shown
