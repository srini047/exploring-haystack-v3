from typing import Annotated

from haystack.components.agents import Agent
from haystack_integrations.components.generators.cohere import CohereChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage
from haystack.tools import ComponentTool, tool
from haystack_integrations.components.websearch.serperdev import SerperDevWebSearch

research_agent = Agent(
    chat_generator=CohereChatGenerator(model="command-a-03-2025"),
    tools=[
        ComponentTool(
            component=SerperDevWebSearch(
                top_k=3,
            ),
            name="web_search",
            description="Search the web for current information on any topic",
        ),
    ],
    system_prompt="You are a research specialist. Search the web to find information.",
)


@tool
def research(query: Annotated[str, "The research question to investigate"]) -> str:
    """Research a topic and return a summary of findings."""
    try:
        result = research_agent.run(messages=[ChatMessage.from_user(query)])
        return result["last_message"].text
    except Exception as e:
        return f"Research failed: {e}"


coordinator = Agent(
    chat_generator=CohereChatGenerator(model="command-a-03-2025"),
    tools=[research],
    system_prompt="You are a coordinator. Delegate research tasks to the research tool.",
    streaming_callback=print_streaming_chunk,
)

result = coordinator.run(
    messages=[
        ChatMessage.from_user("What are the latest developments in Haystack AI?"),
    ],
)

"""
result = {
    "messages": [
        ChatMessage(
            _role=ChatRole.SYSTEM,
            _content=[
                TextContent(
                    text="You are a coordinator. Delegate research tasks to the research tool."
                )
            ],
        ),

        ChatMessage(
            _role=ChatRole.USER,
            _content=[
                TextContent(
                    text="What are the latest developments in Haystack AI?"
                )
            ],
        ),

        ChatMessage(
            _role=ChatRole.ASSISTANT,
            _content=[
                TextContent(
                    text="I will research the latest developments in Haystack AI."
                ),
                ToolCall(
                    tool_name="research",
                    arguments={
                        "query": "latest developments in Haystack AI"
                    },
                    id="research_cg499cpk26bz"
                )
            ],
            _meta={
                "model": "command-a-03-2025",
                "finish_reason": "tool_calls",
                "usage": {
                    "prompt_tokens": 60,
                    "completion_tokens": 22
                }
            }
        ),

        ChatMessage(
            _role=ChatRole.TOOL,
            _content=[
                ToolCallResult(
                    result="Haystack is an open-source AI orchestration framework...",
                    error=False
                )
            ]
        ),

        ChatMessage(
            _role=ChatRole.ASSISTANT,
            _content=[
                TextContent(
                    text="Haystack is an open-source AI orchestration framework..."
                )
            ],
            _meta={
                "model": "command-a-03-2025",
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 201,
                    "completion_tokens": 128
                }
            }
        )
    ],

    "step_count": 2,

    "token_usage": {
        "prompt_tokens": 261,
        "completion_tokens": 150
    },

    "tool_call_counts": {
        "research": 1
    },

    "last_message": ChatMessage(...)
}
"""
