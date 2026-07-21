"""
API usage and costs are a real bottle neck for production applications that runs
agents, sub-agents, etc. This data is now availabe out of the box from the agent
run response.

Also refer to this amazing implementation by @bilge, DevRel Haystack on the same:
(Building a Cost-Aware Agent with Hooks)[https://haystack.deepset.ai/cookbook/cost_aware_agent]
"""

from haystack.tools import tool
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage

from utils import cohere_llm_generator


@tool
def word_count(text: str) -> str:
    """Count words in a string."""
    return str(len(text.split()))


agent_with_tool = Agent(chat_generator=cohere_llm_generator(), tools=[word_count])

result = agent_with_tool.run(
    messages=[ChatMessage.from_user("Count words in: hello there")]
)
print("Answer:", result["last_message"].text)
print("Token usage:", result["token_usage"])
print("Step count:", result["step_count"])
print("Tool call counts:", result["tool_call_counts"])
