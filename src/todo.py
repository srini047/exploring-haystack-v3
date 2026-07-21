import asyncio

from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage

from utils import cohere_llm_generator


# --- 5. Stream the final report, async ---------------------------------------
async def run_streaming():
    streaming_agent = Agent(chat_generator=cohere_llm_generator())
    result = await streaming_agent.run_async(
        messages=[ChatMessage.from_user("Say hello in one sentence.")]
    )
    print("\n--- Final ---")
    print(result["last_message"].text)


asyncio.run(run_streaming())
