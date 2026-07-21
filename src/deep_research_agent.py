"""
Note: agent-pack-haystack 0.0.1's deep_research prompts use a Jinja {% now %}
tag but don't declare jinja2-time as a dependency. `TemplateSyntaxError`
Run `uv add jinja2-time` to avoid error.
"""

from haystack.dataclasses import ChatMessage
from haystack_integrations.agent_pack import create_deep_research_agent
from utils import cohere_llm_generator


research_agent = create_deep_research_agent(
    scope_llm=cohere_llm_generator(),
    orchestrator_llm=cohere_llm_generator(),
    researcher_llm=cohere_llm_generator(),
    summarizer_llm=cohere_llm_generator(),
    writer_llm=cohere_llm_generator(),
    max_concurrent_researchers=1,
    max_orchestrator_steps=1,
    max_content_length=1000,
)

result = research_agent.run(
    messages=[ChatMessage.from_user("What are the latest trends in Agentic AI?")]
)

print("-------")
print(result)
