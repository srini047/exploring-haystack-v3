from haystack_integrations.components.generators.cohere import CohereChatGenerator


def cohere_llm_generator() -> CohereChatGenerator:
    return CohereChatGenerator(model="command-a-03-2025")
