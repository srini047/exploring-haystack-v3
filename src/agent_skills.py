"""
Makes use of already available ``skills/`` directory to load each skills and
convert it into a usable tool for the Agent that is invoked based on the metadata.

Add skills either globally or project level by follwing the TUI:
    ```npx skills add srini047/haystack-skills@haystack```
"""

from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.tools import SkillToolset
from haystack.skill_stores.file_system import FileSystemSkillStore

from utils import cohere_llm_generator

SKILLS_PATH = ".agents/skills"  # Change as per you installed location

skill_store = FileSystemSkillStore(SKILLS_PATH)
skills_toolset = SkillToolset(skill_store)

print("Available skills are:")
available_skills = skill_store.list_skills()
for skill in available_skills:
    print("- ", skill)
print()

haystack_skills_agent = Agent(
    chat_generator=cohere_llm_generator(),
    tools=[skills_toolset],
)

result = haystack_skills_agent.run(
    messages=[ChatMessage.from_user("How to create a simple indexing pipeline?")]
)

print("-------")
print(result["last_message"].text)
