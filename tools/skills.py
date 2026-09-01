from agent.state import AgentState
from tools.base import register_tool

SKILL_TOOLS = [
    {
        "type": "function",
        "name": "load_skill",
        "description": "Load the full SKILL.md content by skill name.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name",
                }
            },
            "required": ["name"],
        },
    }
]


@register_tool(access_state=True)
def load_skill(state: AgentState, name: str) -> str:
    try:
        output = state["skill_manager"].load(name)
    except ValueError as e:
        return f"Error encountered when loading the skill: {e}"
    return output
