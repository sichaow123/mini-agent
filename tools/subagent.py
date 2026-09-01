from agent.state import AgentState
from agent.subagent import SubAgent
from tools.base import register_tool

SUBAGENT_TOOLS = [
    {
        "type": "function",
        "name": "run_subagent",
        "description": "Run a subagent with fresh conversation context and return its final text.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Prompt regarding the task assigned to this subagent",
                }
            },
            "required": ["prompt"],
        },
    }
]


@register_tool(access_state=True)
def run_subagent(state: AgentState, prompt: str) -> str:
    print("\n\033[35m[Subagent started]\033[0m")
    try:
        sub_agent = SubAgent(
            workdir=state["workdir"],
            sandbox=state["sandbox"],
            metrics=state.get("metrics"),
        )
        answer = sub_agent.solve_task(prompt)
    except Exception as e:
        answer = f"Error encountered when calling subagent: {e}"
    print("\033[35m[Subagent stopped]\033[0m\n")
    return answer
