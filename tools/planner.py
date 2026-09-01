from agent.state import AgentState
from tools.base import register_tool

PLANNING_TOOLS = [
    {
        "type": "function",
        "name": "run_planning",
        "description": "Create and manage a task list for the current coding session.",
        "parameters": {
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "List of todos to be solved step by step to complete the task",
                    "maxItems": 20,
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "A planned todo item",
                                "minLength": 1,
                            },
                            "status": {
                                "type": "string",
                                "description": "Current status of the planned todo item",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["content", "status"],
                    },
                }
            },
            "required": ["todos"],
        },
    }
]


@register_tool(access_state=True)
def run_planning(state: AgentState, todos: list | str) -> str:
    try:
        output = state["planner"].update(todos)
    except ValueError as e:
        return f"Error: {e}"
    print(f"\n\033[33m## Current Tasks\033[0m\n{output}")
    return output
