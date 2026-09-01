import subprocess

from agent.state import AgentState
from tools.base import register_tool

SHELL_TOOLS = [
    {
        "type": "function",
        "name": "run_bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run",
                }
            },
            "required": ["command"],
        },
    }
]


@register_tool(access_state=True)
def run_bash(state: AgentState, command: str) -> str:
    try:
        result = state["sandbox"].run_bash(command)
        # Do not rewrite shell output. stdout/stderr may contain literal data
        # such as `echo /workspace` or file contents, and the runtime cannot
        # reliably distinguish those from a path produced by `pwd`.
        out = result.strip()
        output = out if out else "(no output)"
        return output

    except Exception as e:
        return f"Error: {e}"
