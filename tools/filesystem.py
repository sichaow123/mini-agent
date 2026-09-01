from agent.state import AgentState
from tools.base import register_tool

FILESYSTEM_TOOLS = [
    {
        "type": "function",
        "name": "run_read",
        "description": "Read file contents.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Number of lines to skip before reading",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 2000,
                    "description": "Maximum number of lines to return; omit offset and limit to read the whole file",
                },
            },
            "required": ["path"],
        },
    },
    {
        "type": "function",
        "name": "run_write",
        "description": "Write content to a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "The content to be written to the file",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "type": "function",
        "name": "run_edit",
        "description": "Replace exact text in a file once.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit",
                },
                "old_text": {
                    "type": "string",
                    "description": "The old text to be replaced",
                },
                "new_text": {
                    "type": "string",
                    "description": "The new text to replace with",
                },
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "type": "function",
        "name": "run_glob",
        "description": "Find files matching a glob pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The glob pattern used to filter files",
                }
            },
            "required": ["pattern"],
        },
    },
]


@register_tool(access_state=True)
def run_read(
    state: AgentState,
    path: str,
    offset: int | None = None,
    limit: int | None = None,
) -> str:
    request = {
        "operation": "read",
        "path": state["path_mapper"].to_container(path),
    }
    if offset is not None:
        request["offset"] = offset
    if limit is not None:
        request["limit"] = limit
    result = state["sandbox"].exec_json(request)

    if not result.get("ok", False):
        return f"Error: {result.get('error', '(Unknown reason)')}"

    content = result["content"]
    if result.get("truncated"):
        next_offset = (offset or 0) + (limit or result.get("limit", 0))
        content += (
            f"\n... output truncated; continue with offset={next_offset}, "
            f"limit={result.get('limit')}"
        )
    return content


@register_tool(access_state=True)
def run_write(state: AgentState, path: str, content: str) -> str:
    container_path = state["path_mapper"].to_container(path)
    result = state["sandbox"].exec_json(
        {
            "operation": "write",
            "path": container_path,
            "content": content,
        }
    )

    if not result.get("ok", False):
        return f"Error: {result.get('error', '(Unknown reason)')}"

    display_path = state["path_mapper"].to_display_path(result["path"])
    return f"Wrote {result['bytes']} bytes to {display_path}"


@register_tool(access_state=True)
def run_edit(state: AgentState, path: str, old_text: str, new_text: str) -> str:
    container_path = state["path_mapper"].to_container(path)
    result = state["sandbox"].exec_json(
        {
            "operation": "edit",
            "path": container_path,
            "old_text": old_text,
            "new_text": new_text,
        }
    )

    if not result.get("ok", False):
        return f"Error: {result.get('error', '(Unknown reason)')}"

    display_path = state["path_mapper"].to_display_path(result["path"])
    return f"Done editing {display_path}"


@register_tool(access_state=True)
def run_glob(state: AgentState, pattern: str) -> str:
    container_pattern = state["path_mapper"].to_container(pattern)
    result = state["sandbox"].exec_json(
        {"operation": "glob", "pattern": container_pattern}
    )

    if not result.get("ok", False):
        return f"Error: {result.get('error', '(Unknown reason)')}"

    matches = [
        state["path_mapper"].to_display_path(match)
        for match in result["matches"]
    ]
    return "\n".join(matches) if matches else "(no matches)"
