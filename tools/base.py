from agent.state import AgentState

FUNCTION_MAP = {}


def register_tool(name: str | None = None, access_state: bool = False):
    def decorator(func):
        tool_name = name if name is not None else func.__name__
        FUNCTION_MAP[tool_name] = {"access_state": access_state, "function": func}
        return func

    return decorator


def call_function(state: AgentState, func_name: str, *args, **kwargs):
    try:
        func_config = FUNCTION_MAP.get(func_name)
        if func_config is None:
            return f"Unknown function: {func_name}"
        if func_config["access_state"]:
            return func_config["function"](state, *args, **kwargs)
        return func_config["function"](*args, **kwargs)
    except Exception as e:
        return f"Error encountered during function call: {e}"
