import json

from openai.types.responses import ResponseOutputItem


def _display_arguments(name: str, raw_arguments: str, path_mapper=None) -> str:
    if path_mapper is None:
        return raw_arguments
    try:
        arguments = json.loads(raw_arguments)
    except (TypeError, json.JSONDecodeError):
        return path_mapper.display_text(str(raw_arguments))

    if isinstance(arguments, dict):
        for key in ("path", "cwd"):
            value = arguments.get(key)
            if isinstance(value, str):
                try:
                    arguments[key] = path_mapper.to_display_path(value)
                except ValueError:
                    pass
        if name == "run_bash" and isinstance(arguments.get("command"), str):
            arguments["command"] = path_mapper.display_text(arguments["command"])
    return json.dumps(arguments, ensure_ascii=False)


def log_model_output(
    output: ResponseOutputItem, is_subagent: bool = False, path_mapper=None
):
    if output.type == "message":
        for message in output.content:
            if message.type == "output_text":
                if is_subagent:
                    print(
                        f"    \033[94m[SUB-MESSAGE]\033[0m \033[90m{message.text}\033[0m"
                    )
                else:
                    print(f"\033[94m[MESSAGE]\033[0m {message.text}")
            else:
                if is_subagent:
                    print(
                        f"    \033[94m[SUB-MESSAGE]\033[0m \033[90m{message.refusal}\033[0m"
                    )
                else:
                    print(f"\033[94m[MESSAGE]\033[0m {message.refusal}")
    elif output.type == "reasoning":
        if output.summary:
            for summary in output.summary:
                if is_subagent:
                    print(
                        f"    \033[92m[SUB-REASONING]\033[0m \033[90m{summary.text}\033[0m"
                    )
                else:
                    print(f"\033[92m[REASONING]\033[0m {summary.text}")
        if output.content:
            for content in output.content:
                if is_subagent:
                    print(
                        f"    \033[92m[SUB-REASONING]\033[0m \033[90m{content.text}\033[0m"
                    )
                else:
                    print(f"\033[92m[REASONING]\033[0m {content.text}")
    elif output.type == "function_call":
        arguments = _display_arguments(output.name, output.arguments, path_mapper)
        if is_subagent:
            print(
                f"    \033[93m[SUB-TOOL]\033[0m \033[90mCall [{output.name}] with [{arguments}]\033[0m"
            )
        else:
            print(
                f"\033[93m[TOOL]\033[0m Call \033[93m{output.name}\033[0m with \033[93m{arguments}\033[0m"
            )


def log_function_call_result(
    result: str, is_subagent: bool = False, limit: int = 1000, path_mapper=None
):
    if path_mapper is not None:
        result = path_mapper.display_text(result)
    if len(result) > limit:
        final_result = result[:limit] + f"... ({len(result) - limit} more characters)"
    else:
        final_result = result
    if is_subagent:
        print(f"    \033[93m[SUB-TOOL]\033[0m \033[90mResult:\n{final_result}\033[0m")
    else:
        print(f"\033[93m[TOOL]\033[0m Result:\n{final_result}")
