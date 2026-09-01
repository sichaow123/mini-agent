import json
import os
import time
import uuid
from collections import defaultdict

from openai import OpenAI

from agent.state import init_agent_state
from context.compression import ContextCompactor
from context.memory import (
    build_system,
    consolidate_memories,
    extract_memories,
    load_memories,
)
from runtime.permissions import check_permission
from runtime.sandbox import DockerSandbox
from tools import (
    COMPACT_TOOLS,
    FILESYSTEM_TOOLS,
    PLANNING_TOOLS,
    SHELL_TOOLS,
    SKILL_TOOLS,
    SUBAGENT_TOOLS,
    call_function,
)
from tracing.logger import log_function_call_result, log_model_output
from tracing.tracer import (
    TaskMetrics,
    flush,
    include_prompts,
    include_tool_output,
    observation,
    safe_json,
    trace_attributes,
    traced_llm_call,
    value_hash,
)


class Agent:
    MAIN_TOOLS = [
        *COMPACT_TOOLS,
        *FILESYSTEM_TOOLS,
        *SHELL_TOOLS,
        *PLANNING_TOOLS,
        *SKILL_TOOLS,
        *SUBAGENT_TOOLS,
    ]
    MAX_CONTEXT_LENGTH = 200000
    MAX_REACTIVE_RETRIES = 2
    # Prevent a model from spending an unbounded number of turns retrying a
    # broken approach. This is a per-user-request limit.
    MAX_TURNS = 100
    MAX_IDENTICAL_FAILURES = 2

    def __init__(self, workdir: str):
        session_id = str(uuid.uuid4())
        metrics = TaskMetrics()

        sandbox = DockerSandbox(session_id, workdir)
        try:
            sandbox.set_up()

            client = OpenAI(
                base_url=os.environ.get("BASE_URL"),
                api_key=os.environ.get("API_KEY"),
            )
            model = os.environ.get("MODEL_ID")
            agent_state = init_agent_state(workdir, sandbox, metrics=metrics)
            compactor = ContextCompactor(
                llm_client=client,
                model=model,
                transcript_dir=agent_state["workdir"] / ".transcripts",
                tool_results_dir=agent_state["workdir"]
                / ".task_outputs"
                / "tool-results",
                metrics=metrics,
            )
        except Exception:
            try:
                sandbox.shut_down()
            except Exception as cleanup_error:
                # Preserve the original initialization exception while still
                # making cleanup failure visible to the operator.
                print(
                    "Failed to clean up sandbox after Agent init failure: "
                    f"{cleanup_error}"
                )
            raise

        self._session_id = session_id
        self._metrics = metrics
        self._sandbox = sandbox
        self._client = client
        self._model = model
        self._agent_state = agent_state
        self._compactor = compactor
        self.input_list = []
        self._last_response_text = ""

    def clean_up(self):
        sandbox_error = None
        try:
            self._sandbox.shut_down()
        except Exception as error:
            sandbox_error = error
        flush()
        if sandbox_error is not None:
            raise sandbox_error

    @property
    def metrics(self) -> dict:
        return self._metrics.as_dict()

    @staticmethod
    def _tool_call_key(item) -> str:
        """Return a stable key for detecting repeated tool calls."""
        try:
            arguments = json.loads(item.arguments)
        except (TypeError, json.JSONDecodeError):
            arguments = item.arguments
        normalized = json.dumps(
            arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"{item.name}:{normalized}"

    @staticmethod
    def _tool_result_failed(result: str) -> bool:
        """Recognize failures returned by tools or the permission gate."""
        return result.lstrip().startswith(
            (
                "Error:",
                "Blocked:",
                "Calling function ",
                "Network access was rejected",
                "Action rejected",
            )
        )

    @staticmethod
    def _response_text(output_items) -> str:
        parts = []
        for item in output_items:
            if item.type != "message":
                continue
            for content in item.content:
                if content.type == "output_text":
                    parts.append(content.text)
        return "\n".join(parts)

    def react(self, user_query: str):
        with trace_attributes(
            session_id=self._session_id,
            metadata={
                "runtime": "docker",
                "workdir_hash": value_hash(self._agent_state["workdir"]),
            },
            tags=["mini-agent"],
        ):
            with observation(
                as_type="agent",
                name="agent.react",
                input=(
                    {"user_query": safe_json(user_query)}
                    if include_prompts()
                    else {"prompt_logging": "disabled"}
                ),
            ) as agent_trace:
                try:
                    result = self._react_impl(user_query)
                except BaseException:
                    self._metrics.termination_reason = "error"
                    raise

                agent_trace.update(
                    output={"result": result},
                )

                return result

    def _react_impl(self, user_query: str):
        self.input_list.append({"role": "user", "content": user_query})
        rounds_since_todo = None
        reactive_retries = 0
        turn = 0
        failed_call_counts: dict[str, int] = defaultdict(int)
        final_response_only = False

        relevant_memories = load_memories(
            self._agent_state,
            self._client,
            self._model,
            self.input_list,
            metrics=self._metrics,
        )

        while turn < Agent.MAX_TURNS:
            turn += 1
            self._metrics.llm_turns += 1
            if (
                self._compactor.estimate_chars(self.input_list)
                > Agent.MAX_CONTEXT_LENGTH
            ):
                print(
                    f"\033[93m[TOOL] # chars before: {self._compactor.estimate_chars(self.input_list)}\033[0m"
                )
                self.input_list[:] = self._compactor.prepare(
                    self.input_list, user_query
                )
                self._metrics.context_compactions += 1
                self._metrics.max_context_chars = max(
                    self._metrics.max_context_chars,
                    self._compactor.estimate_chars(self.input_list),
                )
                print(
                    f"\033[93m[TOOL] # chars after: {self._compactor.estimate_chars(self.input_list)}\033[0m"
                )
            try:
                system_prompt = build_system(
                    self._agent_state,
                    relevant_memories,
                    active_request=user_query,
                )
                # Once a repeated failure is detected, give the model one
                # final turn to summarize the blocker without allowing it
                # to issue another tool call.
                tools = [] if final_response_only else Agent.MAIN_TOOLS
                response = traced_llm_call(
                    self._client,
                    model=os.environ.get("MODEL_ID"),
                    input=self.input_list,
                    instructions=system_prompt,
                    tools=tools,
                    name="llm.responses.create",
                    metadata={
                        "turn": turn,
                        "reactive_retry": reactive_retries,
                    },
                    metrics=self._metrics,
                )
                reactive_retries = 0
            except Exception as error:
                with observation(
                    as_type="event",
                    name="llm.error",
                    input={
                        "turn": turn,
                        "error": safe_json(str(error)),
                    },
                ):
                    pass
                too_long = any(
                    text in str(error).lower()
                    for text in ("prompt_too_long", "too many tokens")
                )
                if too_long and reactive_retries < Agent.MAX_REACTIVE_RETRIES:
                    print("[reactive compact]")
                    self._metrics.reactive_compactions += 1
                    self._metrics.retries += 1
                    self.input_list[:] = self._compactor.reactive_compact(
                        self.input_list, user_query
                    )
                    reactive_retries += 1
                    continue
                raise

            self.input_list.extend(response.output)
            response_text = self._response_text(response.output)
            if response_text:
                self._last_response_text = response_text

            call_tool = False
            use_todo = False
            do_compression = False
            stop_after_round = False
            for item in response.output:
                log_model_output(item, path_mapper=self._agent_state["path_mapper"])
                if item.type == "function_call":
                    call_tool = True
                    self._metrics.tool_calls += 1

                    if final_response_only:
                        with observation(
                            as_type="event",
                            name="runtime.tool_blocked_after_failure",
                            input={"tool_name": item.name},
                        ):
                            pass
                        self.input_list.append(
                            {
                                "type": "function_call_output",
                                "call_id": item.call_id,
                                "output": (
                                    "Tool calls are disabled after repeated "
                                    "failures. Provide a final answer instead."
                                ),
                            }
                        )
                        continue

                    if item.name == "run_todo_write":
                        use_todo = True
                        rounds_since_todo = 0
                    elif item.name == "compact":
                        do_compression = True

                    call_key = self._tool_call_key(item)
                    if failed_call_counts[call_key] >= Agent.MAX_IDENTICAL_FAILURES:
                        func_call_result = (
                            "Error: this exact tool call has already failed "
                            f"{Agent.MAX_IDENTICAL_FAILURES} times. Do not "
                            "repeat it; use a different approach or report "
                            "the blocker to the user."
                        )
                        stop_after_round = True
                        with observation(
                            as_type="event",
                            name="runtime.duplicate_tool_call_blocked",
                            input={"tool_name": item.name},
                        ):
                            pass
                    else:
                        fc_approved, fc_reason = check_permission(
                            self._agent_state, item
                        )
                        if not fc_approved:
                            func_call_result = (
                                f"Calling function {item.name} with arguments "
                                f"{item.arguments} was rejected. Reason: {fc_reason}"
                            )
                            with observation(
                                as_type="event",
                                name="permission.denied",
                                input={
                                    "tool_name": item.name,
                                    "reason": safe_json(fc_reason),
                                },
                            ):
                                pass
                        else:
                            arguments = json.loads(item.arguments)
                            started_at = time.perf_counter()
                            with observation(
                                as_type="tool",
                                name=item.name,
                                input={
                                    "arguments": safe_json(arguments),
                                    "call_id": item.call_id,
                                },
                            ) as tool_trace:
                                try:
                                    func_call_result = call_function(
                                        self._agent_state,
                                        item.name,
                                        **arguments,
                                    )
                                    tool_trace.update(
                                        output=(
                                            safe_json(func_call_result)
                                            if include_tool_output()
                                            else {"output_logging": "disabled"}
                                        ),
                                        metadata={
                                            "ok": not str(func_call_result)
                                            .lstrip()
                                            .startswith("Error:"),
                                            "duration_ms": round(
                                                (time.perf_counter() - started_at)
                                                * 1000
                                            ),
                                        },
                                    )
                                except Exception as error:
                                    tool_trace.update(
                                        output={
                                            "error": str(error),
                                            "duration_ms": round(
                                                (time.perf_counter() - started_at)
                                                * 1000
                                            ),
                                        },
                                    )
                                    raise

                        if self._tool_result_failed(func_call_result):
                            failed_call_counts[call_key] += 1
                            self._metrics.failed_tool_calls += 1
                        else:
                            # A successful retry resets the failure streak for
                            # this call, so legitimate repeated reads remain
                            # possible.
                            failed_call_counts.pop(call_key, None)

                    log_function_call_result(
                        func_call_result,
                        path_mapper=(
                            None
                            if item.name == "run_bash"
                            else self._agent_state["path_mapper"]
                        ),
                    )
                    self.input_list.append(
                        {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": func_call_result,
                        }
                    )

            if stop_after_round:
                self.input_list.append(
                    {
                        "role": "user",
                        "content": (
                            "<runtime_guard>"
                            "The same tool call has failed repeatedly. "
                            "Stop troubleshooting, do not call tools again, "
                            "and provide a concise final answer describing "
                            "the blocker and any viable next step."
                            "</runtime_guard>"
                        ),
                    }
                )
                final_response_only = True

            if not call_tool:
                self._metrics.natural_termination = True
                self._metrics.termination_reason = "completed"
                if extract_memories(
                    self._agent_state,
                    self._client,
                    self._model,
                    self.input_list,
                    metrics=self._metrics,
                ):
                    consolidate_memories(
                        self._agent_state,
                        self._client,
                        self._model,
                        metrics=self._metrics,
                    )
                return self._last_response_text or None

            if do_compression:
                self.input_list[:] = self._compactor.compact_history(
                    self.input_list, user_query
                )
                self._metrics.context_compactions += 1

            if call_tool:
                self._metrics.tool_rounds += 1

            # rounds_since_todo = 0 if use_todo else rounds_since_todo + 1
            if rounds_since_todo is not None:
                rounds_since_todo += 1
                if rounds_since_todo >= 3:
                    self.input_list.append(
                        {
                            "role": "user",
                            "content": "<reminder>Update your todos.</reminder>",
                        }
                    )
                    rounds_since_todo = 0

        message = (
            f"Agent stopped after reaching the maximum of {Agent.MAX_TURNS} "
            "turns without producing a final answer."
        )
        self._metrics.termination_reason = "max_turns"
        print(message)
        return message
