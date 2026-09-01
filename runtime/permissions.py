import json
import re
import shlex
from pathlib import Path

from openai.types.responses import ResponseFunctionToolCall

from agent.state import AgentState
from runtime.executor import CONTAINER_WORKDIR

# Gate 1: hard deny list - always forbidden. These checks are intentionally
# based on parsed command words instead of raw substrings; a substring check
# is easy to bypass with flags, quoting, or a path prefix.
DENY_COMMANDS = {
    "sudo",
    "doas",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "mkfs",
    "fdisk",
    "parted",
    "chroot",
    "insmod",
    "modprobe",
}
SHELL_OPERATORS = {";", "&&", "||", "|", "&", "(", ")"}
COMMAND_WRAPPERS = {"env", "command", "exec", "nohup", "nice", "ionice", "timeout"}
ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
REDIRECTION_RE = re.compile(r"^(?:\d+)?(?:>>|>|<>|<&|>&)$")


def _tokens(command: str) -> list[str] | None:
    """Return shell-like tokens, or None when the command is not parseable."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)
    except ValueError:
        return None


def _command_segments(tokens: list[str]) -> list[list[str]]:
    """Split tokens into command segments while retaining command arguments."""
    segments: list[list[str]] = []
    current: list[str] = []
    skip_redirection_target = False

    for token in tokens:
        if skip_redirection_target:
            skip_redirection_target = False
            continue
        if token in SHELL_OPERATORS:
            if current:
                segments.append(current)
                current = []
            continue
        if REDIRECTION_RE.match(token) or token in {">", ">>", "<", "<<"}:
            skip_redirection_target = True
            continue
        if ASSIGNMENT_RE.match(token) and not current:
            continue
        current.append(token)

    if current:
        segments.append(current)
    return segments


def _basename(word: str) -> str:
    return word.rsplit("/", 1)[-1]


def _effective_command(segment: list[str]) -> tuple[str, list[str]]:
    """Unwrap common command launchers such as ``env`` and ``timeout``."""
    index = 0
    while index < len(segment):
        word = segment[index]
        name = _basename(word).lower()
        if name not in COMMAND_WRAPPERS:
            break
        index += 1
        # Skip wrapper flags, assignments, and timeout's numeric duration.
        while index < len(segment) and (
            segment[index].startswith("-")
            or ASSIGNMENT_RE.match(segment[index])
            or segment[index].isdigit()
        ):
            index += 1
    if index >= len(segment):
        return "", []
    return _basename(segment[index]).lower(), segment[index + 1 :]


def _has_flag(args: list[str], *flags: str) -> bool:
    for arg in args:
        if arg == "--":
            break
        if arg in flags:
            return True
        if arg.startswith("-") and not arg.startswith("--"):
            if any(flag.lstrip("-") in arg[1:] for flag in flags):
                return True
    return False


def _redirection_targets(tokens: list[str]) -> list[str]:
    targets = []
    for index, token in enumerate(tokens[:-1]):
        if REDIRECTION_RE.match(token) or token in {">", ">>", "<", "<<"}:
            targets.append(tokens[index + 1])
    return targets


def _is_workspace_target(workdir: Path, target: str) -> bool:
    if target in {"/dev/null", "/dev/stdout", "/dev/stderr"}:
        return True
    try:
        target_path = Path(target)
        if target_path.is_absolute() and target_path.is_relative_to(CONTAINER_WORKDIR):
            return True
        resolved = (workdir / target_path).resolve()
        return resolved.is_relative_to(workdir.resolve())
    except (OSError, ValueError):
        return False


def _destructive_reasons(command: str, workdir: Path | None = None) -> list[str]:
    tokens = _tokens(command)
    if tokens is None:
        return ["command contains invalid or incomplete shell syntax"]

    reasons: list[str] = []
    segments = _command_segments(tokens)

    # Shell indirection makes static policy checks unreliable. Ask instead of
    # pretending that a command such as `bash -c "$CMD"` was fully inspected.
    if "$(" in command or "`" in command:
        reasons.append("command substitution is not statically inspectable")
    if re.search(r"\$\{?[A-Za-z_][A-Za-z0-9_]*", command):
        reasons.append("variable expansion may hide the executed command")

    redirection_targets = _redirection_targets(tokens)
    unsafe_redirection = (
        redirection_targets
        if workdir is None
        else [
            target
            for target in redirection_targets
            if not _is_workspace_target(workdir, target)
        ]
    )
    if unsafe_redirection:
        reasons.append("shell output/input redirection can overwrite files")

    if any(
        token == "/dev/sda"
        or token.startswith("/dev/sd")
        or token.startswith("/dev/nvme")
        for token in tokens
    ):
        reasons.append("raw disk device access is forbidden")

    for segment in segments:
        if not segment:
            continue
        name, args = _effective_command(segment)
        if not name:
            continue

        if name in DENY_COMMANDS:
            reasons.append(f"privileged/system command '{name}' is forbidden")

        if name in {"bash", "sh", "zsh", "fish", "eval", "xargs"}:
            if "-c" in args or name in {"eval", "xargs"}:
                reasons.append(f"indirect command execution through '{name}'")

        if name == "rm":
            recursive = _has_flag(args, "-r", "-R", "--recursive")
            force = _has_flag(args, "-f", "--force")
            reasons.append(
                "file deletion command requires confirmation"
                + (" (recursive/force)" if recursive or force else "")
            )
            paths = [arg for arg in args if not arg.startswith("-") and arg != "--"]
            if any(
                path in {"/", "/*", "/etc", "/usr", "/var", "/home"} for path in paths
            ):
                return ["deletion targets the container root or a system directory"]

        if name == "find" and any(
            arg in {"-delete", "-exec", "-execdir"} for arg in args
        ):
            reasons.append("find can delete or modify many files")

        if name == "git" and args and args[0] == "clean":
            reasons.append("git clean permanently removes untracked files")

        if name in {"chmod", "chown"} and (
            _has_flag(args, "-R", "--recursive")
            or any(arg.lstrip("+-") in {"777", "a+rwx", "a=rwx"} for arg in args)
        ):
            reasons.append(f"recursive or overly broad '{name}' permission change")

        if name == "dd" and (
            any(arg.startswith("if=") or arg.startswith("of=/dev/") for arg in args)
        ):
            reasons.append("raw disk device access is forbidden")
        elif name == "dd" or name in {"mount", "umount", "iptables", "nft"}:
            reasons.append(f"device/system modification command '{name}'")

    return list(dict.fromkeys(reasons))


NETWORK_COMMANDS = {
    "curl",
    "wget",
    "fetch",
    "aria2c",
    "scp",
    "sftp",
    "ssh",
    "ftp",
    "nc",
    "netcat",
    "telnet",
    "rsync",
    "apt",
    "apt-get",
    "apk",
    "pip",
    "pip3",
    "uv",
    "poetry",
    "npm",
    "pnpm",
    "yarn",
    "cargo",
}
NETWORK_GIT_SUBCOMMANDS = {"clone", "fetch", "pull", "push", "submodule"}


def command_requests_network(command: str) -> bool:
    """Conservatively identify commands that normally need outbound access."""
    tokens = _tokens(command)
    if tokens is None:
        return True
    if (
        "$(" in command
        or "`" in command
        or re.search(r"\$\{?[A-Za-z_][A-Za-z0-9_]*", command)
    ):
        return True

    for segment in _command_segments(tokens):
        name, args = _effective_command(segment)
        if name in NETWORK_COMMANDS:
            if name in {"npm", "pnpm", "yarn"}:
                return bool(
                    args and args[0] in {"install", "i", "add", "update", "publish"}
                )
            if name == "cargo":
                return bool(
                    args and args[0] in {"build", "install", "fetch", "add", "update"}
                )
            return True
        if name in {"python", "python3", "ruby", "node"} and len(args) >= 2:
            if args[0] in {"-m", "--module"} and args[1] in {
                "pip",
                "pip3",
                "poetry",
                "uv",
            }:
                return True
        if name == "git" and args and args[0] in NETWORK_GIT_SUBCOMMANDS:
            return True
    return False


def check_deny_list(command: str) -> str | None:
    if not isinstance(command, str) or not command.strip():
        return "Blocked: command must be a non-empty string"

    reasons = _destructive_reasons(command)
    hard_reasons = [
        reason
        for reason in reasons
        if (
            "forbidden" in reason
            or "targets the container root" in reason
            or "raw disk device" in reason
        )
    ]
    if hard_reasons:
        return f"Blocked: {hard_reasons[0]}"
    return None


# Gate 2: Rule matching - context-dependent checks
PERMISSION_RULES = [
    {
        "tools": ["run_write", "run_edit"],
        "check": lambda workdir, args: not _is_workspace_target(
            workdir, args.get("path", "")
        ),
        "message": "Writing outside workspace",
    },
]


def check_rules(workdir: Path, tool_name: str, args: dict) -> str | None:
    if tool_name == "run_bash":
        reasons = _destructive_reasons(args.get("command", ""), workdir)
        if reasons:
            return "Potentially destructive or indirect command: " + "; ".join(reasons)

    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](workdir, args):
            return rule["message"]
    return None


# Gate 3: User approval - wait for confirmation after rule match
def ask_user(tool_name: str, args: dict, reason: str) -> str:
    print(f"\n\033[33m[permission] {reason}\033[0m")
    print(f"   Tool: {tool_name}({args})")
    choice = input("   Allow? [y/N] ").strip().lower()
    return "allow" if choice in ("y", "yes") else "deny"


# Pipeline: all three gates chained
def check_permission(
    state: AgentState, block: ResponseFunctionToolCall
) -> tuple[bool, str]:
    try:
        arguments = json.loads(block.arguments)
        rule_arguments = dict(arguments) if isinstance(arguments, dict) else arguments
        external_path_tool = {
            "run_read": "ro",
            "run_write": "rw",
            "run_edit": "rw",
        }.get(block.name)
        external_access_granted = False
        if external_path_tool and isinstance(arguments.get("path"), str):
            external_path = arguments["path"]
            mapper = state["path_mapper"]
            try:
                mapper.to_container(external_path)
            except ValueError:
                if not Path(external_path).is_absolute():
                    raise
                decision = ask_user(
                    block.name,
                    arguments,
                    f"Accessing external host path ({external_path}) requires a Docker mount ({external_path_tool})",
                )
                if decision == "deny":
                    return False, "Action rejected by the user"
                try:
                    state["sandbox"].approve_external_mount(
                        external_path, mode=external_path_tool
                    )
                    rule_arguments["path"] = mapper.to_container(external_path)
                    external_access_granted = True
                except Exception as error:
                    return False, f"Failed to mount external path: {error}"
            else:
                current_mode = state["sandbox"].external_mount_mode(external_path)
                if current_mode == "ro" and external_path_tool == "rw":
                    decision = ask_user(
                        block.name,
                        arguments,
                        f"Upgrade the existing external mount for {external_path} from read-only to read-write",
                    )
                    if decision == "deny":
                        return False, "Action rejected by the user"
                    try:
                        state["sandbox"].approve_external_mount(
                            external_path, mode="rw"
                        )
                    except Exception as error:
                        return False, f"Failed to upgrade external mount: {error}"
                    external_access_granted = True
                elif current_mode in {"ro", "rw"}:
                    external_access_granted = True
        if block.name == "run_bash":
            reason = check_deny_list(arguments.get("command", ""))
            if reason:
                print(f"\n\033[31m[blocked] {reason}\033[0m")
                return False, reason
        reason = (
            None
            if external_access_granted
            else check_rules(state["workdir"], block.name, rule_arguments)
        )
        if reason:
            decision = ask_user(block.name, arguments, reason)
            if decision == "deny":
                return False, "Action rejected by the user"

        # Shell commands are the only current tool that can directly access
        # the network. Keep the session sandbox offline until the user
        # explicitly grants network access; one approval enables networking
        # for the remainder of this Agent session.
        if (
            block.name == "run_bash"
            and not state["sandbox"].network_enabled
            and command_requests_network(arguments.get("command", ""))
        ):
            decision = ask_user(
                block.name,
                arguments,
                "Network access is disabled for this session. Allow it for the remainder of the session?",
            )
            if decision == "deny":
                return False, "Network access was rejected by the user"
            try:
                state["sandbox"].enable_network()
            except Exception as error:
                return False, f"Failed to enable network access: {error}"

        return True, "Action approved"
    except Exception as e:
        reason = f"Error encountered during permission checking: {e}"
        print(reason)
        return False, reason
