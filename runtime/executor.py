import json
import os
import socket as py_socket

from docker.utils.socket import consume_socket_output, demux_adaptor, frames_iter


CONTAINER_WORKDIR = os.environ.get("DOCKER_SANDBOX_WORKDIR", "/workspace")
SHELL_STATE_DIR = "/home/agent/.agent-shell-state"
SHELL_STATE_ENV = f"{SHELL_STATE_DIR}/exports"
SHELL_STATE_CWD = f"{SHELL_STATE_DIR}/cwd"
DEFAULT_COMMAND_TIMEOUT = int(os.environ.get("DOCKER_SANDBOX_COMMAND_TIMEOUT", "300"))
COMMAND_TIMEOUT_GRACE = int(
    os.environ.get("DOCKER_SANDBOX_COMMAND_TIMEOUT_GRACE", "10")
)
DEFAULT_FILE_OPS_TIMEOUT = int(
    os.environ.get("DOCKER_SANDBOX_FILE_OPS_TIMEOUT", "60")
)
FILE_OPS_TIMEOUT_GRACE = int(
    os.environ.get("DOCKER_SANDBOX_FILE_OPS_TIMEOUT_GRACE", "5")
)
DOCKER_SOCKET_TIMEOUT_MARGIN = int(
    os.environ.get("DOCKER_SANDBOX_SOCKET_TIMEOUT_MARGIN", "5")
)


class DockerExecutor:
    """Execute commands and tool runners inside one session container."""

    def __init__(self, client, container=None, path_mapper=None):
        self._client = client
        self._container = container
        self._path_mapper = path_mapper

    def set_container(self, container):
        self._container = container

    def set_path_mapper(self, path_mapper):
        self._path_mapper = path_mapper

    def _require_container(self):
        if self._container is None:
            raise RuntimeError("Sandbox container is not available")
        return self._container

    @staticmethod
    def _set_connection_timeout(connection, timeout: int) -> None:
        """Set timeout on Docker SDK sockets when the transport exposes one."""
        setter = getattr(connection, "settimeout", None)
        if callable(setter):
            setter(timeout)
            return

        # docker-py commonly returns socket.SocketIO. SocketIO does not
        # expose settimeout(), but its underlying socket does.
        raw_socket = getattr(connection, "_sock", None)
        setter = getattr(raw_socket, "settimeout", None)
        if callable(setter):
            setter(timeout)

    @staticmethod
    def _send_connection(connection, payload: bytes) -> None:
        """Send bytes across standard sockets and docker-py SocketIO objects."""
        sender = getattr(connection, "sendall", None)
        if callable(sender):
            sender(payload)
            return

        # docker-py returns a read-only socket.SocketIO wrapper for the
        # hijacked HTTP response. Its write() method may fail with
        # "File or stream is not writable"; the underlying socket remains the
        # bidirectional channel used for Docker exec stdin/stdout.
        raw_socket = getattr(connection, "_sock", None)
        sender = getattr(raw_socket, "sendall", None)
        if callable(sender):
            sender(payload)
            return

        writer = getattr(connection, "write", None)
        if callable(writer):
            written = writer(payload)
            if written is not None and written != len(payload):
                raise OSError(
                    f"Docker exec socket wrote {written} of {len(payload)} bytes"
                )
            flush = getattr(connection, "flush", None)
            if callable(flush):
                flush()
            return

        raise OSError("Docker exec connection does not support writing")

    @staticmethod
    def _shutdown_connection_write(connection) -> None:
        shutdown = getattr(connection, "shutdown", None)
        if callable(shutdown):
            shutdown(py_socket.SHUT_WR)
            return

        raw_socket = getattr(connection, "_sock", None)
        shutdown = getattr(raw_socket, "shutdown", None)
        if callable(shutdown):
            shutdown(py_socket.SHUT_WR)

    def run_bash(self, command: str, timeout: int | None = None):
        container = self._require_container()

        if self._path_mapper is not None:
            command = self._path_mapper.to_container_command(command)

        if timeout is None:
            timeout = DEFAULT_COMMAND_TIMEOUT

        if timeout <= 0:
            return "Error: command timeout must be greater than zero"
        if COMMAND_TIMEOUT_GRACE < 0:
            return "Error: command timeout grace period cannot be negative"

        # Restore cwd/exported variables before the command and save them from
        # the EXIT trap afterwards. This gives each session shell-like state
        # while retaining a command-level timeout.
        shell_script = f"""
state_dir='{SHELL_STATE_DIR}'
state_env='{SHELL_STATE_ENV}'
state_cwd='{SHELL_STATE_CWD}'
mkdir -p -- "$state_dir"

if [ -f "$state_env" ]; then
    . "$state_env"
fi

if [ -s "$state_cwd" ]; then
    saved_cwd=$(cat -- "$state_cwd")
    cd -- "$saved_cwd" 2>/dev/null || cd -- '{CONTAINER_WORKDIR}'
else
    cd -- '{CONTAINER_WORKDIR}'
fi

save_shell_state() {{
    status=$?
    pwd -P > "$state_cwd" 2>/dev/null || printf '%s\\n' '{CONTAINER_WORKDIR}' > "$state_cwd"
    export -p > "$state_env"
    exit "$status"
}}
trap save_shell_state EXIT

# timeout signals the command process group. Keep a defensive trap here as
# well, so descendants started by the command are terminated with the shell.
terminate_process_group() {{
    trap - TERM INT
    kill -- -"$$" 2>/dev/null || true
    exit 143
}}
trap terminate_process_group TERM INT

{command}
"""
        timeout_command = [
            "/usr/bin/timeout",
            "--signal=TERM",
            f"--kill-after={COMMAND_TIMEOUT_GRACE}s",
            f"{timeout}s",
            "/usr/bin/setsid",
            "bash",
            "-c",
            shell_script,
        ]

        try:
            exit_code, output = container.exec_run(
                cmd=timeout_command,
                workdir=CONTAINER_WORKDIR,
                demux=False,
            )
        except Exception as error:
            return f"Error: {error}"

        result_text = output.decode("utf-8", errors="replace") if output else ""
        if exit_code == 124:
            return f"Error: Command timed out after {timeout}s\n{result_text}".rstrip()
        if exit_code == 137:
            return (
                f"Error: Command was killed after exceeding the {timeout}s timeout "
                f"and {COMMAND_TIMEOUT_GRACE}s grace period\n{result_text}"
            ).rstrip()
        if exit_code == 0:
            return result_text
        return f"Error: {result_text}"

    def exec_json(
        self, request: dict, timeout: int | None = None
    ) -> dict:
        container = self._require_container()
        if timeout is None:
            timeout = DEFAULT_FILE_OPS_TIMEOUT
        if timeout <= 0:
            return {"ok": False, "error": "file operation timeout must be greater than zero"}
        if FILE_OPS_TIMEOUT_GRACE < 0:
            return {"ok": False, "error": "file operation timeout grace period cannot be negative"}

        exec_info = self._client.api.exec_create(
            container.id,
            cmd=[
                "/usr/bin/timeout",
                "--signal=TERM",
                f"--kill-after={FILE_OPS_TIMEOUT_GRACE}s",
                f"{timeout}s",
                "python",
                "/opt/sandbox-tools/file_ops.py",
            ],
            stdin=True,
            stdout=True,
            stderr=True,
            tty=False,
            workdir=CONTAINER_WORKDIR,
        )

        exec_id = exec_info["Id"]
        connection = None
        try:
            connection = self._client.api.exec_start(
                exec_id,
                tty=False,
                socket=True,
            )
            # Docker's client timeout covers API calls, not an open exec
            # stream. Set a socket timeout so a stuck runner cannot block the
            # host agent forever. The in-container timeout remains the source
            # of truth for terminating the runner.
            self._set_connection_timeout(
                connection,
                timeout + FILE_OPS_TIMEOUT_GRACE + DOCKER_SOCKET_TIMEOUT_MARGIN
            )
            payload = json.dumps(request, ensure_ascii=False).encode("utf-8")
            self._send_connection(connection, payload)
            try:
                self._shutdown_connection_write(connection)
            except (AttributeError, OSError):
                pass

            demuxed_frames = (
                demux_adaptor(stream_id, data)
                for stream_id, data in frames_iter(connection, tty=False)
            )
            stdout, stderr = consume_socket_output(
                demuxed_frames,
                demux=True,
            )
        except Exception as error:
            if isinstance(error, py_socket.timeout):
                return {
                    "ok": False,
                    "error": f"File operation timed out after {timeout}s",
                    "exec_id": exec_id,
                }
            return {"ok": False, "error": f"Docker exec failed: {error}"}
        finally:
            if connection is not None:
                connection.close()

        try:
            exit_code = self._client.api.exec_inspect(exec_id).get("ExitCode")
        except Exception as error:
            return {
                "ok": False,
                "error": f"Could not inspect file operation: {error}",
                "exec_id": exec_id,
            }
        output = (stdout or b"").decode("utf-8", errors="replace").strip()
        error_output = (stderr or b"").decode("utf-8", errors="replace").strip()

        try:
            content = json.loads(output)
        except json.JSONDecodeError as error:
            message = f"Invalid runner response: {error}"
            if error_output:
                message += f"; stderr: {error_output}"
            elif output:
                message += f"; stdout: {output[:1000]}"
            content = {"ok": False, "error": message}

        if not isinstance(content, dict):
            content = {"ok": False, "error": "Runner response must be a JSON object"}
        if error_output:
            content["stderr"] = error_output
        content["exit_code"] = exit_code
        if exit_code in (124, 137):
            content["ok"] = False
            content["error"] = (
                f"File operation timed out after {timeout}s"
                if exit_code == 124
                else f"File operation was killed after exceeding the {timeout}s timeout "
                f"and {FILE_OPS_TIMEOUT_GRACE}s grace period"
            )
        return content
