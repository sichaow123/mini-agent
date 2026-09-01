import os
import json
import threading
import time

import docker
from runtime.executor import CONTAINER_WORKDIR, DockerExecutor
from runtime.path_mapper import PathMapper

DOCKER_CLIENT_TIMEOUT = int(os.environ.get("DOCKER_SANDBOX_TIMEOUT", "1800"))
DOCKER_CLIENT_POOL_SIZE = int(os.environ.get("DOCKER_SANDBOX_POOL_SIZE", "128"))
DOCKER_BASE_IMAGE = os.environ.get(
    "DOCKER_SANDBOX_BASE_IMAGE", "mini-agent-sandbox:base"
)
DOCKER_SANDBOX_AGENT_UID = int(
    os.environ.get("DOCKER_SANDBOX_AGENT_UID", "1000")
)
DOCKER_SANDBOX_AGENT_GID = int(
    os.environ.get("DOCKER_SANDBOX_AGENT_GID", "1000")
)
NETWORK_NAME_PREFIX = "mini-agent-net-"


def _docker_client() -> docker.DockerClient:
    return docker.from_env(
        timeout=DOCKER_CLIENT_TIMEOUT,
        max_pool_size=DOCKER_CLIENT_POOL_SIZE,
    )


class DockerSandbox:
    def __init__(self, instance_id: str, workdir: str):
        self._client = _docker_client()
        self._container = None
        self._path_mapper = PathMapper(workdir, CONTAINER_WORKDIR)
        self._executor = DockerExecutor(self._client, path_mapper=self._path_mapper)
        self._network = None
        self._network_enabled = False
        self._approved_mounts = {}
        self._instance_id = instance_id
        self._workdir = workdir
        # A session has one mutable container and one shell-state snapshot.
        # Serialize operations that touch either resource so concurrent tool
        # calls cannot race with recovery, cleanup, or one another.
        self._lock = threading.RLock()

    @property
    def network_enabled(self) -> bool:
        with self._lock:
            return self._network_enabled

    def set_up(self):
        with self._lock:
            self._ensure_container_running()

    def _set_container(self, container):
        self._container = container
        self._executor.set_container(container)

    def _recreate_container(self):
        """Replace a missing or unrecoverable session container."""
        old_container = self._container
        if old_container is not None:
            try:
                old_container.remove(force=True)
            except docker.errors.NotFound:
                self._set_container(None)
            # Keep self._container when removal fails. The caller can then
            # retry cleanup instead of losing the only reference to a live
            # or partially removed container.
            else:
                self._set_container(None)

        self.create_container()
        self._container.start()

    def _ensure_container_running(self):
        """Ensure that the session container is running before an exec call."""
        if self._container is None:
            self.create_container()
            self._container.start()
            return

        try:
            self._container.reload()
        except docker.errors.NotFound:
            self._set_container(None)
            self.create_container()
            self._container.start()
            return

        status = self._container.status
        if status == "running":
            return

        if status == "paused":
            try:
                self._container.unpause()
            except (docker.errors.APIError, docker.errors.NotFound):
                self._recreate_container()
            else:
                self._container.reload()
                if self._container.status == "running":
                    return
        elif status == "restarting":
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                time.sleep(0.2)
                self._container.reload()
                if self._container.status == "running":
                    return
        else:
            try:
                self._container.start()
            except (docker.errors.APIError, docker.errors.NotFound):
                self._recreate_container()
            else:
                self._container.reload()
                if self._container.status == "running":
                    return

        # Restart/unpause did not recover the container.
        self._recreate_container()

    def shut_down(self):
        with self._lock:
            self._shut_down_locked()

    def _shut_down_locked(self):
        """Clean up resources; caller must hold ``self._lock``."""
        container = self._container
        network = self._network
        cleanup_errors = []
        container_removed = container is None
        network_removed = network is None

        if container is not None and network is not None:
            try:
                network.disconnect(container, force=True)
            except docker.errors.NotFound:
                pass
            except Exception as error:
                cleanup_errors.append(f"network disconnect failed: {error}")

        if container is not None:
            try:
                container.stop(timeout=30)
            except docker.errors.NotFound:
                pass
            except Exception as error:
                cleanup_errors.append(f"container stop failed: {error}")

            try:
                container.remove(force=True)
            except docker.errors.NotFound:
                container_removed = True
            except Exception as error:
                cleanup_errors.append(f"container remove failed: {error}")
            else:
                container_removed = True

        if network is not None:
            try:
                network.remove()
            except docker.errors.NotFound:
                network_removed = True
            except Exception as error:
                cleanup_errors.append(f"network remove failed: {error}")
            else:
                network_removed = True

        if container_removed:
            self._set_container(None)
        if network_removed:
            self._network = None
            self._network_enabled = False

        if cleanup_errors:
            raise RuntimeError("; ".join(cleanup_errors))

    def enable_network(self):
        """Enable outbound networking for the current session container.

        Containers are initially created with ``network_mode='none'``. Do
        not try to attach a bridge endpoint to that existing container: Docker
        versions and runtimes differ in how they handle connecting a
        ``none``-mode container. Instead, create the session network first and
        recreate the container with that network as its primary network mode.
        The workspace and approved external mounts are bind mounts, so their
        contents survive the recreation.
        """
        with self._lock:
            if self._network_enabled:
                return

            self._ensure_container_running()
            try:
                network_name = f"{NETWORK_NAME_PREFIX}{self._instance_id}"
                self._network = self._client.networks.create(
                    name=network_name,
                    driver="bridge",
                    labels={"mini-agent.session": self._instance_id},
                )
                self._network_enabled = True
                self._recreate_container()
            except Exception:
                if self._network is not None:
                    try:
                        self._network.remove()
                    except docker.errors.NotFound:
                        pass
                    finally:
                        self._network = None
                self._network_enabled = False
                raise

    def create_container(self):
        """
        Creates a container from an instance image.

        Args:
            client (docker.DockerClient): Docker client for creating the container
        """
        with self._lock:
            self._create_container_locked()

    def _create_container_locked(self):
        """Create the session container; caller must hold ``self._lock``."""
        container = None
        try:
            # Fail early with an actionable error. Continuing after a missing
            # image makes Docker report the problem much later during create().
            try:
                self._client.images.get(DOCKER_BASE_IMAGE)
            except docker.errors.ImageNotFound as error:
                raise RuntimeError(
                    f"Sandbox image {DOCKER_BASE_IMAGE!r} is not available. "
                    "Build it first, for example: "
                    "docker build -f docker/Dockerfile "
                    f"-t {DOCKER_BASE_IMAGE} ."
                ) from error

            print(f"Creating container for {self._instance_id}...")

            # Remove any existing container with this name (handles ghost containers)
            try:
                old = self._client.containers.get(self._instance_id)
                old.remove(force=True)
                print(f"Removed existing container {self._instance_id}")
            except docker.errors.NotFound:
                pass

            container = self._client.containers.create(
                image=DOCKER_BASE_IMAGE,
                name=self._instance_id,
                command=["sleep", "infinity"],
                detach=True,
                auto_remove=False,
                user="agent",
                working_dir=CONTAINER_WORKDIR,
                environment={
                    "DOCKER_SANDBOX_WORKDIR": CONTAINER_WORKDIR,
                    "DOCKER_SANDBOX_ALLOWED_ROOTS": json.dumps(
                        [CONTAINER_WORKDIR, *self._approved_mounts]
                    ),
                },
                volumes={
                    str(self._workdir): {
                        "bind": CONTAINER_WORKDIR,
                        "mode": "rw",
                    },
                    **{
                        mount["host_path"]: {
                            "bind": container_root,
                            "mode": mount["mode"],
                        }
                        for container_root, mount in self._approved_mounts.items()
                    }
                },
                # Start offline. After user approval, enable_network() creates
                # a session bridge and recreates the container with that
                # network as its primary network mode.
                network_mode=(
                    self._network.name
                    if self._network_enabled and self._network is not None
                    else "none"
                ),
                # Make the image filesystem immutable. The workspace and
                # explicitly declared tmpfs mounts remain writable.
                read_only=True,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                mem_limit="1g",
                pids_limit=256,
                tmpfs={
                    # Python package builds and native extensions may need to
                    # execute/load files from the temporary build directory.
                    # Keep the mount isolated, but do not use noexec here.
                    "/tmp": "rw,exec,nosuid,nodev,size=512m",
                    # Keep session-local user configuration/cache writable
                    # without making the image root filesystem writable.
                    "/home/agent": (
                        # pip --user installs native .so extensions here; an
                        # implicit noexec tmpfs prevents Python from loading
                        # them (e.g. numpy and pyerfa).
                        "rw,exec,nosuid,nodev,size=1g,"
                        f"uid={DOCKER_SANDBOX_AGENT_UID},"
                        f"gid={DOCKER_SANDBOX_AGENT_GID},mode=0700"
                    ),
                },
                labels={
                    "mini-agent.session": self._instance_id,
                },
            )

            print(f"Container for {self._instance_id} created: {container.id}")
            self._set_container(container)

        except Exception as e:
            print(f"Error creating container for {self._instance_id}: {e}")
            self.shut_down()
            raise

    def approve_external_mount(self, host_path: str, mode: str = "ro") -> str:
        """Authorize and mount the directory containing an external path."""
        if mode not in {"ro", "rw"}:
            raise ValueError("external mount mode must be 'ro' or 'rw'")

        requested = os.path.abspath(os.path.expanduser(host_path))
        resolved = os.path.realpath(requested)
        if resolved == os.path.sep:
            raise ValueError("refusing to mount the host filesystem root")

        target = resolved if os.path.isdir(resolved) else os.path.dirname(resolved)
        if not os.path.isdir(target):
            raise ValueError(f"external mount parent is not a directory: {target}")

        with self._lock:
            for container_root, mount in self._approved_mounts.items():
                if os.path.realpath(mount["host_path"]) == target:
                    if mount["mode"] == "ro" and mode == "rw":
                        mount["mode"] = "rw"
                        self._recreate_container()
                    return container_root

            container_root = f"/external/{len(self._approved_mounts)}"
            self._approved_mounts[container_root] = {
                "host_path": target,
                "mode": mode,
            }
            self._path_mapper.add_external_mount(target, container_root)
            self._recreate_container()
            return container_root

    def external_mount_mode(self, path: str) -> str | None:
        """Return the current mode for an already authorized external path."""
        try:
            container_path = self._path_mapper.to_container(path)
        except ValueError:
            return None

        container_path = os.path.normpath(container_path)
        with self._lock:
            for container_root, mount in self._approved_mounts.items():
                root = os.path.normpath(container_root)
                if container_path == root or container_path.startswith(root + os.sep):
                    return mount["mode"]
        return None

    def run_bash(self, command: str, timeout=None):
        with self._lock:
            self._ensure_container_running()
            return self._executor.run_bash(command, timeout=timeout)

    def exec_json(self, request: dict, timeout: int | None = None) -> dict:
        with self._lock:
            try:
                self._ensure_container_running()
            except Exception as error:
                return {"ok": False, "error": f"Sandbox recovery failed: {error}"}
            return self._executor.exec_json(request, timeout=timeout)
