"""Translate paths between the host workspace and the session container."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath


class PathMapper:
    def __init__(self, host_root: str | Path, container_root: str = "/workspace"):
        self.host_root = Path(host_root).resolve()
        self.container_root = PurePosixPath(container_root)
        self.external_mounts: dict[str, Path] = {}

    def add_external_mount(self, host_root: str | Path, container_root: str) -> None:
        self.external_mounts[container_root] = Path(host_root).resolve()

    def _external_container_path(self, host_path: Path) -> str | None:
        for container_root, mount_root in sorted(
            self.external_mounts.items(), key=lambda item: len(str(item[1])), reverse=True
        ):
            try:
                relative = host_path.relative_to(mount_root)
            except ValueError:
                continue
            return str(PurePosixPath(container_root) / PurePosixPath(relative.as_posix()))
        return None

    def to_container(self, value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("path must be a non-empty string")

        path = Path(value)
        if path.is_absolute():
            try:
                relative = path.resolve().relative_to(self.host_root)
            except ValueError:
                external_path = self._external_container_path(path.resolve())
                if external_path is not None:
                    return external_path
                container_path = PurePosixPath(value)
                if (
                    container_path == self.container_root
                    or container_path.is_relative_to(self.container_root)
                    or self.is_external_container_path(value)
                ):
                    return str(container_path)
                raise ValueError(f"path is outside workspace: {value}")
            return str(self.container_root / PurePosixPath(relative.as_posix()))
        return value

    def to_host(self, value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("path must be a non-empty string")

        container_path = PurePosixPath(value)
        if container_path.is_absolute():
            try:
                relative = container_path.relative_to(self.container_root)
            except ValueError:
                for external_root, mount_root in self.external_mounts.items():
                    try:
                        return str(
                            mount_root
                            / Path(
                                container_path.relative_to(
                                    PurePosixPath(external_root)
                                ).as_posix()
                            )
                        )
                    except ValueError:
                        continue
                raise ValueError(f"path is outside container workspace: {value}")
        else:
            relative = container_path
        return str(self.host_root / Path(relative.as_posix()))

    def is_external_container_path(self, value: str) -> bool:
        path = PurePosixPath(value)
        return any(
            path == PurePosixPath(root) or path.is_relative_to(PurePosixPath(root))
            for root in self.external_mounts
        )

    def to_container_relative(self, value: str) -> str:
        """Map a path/pattern to a workspace-relative container value."""
        mapped = PurePosixPath(self.to_container(value))
        if mapped.is_absolute():
            try:
                relative = mapped.relative_to(self.container_root)
            except ValueError as error:
                for external_root in self.external_mounts:
                    try:
                        return str(mapped.relative_to(PurePosixPath(external_root)))
                    except ValueError:
                        continue
                raise ValueError(f"path is outside container workspace: {value}") from error
            return relative.as_posix() or "."
        return str(mapped)

    def to_container_command(self, command: str) -> str:
        """Translate explicit host workspace paths embedded in a shell command.

        This deliberately only rewrites the configured workspace prefix. It
        does not attempt to parse arbitrary shell syntax or rewrite unrelated
        strings, so commands using shell indirection remain subject to the
        normal permission checks.
        """
        if not isinstance(command, str):
            raise ValueError("command must be a string")

        host_root = re.escape(str(self.host_root).rstrip("/"))
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_.-]){host_root}"
            rf"(?=(?:/|['\" );&|<>]|$))"
        )
        return pattern.sub(str(self.container_root).rstrip("/"), command)

    def to_display_path(self, value: str) -> str:
        """Return a host-facing path while accepting either path namespace."""
        path = Path(value)
        if path.is_absolute():
            try:
                resolved = path.resolve()
                resolved.relative_to(self.host_root)
                return str(resolved)
            except ValueError:
                try:
                    return self.to_host(value)
                except ValueError:
                    return value
        return self.to_host(value)

    def display_text(self, text: str) -> str:
        """Convert only known absolute/relative persisted workspace references."""
        host_root = str(self.host_root)
        container_prefix = str(self.container_root).rstrip("/") + "/"
        text = text.replace(container_prefix, host_root + "/")
        for container_root in sorted(self.external_mounts, key=len, reverse=True):
            external_prefix = container_root.rstrip("/") + "/"
            text = text.replace(
                external_prefix,
                str(self.external_mounts[container_root]) + "/",
            )
        text = text.replace(
            "Full output: .task_outputs/",
            f"Full output: {host_root}/.task_outputs/",
        )
        return text
