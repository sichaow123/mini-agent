#!/usr/local/bin/python
"""JSON file-operation runner for the Agent session sandbox."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from pathlib import Path

WORKSPACE = Path(
    os.environ.get("DOCKER_SANDBOX_WORKDIR", "/workspace")
).resolve()
try:
    ALLOWED_ROOTS = [
        Path(root).resolve()
        for root in json.loads(
            os.environ.get(
                "DOCKER_SANDBOX_ALLOWED_ROOTS", json.dumps([str(WORKSPACE)])
            )
        )
    ]
except (TypeError, json.JSONDecodeError):
    ALLOWED_ROOTS = [WORKSPACE]
if WORKSPACE not in ALLOWED_ROOTS:
    ALLOWED_ROOTS.insert(0, WORKSPACE)
MAX_READ_BYTES = 10 * 1024 * 1024
MAX_READ_LINES = 2_000
MAX_WRITE_BYTES = 10 * 1024 * 1024
MAX_RESULT_BYTES = 50 * 1024 * 1024
MAX_GLOB_MATCHES = 10_000
MAX_GLOB_PATTERN_LENGTH = 1_000


def fail(message: str) -> dict:
    return {"ok": False, "error": message}


def resolve_workspace_path(user_path: str) -> Path:
    if not isinstance(user_path, str) or not user_path:
        raise ValueError("path must be a non-empty string")

    candidate = Path(user_path)
    path = candidate.resolve() if candidate.is_absolute() else (WORKSPACE / candidate).resolve()
    if not any(path == root or path.is_relative_to(root) for root in ALLOWED_ROOTS):
        raise ValueError("path is outside the authorized workspace")
    return path


def resolve_glob_pattern(pattern: str) -> str:
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("pattern must be a non-empty string")
    if len(pattern) > MAX_GLOB_PATTERN_LENGTH:
        raise ValueError("glob pattern is too long")
    if any(part == ".." for part in Path(pattern).parts):
        raise ValueError("glob pattern cannot contain '..'")
    if Path(pattern).is_absolute():
        pattern_path = Path(pattern).resolve()
        if not any(
            pattern_path == root or pattern_path.is_relative_to(root)
            for root in ALLOWED_ROOTS
        ):
            raise ValueError("glob pattern is outside the authorized workspace")
    return pattern


def atomic_write(path: Path, content: str) -> None:
    """Write beside the target, then replace it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    preserved_mode = None
    if path.exists() and not path.is_symlink():
        preserved_mode = stat.S_IMODE(path.stat().st_mode)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            ) as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            if preserved_mode is not None:
                os.chmod(temporary_file.name, preserved_mode)
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)

        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def read_file(request: dict) -> dict:
    path = resolve_workspace_path(request["path"])
    if not path.is_file():
        return fail(f"not a file: {request['path']}")

    # Omitting both parameters means "read the file". This is the common
    # case and avoids forcing the model to make several calls for ordinary
    # source files. Supplying either parameter explicitly opts into paging.
    paged = "offset" in request or "limit" in request
    offset = request.get("offset", 0)
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        return fail("offset must be a non-negative integer")

    limit = request.get("limit")
    if paged:
        if limit is None:
            limit = MAX_READ_LINES
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            return fail("limit must be a positive integer")
        if limit > MAX_READ_LINES:
            return fail(f"limit must not exceed {MAX_READ_LINES} lines")

    if not paged:
        with path.open("rb") as file:
            content_bytes = file.read(MAX_READ_BYTES + 1)
        if len(content_bytes) > MAX_READ_BYTES:
            return fail(
                f"file is larger than {MAX_READ_BYTES} bytes; "
                "use offset and limit to read it in segments"
            )
        try:
            content = content_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            return fail(f"file is not valid UTF-8: {error}")
        return {
            "ok": True,
            "operation": "read",
            "path": request["path"],
            "offset": None,
            "limit": None,
            "content": content,
            "truncated": False,
        }

    selected_lines = []
    selected_bytes = 0
    truncated = False
    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file):
                if line_number < offset:
                    continue
                if len(selected_lines) >= limit:
                    truncated = True
                    break

                line_bytes = len(line.encode("utf-8"))
                if selected_bytes + line_bytes > MAX_READ_BYTES:
                    truncated = True
                    break
                selected_lines.append(line)
                selected_bytes += line_bytes
    except UnicodeDecodeError as error:
        return fail(f"file is not valid UTF-8: {error}")

    content = "".join(selected_lines)

    return {
        "ok": True,
        "operation": "read",
        "path": request["path"],
        "offset": offset,
        "limit": limit,
        "content": content,
        "truncated": truncated,
    }


def write_file(request: dict) -> dict:
    path = resolve_workspace_path(request["path"])
    content = request.get("content")
    if not isinstance(content, str):
        return fail("content must be a string")
    encoded = content.encode("utf-8")
    if len(encoded) > MAX_WRITE_BYTES:
        return fail(f"content is larger than {MAX_WRITE_BYTES} bytes")

    atomic_write(path, content)
    return {
        "ok": True,
        "operation": "write",
        "path": request["path"],
        "bytes": len(encoded),
    }


def edit_file(request: dict) -> dict:
    path = resolve_workspace_path(request["path"])
    if not path.is_file():
        return fail(f"not a file: {request['path']}")

    old_text = request.get("old_text")
    new_text = request.get("new_text")
    if not isinstance(old_text, str) or not isinstance(new_text, str):
        return fail("old_text and new_text must be strings")

    text = path.read_text(encoding="utf-8")
    if old_text not in text:
        return fail("old_text not found")

    replacement = text.replace(old_text, new_text, 1)
    if len(replacement.encode("utf-8")) > MAX_WRITE_BYTES:
        return fail(f"edited file is larger than {MAX_WRITE_BYTES} bytes")

    atomic_write(path, replacement)
    return {
        "ok": True,
        "operation": "edit",
        "path": request["path"],
    }


def glob_files(request: dict) -> dict:
    pattern = resolve_glob_pattern(request["pattern"])
    pattern_path = Path(pattern)
    if pattern_path.is_absolute():
        root = next(
            root
            for root in ALLOWED_ROOTS
            if pattern_path == root or pattern_path.is_relative_to(root)
        )
        glob_pattern = pattern_path.relative_to(root)
    else:
        root = WORKSPACE
        glob_pattern = pattern

    matches = []
    for path in root.glob(str(glob_pattern)):
        try:
            resolved = path.resolve()
            if resolved == root or resolved.is_relative_to(root):
                relative = resolved.relative_to(root)
                match = (
                    str(root / relative)
                    if pattern_path.is_absolute()
                    else str(relative)
                )
                matches.append(match)
                if len(matches) >= MAX_GLOB_MATCHES:
                    break
        except OSError:
            continue

    matches.sort()
    return {
        "ok": True,
        "operation": "glob",
        "pattern": pattern,
        "matches": matches,
        "truncated": len(matches) >= MAX_GLOB_MATCHES,
    }


def handle(request: dict) -> dict:
    if not isinstance(request, dict):
        return fail("request must be a JSON object")

    operation = request.get("operation")
    if operation == "read":
        return read_file(request)
    if operation == "write":
        return write_file(request)
    if operation == "edit":
        return edit_file(request)
    if operation == "glob":
        return glob_files(request)
    return fail(f"unknown operation: {operation}")


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except json.JSONDecodeError as error:
        response = fail(f"invalid JSON: {error}")
    else:
        try:
            response = handle(request)
        except (KeyError, TypeError, ValueError, OSError, UnicodeError) as error:
            response = fail(str(error))

    encoded = json.dumps(response, ensure_ascii=False)
    if len(encoded.encode("utf-8")) > MAX_RESULT_BYTES:
        response = fail(f"result is larger than {MAX_RESULT_BYTES} bytes")

    print(json.dumps(response, ensure_ascii=False))
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
