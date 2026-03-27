"""Async-safe embedding index stored as ``_embeddings.json`` per directory.

Each directory in the memory tree contains an ``_embeddings.json`` mapping
filenames to embedding vectors.  The assembler reads this during top-down
search; the compactor and updater write it after creating/updating nodes.

Concurrent safety is achieved via:
- File-level locking (``fcntl.flock``) for mutual exclusion
- Atomic rename (write to ``.tmp``, then ``os.rename``) for crash safety
- Monotonic ``_version`` counter for staleness detection
"""

import fcntl
import json
import os

from typing import Any

from memos.log import get_logger


logger = get_logger(__name__)

EMBEDDINGS_FILENAME = "_embeddings.json"
_LOCK_SUFFIX = ".lock"


class EmbeddingIndex:
    """Manages ``_embeddings.json`` in a single directory."""

    def __init__(self, dir_path: str) -> None:
        self.dir_path = dir_path
        self._filepath = os.path.join(dir_path, EMBEDDINGS_FILENAME)
        self._lockpath = self._filepath + _LOCK_SUFFIX

    def read(self) -> dict[str, list[float]]:
        """Read the embedding index.  Returns ``{}`` if the file is missing."""
        if not os.path.isfile(self._filepath):
            return {}
        try:
            with open(self._filepath, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt or unreadable %s, returning empty", self._filepath)
            return {}
        # Strip metadata keys
        return {k: v for k, v in data.items() if not k.startswith("_")}

    def read_meta(self) -> dict[str, Any]:
        """Read the full index including metadata (``_version``, ``_model``)."""
        if not os.path.isfile(self._filepath):
            return {}
        try:
            with open(self._filepath, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def update(self, entries: dict[str, list[float]], model: str | None = None) -> None:
        """Atomic read-modify-write: merge *entries* into the index.

        Thread/process-safe via file lock + atomic rename.
        """
        os.makedirs(self.dir_path, exist_ok=True)
        lock_fd = self._acquire_lock()
        try:
            data = self._read_raw()
            version = data.get("_version", 0)
            data.update(entries)
            data["_version"] = version + 1
            if model:
                data["_model"] = model
            self._write_atomic(data)
        finally:
            self._release_lock(lock_fd)

    def remove(self, keys: list[str]) -> None:
        """Remove entries for *keys* from the index.  Thread-safe."""
        os.makedirs(self.dir_path, exist_ok=True)
        lock_fd = self._acquire_lock()
        try:
            data = self._read_raw()
            changed = False
            for k in keys:
                if k in data:
                    del data[k]
                    changed = True
            if changed:
                data["_version"] = data.get("_version", 0) + 1
                self._write_atomic(data)
        finally:
            self._release_lock(lock_fd)

    def version(self) -> int:
        """Return the current version counter (0 if file missing)."""
        meta = self.read_meta()
        return meta.get("_version", 0)

    # ── Private helpers ───────────────────────────────────────────────────

    def _read_raw(self) -> dict[str, Any]:
        """Read the JSON file without lock (caller must hold lock)."""
        if not os.path.isfile(self._filepath):
            return {"_version": 0}
        try:
            with open(self._filepath, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"_version": 0}

    def _write_atomic(self, data: dict[str, Any]) -> None:
        """Write *data* to a temp file, then atomically rename."""
        tmp_path = self._filepath + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp_path, self._filepath)

    def _acquire_lock(self) -> int:
        """Acquire an exclusive file lock.  Returns the lock file descriptor."""
        fd = os.open(self._lockpath, os.O_CREAT | os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX)
        return fd

    def _release_lock(self, fd: int) -> None:
        """Release the file lock."""
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
