"""Session manager for hierarchical Markdown memory.

Tracks active sessions and access grants in a single ``_sessions.json``
file stored in the root memory directory (one level above all session
directories).

Layout::

    root_dir/
    ├── _sessions.json          ← this file
    ├── _public/                ← visible to every session (skills, tool schemas, …)
    ├── session-abc/            ← Alice's memory tree
    │   ├── _fresh/
    │   └── 01-japan-trip/
    └── session-def/            ← Bob's memory tree
        └── …

Access model
-----------
* Isolation is **filesystem-first**: each session's directory can be
  given OS-level ``chmod 700`` to prevent cross-session file reads.
  The session manager records grants as metadata; the OS enforces them.
* ``_sessions.json`` is the audit ledger only — it does not gate reads.
  When Feature 3 (symlink sharing) is added, the grant entries here
  drive which symlinks are created in ``_shared/``.

Thread / process safety
-----------------------
All mutations acquire a ``threading.Lock`` and write atomically via a
``tmp → rename`` pattern so concurrent processes (e.g. two agent
instances sharing the same root) never corrupt ``_sessions.json``.
"""

import json
import os
import threading

from datetime import datetime
from typing import Any

from memos.log import get_logger


logger = get_logger(__name__)

_SESSIONS_FILENAME = "_sessions.json"
_PUBLIC_DIR = "_public"
_LOCK_SUFFIX = ".lock"


class SessionManager:
    """Lightweight session registry backed by ``_sessions.json``.

    Parameters
    ----------
    root_dir:
        The root memory directory that contains all session subdirectories
        and the ``_sessions.json`` file.
    """

    def __init__(self, root_dir: str) -> None:
        self.root_dir = os.path.abspath(root_dir)
        self._filepath = os.path.join(self.root_dir, _SESSIONS_FILENAME)
        self._lock = threading.Lock()
        os.makedirs(self.root_dir, exist_ok=True)

    # ── Session lifecycle ─────────────────────────────────────────────────

    def register_session(
        self,
        session_id: str,
        user_id: str = "default",
    ) -> None:
        """Register a new session.  Idempotent: re-registration updates ``user_id``.

        Also creates the session's memory directory.
        """
        with self._lock:
            data = self._read()
            sessions = data.setdefault("sessions", {})
            if session_id not in sessions:
                sessions[session_id] = {
                    "user_id": user_id,
                    "created_at": datetime.now().isoformat(),
                    "is_active": True,
                }
            else:
                sessions[session_id]["user_id"] = user_id
                sessions[session_id]["is_active"] = True
            self._write(data)

        # Ensure the session directory exists
        os.makedirs(self.session_dir(session_id), exist_ok=True)
        logger.info("Session registered: %s (user=%s)", session_id, user_id)

    def deactivate_session(self, session_id: str) -> None:
        """Mark a session as inactive.  Does NOT delete its files."""
        with self._lock:
            data = self._read()
            sessions = data.get("sessions", {})
            if session_id in sessions:
                sessions[session_id]["is_active"] = False
                self._write(data)
        logger.info("Session deactivated: %s", session_id)

    def delete_session(self, session_id: str) -> None:
        """Remove a session from the registry and delete all its files.

        Revokes all grants involving this session first.
        """
        with self._lock:
            data = self._read()
            data.get("sessions", {}).pop(session_id, None)
            data["grants"] = [
                g
                for g in data.get("grants", [])
                if g.get("granting_session") != session_id
                and g.get("grantee_session") != session_id
            ]
            self._write(data)

        import shutil

        session_path = self.session_dir(session_id)
        if os.path.isdir(session_path):
            shutil.rmtree(session_path)
        logger.info("Session deleted: %s", session_id)

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return all registered sessions as a list of metadata dicts."""
        data = self._read()
        return [{"session_id": sid, **meta} for sid, meta in data.get("sessions", {}).items()]

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._read().get("sessions", {})

    def session_dir(self, session_id: str) -> str:
        """Return the absolute path to *session_id*'s memory directory."""
        return os.path.join(self.root_dir, session_id)

    def public_dir(self) -> str:
        """Return the path to the shared ``_public/`` directory."""
        path = os.path.join(self.root_dir, _PUBLIC_DIR)
        os.makedirs(path, exist_ok=True)
        return path

    # ── Access grants (audit ledger for Feature 3) ────────────────────────

    def grant_access(
        self,
        granting_session: str,
        grantee_session: str,
        scope: str = "read",
        subtree_path: str | None = None,
    ) -> None:
        """Record that *granting_session* allows *grantee_session* to access
        either a specific *subtree_path* or the full session (if ``None``).

        This is a metadata record only.  Feature 3 (symlink sharing) reads
        these records to create actual symlinks.
        """
        with self._lock:
            data = self._read()
            grants: list[dict[str, Any]] = data.setdefault("grants", [])
            # Idempotent: skip if identical grant already exists
            for g in grants:
                if (
                    g.get("granting_session") == granting_session
                    and g.get("grantee_session") == grantee_session
                    and g.get("subtree_path") == subtree_path
                    and g.get("scope") == scope
                ):
                    return
            grants.append(
                {
                    "granting_session": granting_session,
                    "grantee_session": grantee_session,
                    "scope": scope,
                    "subtree_path": subtree_path,
                    "granted_at": datetime.now().isoformat(),
                }
            )
            self._write(data)
        logger.info(
            "Access granted: %s → %s (scope=%s, path=%s)",
            granting_session,
            grantee_session,
            scope,
            subtree_path,
        )

    def revoke_access(
        self,
        granting_session: str,
        grantee_session: str,
        subtree_path: str | None = None,
    ) -> None:
        """Remove a grant.  If *subtree_path* is ``None``, revokes all grants
        between the two sessions.
        """
        with self._lock:
            data = self._read()
            before = len(data.get("grants", []))
            data["grants"] = [
                g
                for g in data.get("grants", [])
                if not (
                    g.get("granting_session") == granting_session
                    and g.get("grantee_session") == grantee_session
                    and (subtree_path is None or g.get("subtree_path") == subtree_path)
                )
            ]
            removed = before - len(data["grants"])
            self._write(data)
        logger.info(
            "Access revoked: %s → %s (removed %d grant(s))",
            granting_session,
            grantee_session,
            removed,
        )

    def get_grants_for(self, grantee_session: str) -> list[dict[str, Any]]:
        """Return all grants where *grantee_session* is the recipient."""
        return [
            g for g in self._read().get("grants", []) if g.get("grantee_session") == grantee_session
        ]

    def get_visible_paths(self, session_id: str) -> list[str]:
        """Return all absolute paths visible to *session_id*.

        Always includes:
        - The session's own memory directory
        - The ``_public/`` directory

        Also includes any paths granted by other sessions (used by Feature 3).
        """
        paths: list[str] = [self.session_dir(session_id), self.public_dir()]
        for grant in self.get_grants_for(session_id):
            granter = grant["granting_session"]
            subtree = grant.get("subtree_path")
            if subtree:
                full = os.path.join(self.session_dir(granter), subtree)
            else:
                full = self.session_dir(granter)
            if full not in paths:
                paths.append(full)
        return paths

    # ── Private helpers ───────────────────────────────────────────────────

    def _read(self) -> dict[str, Any]:
        """Read ``_sessions.json``.  Returns empty structure if missing."""
        if not os.path.isfile(self._filepath):
            return {"sessions": {}, "grants": []}
        try:
            with open(self._filepath, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt _sessions.json; starting fresh")
            return {"sessions": {}, "grants": []}

    def _write(self, data: dict[str, Any]) -> None:
        """Atomically write *data* to ``_sessions.json``."""
        tmp = self._filepath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self._filepath)
