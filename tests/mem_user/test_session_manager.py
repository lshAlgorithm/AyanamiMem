"""Tests for SessionManager — lightweight session registry."""

import os
import threading

import pytest

from memos.mem_user.session_manager import SessionManager


@pytest.fixture()
def mgr(tmp_path):
    return SessionManager(str(tmp_path / "memos_root"))


# ── Registration ──────────────────────────────────────────────────────────────


class TestRegister:
    def test_register_creates_session(self, mgr):
        mgr.register_session("s1", user_id="alice")
        assert mgr.session_exists("s1")

    def test_register_creates_dir(self, mgr):
        mgr.register_session("s1")
        assert os.path.isdir(mgr.session_dir("s1"))

    def test_register_idempotent(self, mgr):
        mgr.register_session("s1", user_id="alice")
        mgr.register_session("s1", user_id="alice2")  # re-register
        sessions = {s["session_id"]: s for s in mgr.list_sessions()}
        assert sessions["s1"]["user_id"] == "alice2"
        assert len(sessions) == 1

    def test_register_stores_metadata(self, mgr):
        mgr.register_session("s1", user_id="alice")
        sessions = {s["session_id"]: s for s in mgr.list_sessions()}
        assert sessions["s1"]["is_active"] is True
        assert "created_at" in sessions["s1"]

    def test_deactivate(self, mgr):
        mgr.register_session("s1")
        mgr.deactivate_session("s1")
        sessions = {s["session_id"]: s for s in mgr.list_sessions()}
        assert sessions["s1"]["is_active"] is False

    def test_delete_removes_dir_and_registry(self, mgr):
        mgr.register_session("s1")
        session_path = mgr.session_dir("s1")
        assert os.path.isdir(session_path)
        mgr.delete_session("s1")
        assert not mgr.session_exists("s1")
        assert not os.path.isdir(session_path)

    def test_session_dir_path(self, mgr):
        mgr.register_session("my-session")
        path = mgr.session_dir("my-session")
        assert path.endswith("my-session")
        assert os.path.isdir(path)

    def test_public_dir_created(self, mgr):
        public = mgr.public_dir()
        assert os.path.isdir(public)
        assert public.endswith("_public")


# ── Grants ────────────────────────────────────────────────────────────────────


class TestGrants:
    def test_grant_and_revoke(self, mgr):
        mgr.register_session("alice")
        mgr.register_session("bob")
        mgr.grant_access("alice", "bob", scope="read")
        grants = mgr.get_grants_for("bob")
        assert len(grants) == 1
        assert grants[0]["granting_session"] == "alice"
        assert grants[0]["scope"] == "read"

        mgr.revoke_access("alice", "bob")
        assert mgr.get_grants_for("bob") == []

    def test_grant_idempotent(self, mgr):
        mgr.register_session("alice")
        mgr.register_session("bob")
        mgr.grant_access("alice", "bob")
        mgr.grant_access("alice", "bob")  # duplicate
        assert len(mgr.get_grants_for("bob")) == 1

    def test_grant_subtree(self, mgr):
        mgr.register_session("alice")
        mgr.register_session("bob")
        mgr.grant_access("alice", "bob", subtree_path="01-japan-trip/")
        grants = mgr.get_grants_for("bob")
        assert grants[0]["subtree_path"] == "01-japan-trip/"

    def test_revoke_subtree_only(self, mgr):
        mgr.register_session("alice")
        mgr.register_session("bob")
        mgr.grant_access("alice", "bob", subtree_path="01-japan/")
        mgr.grant_access("alice", "bob", subtree_path="02-work/")
        mgr.revoke_access("alice", "bob", subtree_path="01-japan/")
        remaining = mgr.get_grants_for("bob")
        assert len(remaining) == 1
        assert remaining[0]["subtree_path"] == "02-work/"

    def test_revoke_all_on_delete(self, mgr):
        mgr.register_session("alice")
        mgr.register_session("bob")
        mgr.grant_access("alice", "bob")
        mgr.delete_session("alice")
        assert mgr.get_grants_for("bob") == []

    def test_no_grants_returns_empty(self, mgr):
        mgr.register_session("alice")
        assert mgr.get_grants_for("alice") == []


# ── Visible paths ─────────────────────────────────────────────────────────────


class TestVisiblePaths:
    def test_own_dir_and_public(self, mgr):
        mgr.register_session("alice")
        paths = mgr.get_visible_paths("alice")
        assert mgr.session_dir("alice") in paths
        assert mgr.public_dir() in paths

    def test_includes_granted_session(self, mgr):
        mgr.register_session("alice")
        mgr.register_session("bob")
        mgr.grant_access("alice", "bob")
        paths = mgr.get_visible_paths("bob")
        assert mgr.session_dir("alice") in paths
        assert mgr.session_dir("bob") in paths

    def test_includes_subtree_path(self, mgr):
        mgr.register_session("alice")
        mgr.register_session("bob")
        mgr.grant_access("alice", "bob", subtree_path="01-japan/")
        paths = mgr.get_visible_paths("bob")
        expected = os.path.join(mgr.session_dir("alice"), "01-japan/")
        assert expected in paths
        # Full alice session dir should NOT be in paths
        assert mgr.session_dir("alice") not in paths

    def test_no_duplicate_paths(self, mgr):
        mgr.register_session("alice")
        mgr.grant_access("alice", "alice")  # grant to self (edge case)
        paths = mgr.get_visible_paths("alice")
        assert len(paths) == len(set(paths))


# ── Persistence ───────────────────────────────────────────────────────────────


class TestPersistence:
    def test_sessions_survive_reload(self, tmp_path):
        root = str(tmp_path / "root")
        mgr1 = SessionManager(root)
        mgr1.register_session("s1", user_id="alice")
        mgr1.grant_access("s1", "s2")

        # Create a fresh instance pointing to the same root
        mgr2 = SessionManager(root)
        assert mgr2.session_exists("s1")
        assert len(mgr2.get_grants_for("s2")) == 1

    def test_corrupt_json_resets_gracefully(self, tmp_path):
        root = str(tmp_path / "root")
        mgr = SessionManager(root)
        os.makedirs(root, exist_ok=True)
        # Write corrupt JSON
        with open(os.path.join(root, "_sessions.json"), "w") as f:
            f.write("{NOT VALID JSON")
        # Should not raise; should return empty state
        assert mgr.list_sessions() == []

    def test_atomic_write_no_tmp_left(self, tmp_path):
        root = str(tmp_path / "root")
        mgr = SessionManager(root)
        mgr.register_session("s1")
        files = os.listdir(root)
        assert not any(f.endswith(".tmp") for f in files)


# ── Concurrency ───────────────────────────────────────────────────────────────


class TestConcurrency:
    def test_concurrent_registration(self, tmp_path):
        root = str(tmp_path / "root")
        mgr = SessionManager(root)
        errors: list[Exception] = []

        def register(i: int) -> None:
            try:
                mgr.register_session(f"session-{i}", user_id=f"user-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(mgr.list_sessions()) == 10

    def test_concurrent_grants(self, tmp_path):
        root = str(tmp_path / "root")
        mgr = SessionManager(root)
        mgr.register_session("alice")
        errors: list[Exception] = []

        def add_grant(i: int) -> None:
            try:
                mgr.register_session(f"user-{i}")
                mgr.grant_access("alice", f"user-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_grant, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
