"""Tests for hierarchical_markdown.chunker — chat messages to leaf .md files."""

import os

from unittest.mock import MagicMock

import pytest

from memos.memories.textual.hierarchical_markdown.chunker import ChatChunker
from memos.memories.textual.hierarchical_markdown.embeddings import EmbeddingCache
from memos.memories.textual.hierarchical_markdown.fs import FRESH_DIR, read_md


@pytest.fixture()
def mock_embedder():
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]
    return embedder


@pytest.fixture()
def chunker(tmp_path, mock_embedder):
    cache = EmbeddingCache()
    return ChatChunker(str(tmp_path), mock_embedder, leaf_chunk_tokens=200, emb_cache=cache)


class TestChunkerBasic:
    def test_single_message(self, chunker, tmp_path):
        messages = [{"role": "user", "content": "Hello, I'm planning a trip."}]
        written = chunker.chunk(messages, session_id="test-session")
        assert len(written) == 1
        fpath = tmp_path / FRESH_DIR / written[0]
        assert fpath.is_file()

    def test_leaf_metadata(self, chunker, tmp_path):
        messages = [{"role": "user", "content": "I love Python programming."}]
        written = chunker.chunk(messages)
        fpath = tmp_path / FRESH_DIR / written[0]
        meta, body, _edges = read_md(str(fpath))
        assert meta["depth"] == 0
        assert meta["node_kind"] == "leaf"
        assert meta["descendant_count"] == 1
        assert meta["stale"] is False
        assert "Python programming" in body

    def test_multiple_messages_single_chunk(self, chunker, tmp_path):
        messages = [
            {"role": "user", "content": "Hi there."},
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "Tell me about Japan."},
        ]
        written = chunker.chunk(messages)
        assert len(written) == 1
        fpath = tmp_path / FRESH_DIR / written[0]
        _meta, body, _edges = read_md(str(fpath))
        assert "User:" in body
        assert "Assistant:" in body

    def test_splits_at_token_boundary(self, tmp_path, mock_embedder):
        # Very small token limit to force splitting
        chunker = ChatChunker(str(tmp_path), mock_embedder, leaf_chunk_tokens=20)
        messages = [
            {"role": "user", "content": "A" * 200},
            {"role": "assistant", "content": "B" * 200},
        ]
        written = chunker.chunk(messages)
        assert len(written) == 2

    def test_empty_messages(self, chunker):
        assert chunker.chunk([]) == []

    def test_sequential_numbering(self, chunker, tmp_path):
        chunker.chunk([{"role": "user", "content": "First conversation."}])
        chunker.chunk([{"role": "user", "content": "Second conversation."}])
        fresh_dir = tmp_path / FRESH_DIR
        files = sorted(f for f in os.listdir(str(fresh_dir)) if f.endswith(".md"))
        assert len(files) == 2
        assert files[0].startswith("001-")
        assert files[1].startswith("002-")

    def test_embeddings_cache_updated(self, chunker, tmp_path):
        """Leaf key should be registered in the in-memory EmbeddingCache."""
        chunker.chunk([{"role": "user", "content": "Test embedding index."}])
        # Access the cache that was passed in during fixture construction
        cache = chunker._emb_cache
        fresh_dir = str(tmp_path / FRESH_DIR)
        children = cache.children_of(fresh_dir)
        assert len(children) == 1

    def test_session_id_in_metadata(self, chunker, tmp_path):
        written = chunker.chunk([{"role": "user", "content": "Hello."}], session_id="my-session")
        fpath = tmp_path / FRESH_DIR / written[0]
        meta, _body, _edges = read_md(str(fpath))
        assert meta["session_id"] == "my-session"

    def test_multipart_content(self, chunker, tmp_path):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this image."},
                    {"type": "image_url", "url": "http://example.com/img.png"},
                ],
            }
        ]
        written = chunker.chunk(messages)
        assert len(written) == 1
        fpath = tmp_path / FRESH_DIR / written[0]
        _meta, body, _edges = read_md(str(fpath))
        assert "Look at this image." in body
