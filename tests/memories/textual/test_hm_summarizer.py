"""Tests for hierarchical_markdown.summarizer — LLM summarisation."""

import json

from unittest.mock import MagicMock

import pytest

from memos.memories.textual.hierarchical_markdown.fs import write_md
from memos.memories.textual.hierarchical_markdown.summarizer import HierarchicalSummarizer


@pytest.fixture()
def mock_llm():
    llm = MagicMock()
    llm.generate.return_value = json.dumps(
        {
            "key": "Japan trip planning",
            "summary": "User plans to visit Japan. Budget 3000 USD. Interested in Kyoto.",
            "tags": ["travel", "japan"],
        }
    )
    return llm


@pytest.fixture()
def mock_embedder():
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1, 0.2, 0.3]]
    return embedder


@pytest.fixture()
def summarizer(mock_llm, mock_embedder):
    return HierarchicalSummarizer(mock_llm, mock_embedder, condensed_target_tokens=512)


class TestSummarize:
    def test_basic_summarization(self, summarizer, tmp_path):
        # Write two children
        f1 = str(tmp_path / "001-destinations.md")
        f2 = str(tmp_path / "002-budget.md")
        write_md(f1, "User wants to visit Kyoto.", {"key": "destinations", "depth": 0})
        write_md(f2, "Budget is 3000 USD.", {"key": "budget", "depth": 0})

        summary, key, tags = summarizer.summarize([f1, f2], depth=1)
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert isinstance(key, str)
        assert isinstance(tags, list)

    def test_llm_called_with_children_content(self, summarizer, mock_llm, tmp_path):
        f1 = str(tmp_path / "001-test.md")
        write_md(f1, "Important content here.", {"key": "test", "depth": 0})

        summarizer.summarize([f1], depth=1)
        mock_llm.generate.assert_called_once()
        prompt = mock_llm.generate.call_args[0][0][0]["content"]
        assert "Important content here." in prompt

    def test_escalation_on_long_summary(self, tmp_path, mock_embedder):
        llm = MagicMock()
        # First call returns too-long summary, second returns acceptable one
        llm.generate.side_effect = [
            json.dumps({"key": "long", "summary": "x" * 5000, "tags": []}),
            json.dumps({"key": "short", "summary": "Concise.", "tags": ["ok"]}),
        ]
        s = HierarchicalSummarizer(llm, mock_embedder, condensed_target_tokens=100)
        f1 = str(tmp_path / "001.md")
        write_md(f1, "Content.", {"key": "test", "depth": 0})

        summary, _key, _tags = s.summarize([f1], depth=1)
        assert llm.generate.call_count == 2
        assert summary == "Concise."

    def test_truncation_fallback(self, tmp_path, mock_embedder):
        llm = MagicMock()
        llm.generate.side_effect = Exception("LLM down")
        s = HierarchicalSummarizer(llm, mock_embedder, condensed_target_tokens=50)
        f1 = str(tmp_path / "001.md")
        write_md(f1, "Some content to summarise.", {"key": "test", "depth": 0})

        summary, key, _tags = s.summarize([f1], depth=1)
        assert isinstance(summary, str)
        assert key == "summary"  # fallback key

    def test_update_mode(self, summarizer, mock_llm, tmp_path):
        f1 = str(tmp_path / "001.md")
        write_md(f1, "Updated content.", {"key": "updated", "depth": 0})

        summarizer.summarize([f1], depth=1, previous_summary="Old summary text.")
        prompt = mock_llm.generate.call_args[0][0][0]["content"]
        assert "Old summary text." in prompt

    def test_depth2_uses_different_prompt(self, summarizer, mock_llm, tmp_path):
        f1 = str(tmp_path / "001.md")
        write_md(f1, "Content.", {"key": "test", "depth": 1})

        summarizer.summarize([f1], depth=2)
        prompt = mock_llm.generate.call_args[0][0][0]["content"]
        assert "thematic" in prompt.lower() or "summaries" in prompt.lower()

    def test_depth3_uses_abstract_prompt(self, summarizer, mock_llm, tmp_path):
        f1 = str(tmp_path / "001.md")
        write_md(f1, "Content.", {"key": "test", "depth": 2})

        summarizer.summarize([f1], depth=3)
        prompt = mock_llm.generate.call_args[0][0][0]["content"]
        assert "abstract" in prompt.lower() or "overview" in prompt.lower()


class TestSummarizeForKey:
    def test_basic(self, summarizer, mock_llm):
        mock_llm.generate.return_value = json.dumps({"key": "short key", "tags": ["tag1"]})
        key, tags = summarizer.summarize_for_key("Some text to extract key from.")
        assert key == "short key"
        assert tags == ["tag1"]

    def test_fallback_on_error(self, mock_embedder):
        llm = MagicMock()
        llm.generate.side_effect = Exception("fail")
        s = HierarchicalSummarizer(llm, mock_embedder)
        key, _tags = s.summarize_for_key("Some text content here.")
        assert isinstance(key, str)
        assert len(key) > 0
