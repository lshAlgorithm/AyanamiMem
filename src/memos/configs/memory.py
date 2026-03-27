from typing import Any, ClassVar

from pydantic import Field, field_validator, model_validator

from memos.configs.base import BaseConfig
from memos.configs.embedder import EmbedderConfigFactory
from memos.configs.graph_db import GraphDBConfigFactory
from memos.configs.internet_retriever import InternetRetrieverConfigFactory
from memos.configs.llm import LLMConfigFactory
from memos.configs.mem_reader import MemReaderConfigFactory
from memos.configs.reranker import RerankerConfigFactory
from memos.configs.vec_db import VectorDBConfigFactory
from memos.exceptions import ConfigurationError
from memos.memories.textual.prefer_text_memory.config import (
    AdderConfigFactory,
    ExtractorConfigFactory,
    RetrieverConfigFactory,
)


# ─── 1. Global Base Memory Config ─────────────────────────────────────────────


class BaseMemoryConfig(BaseConfig):
    """Base configuration class for memories."""

    cube_id: str | None = Field(
        None,
        description="Unique identifier for a MemCube that contains this memory",
    )


class UninitializedMemoryConfig(BaseMemoryConfig):
    """Uninitialized memory configuration class."""


# ─── 2.1. Activation Memory Configs ───────────────────────────────────────────


class BaseActMemoryConfig(BaseMemoryConfig):
    """Base configuration class for activation memories."""

    memory_filename: str = Field(
        "activation_memory.pickle",
        description="Filename for storing memories",
    )


class KVCacheMemoryConfig(BaseActMemoryConfig):
    """LLM KV Cache Memory configuration class."""

    extractor_llm: LLMConfigFactory = Field(
        ...,
        default_factory=LLMConfigFactory,
        description="LLM configuration for the memory extractor",
    )

    @field_validator("extractor_llm")
    @classmethod
    def validate_extractor_llm(cls, extractor_llm: LLMConfigFactory) -> LLMConfigFactory:
        """Validate the extractor_llm field."""
        if extractor_llm.backend not in ["huggingface", "huggingface_singleton", "vllm"]:
            raise ConfigurationError(
                f"KVCacheMemoryConfig requires extractor_llm backend to be 'huggingface' or 'huggingface_singleton', got '{extractor_llm.backend}'"
            )
        return extractor_llm


# ─── 2.2. Parametric Memory Configs ───────────────────────────────────────────


class BaseParaMemoryConfig(BaseMemoryConfig):
    """Base configuration class for parametric memories."""

    memory_filename: str = Field(
        "parametric_memory.adapter",
        description="Filename for storing memories",
    )


class LoRAMemoryConfig(BaseParaMemoryConfig):
    """LoRA memory configuration class."""

    extractor_llm: LLMConfigFactory = Field(
        ...,
        default_factory=LLMConfigFactory,
        description="LLM configuration for the memory extractor",
    )

    @field_validator("extractor_llm")
    @classmethod
    def validate_extractor_llm(cls, extractor_llm: LLMConfigFactory) -> LLMConfigFactory:
        """Validate the extractor_llm field."""
        if extractor_llm.backend not in ["huggingface", "huggingface_singleton"]:
            raise ConfigurationError(
                f"LoRAMemoryConfig requires extractor_llm backend to be 'huggingface' or 'huggingface_singleton', got '{extractor_llm.backend}'"
            )
        return extractor_llm


# ─── 2.3. Textual Memory Configs ──────────────────────────────────────────────


class BaseTextMemoryConfig(BaseMemoryConfig):
    """Base configuration class for textual memories."""

    memory_filename: str = Field(
        "textual_memory.json",
        description="Filename for storing memories",
    )


class NaiveTextMemoryConfig(BaseTextMemoryConfig):
    """Naive textual memory configuration class."""

    extractor_llm: LLMConfigFactory = Field(
        ...,
        default_factory=LLMConfigFactory,
        description="LLM configuration for the memory extractor",
    )


class GeneralTextMemoryConfig(BaseTextMemoryConfig):
    """General memory configuration class."""

    extractor_llm: LLMConfigFactory = Field(
        ...,
        default_factory=LLMConfigFactory,
        description="LLM configuration for the memory extractor",
    )
    vector_db: VectorDBConfigFactory = Field(
        ...,
        default_factory=VectorDBConfigFactory,
        description="Vector database configuration for the memory storage",
    )
    embedder: EmbedderConfigFactory = Field(
        ...,
        default_factory=EmbedderConfigFactory,
        description="Embedder configuration for the memory embedding",
    )


class MarkdownTextMemoryConfig(BaseTextMemoryConfig):
    """Markdown file-based textual memory configuration.

    Each memory is stored as a human-readable ``.md`` file with YAML
    frontmatter (metadata), a body (memory content), and an optional
    ``## Edges`` section that encodes relationships to other nodes.

    Retrieval is done locally via ``sentence-transformers`` cosine
    similarity on the ``key`` field — no external API calls required.
    """

    memory_dir: str = Field(
        default=".memos/markdown",
        description="Root directory where individual .md memory files are stored.",
    )

    extractor_llm: LLMConfigFactory = Field(
        ...,
        default_factory=LLMConfigFactory,
        description="LLM configuration for extracting memories from conversations.",
    )

    embedder: EmbedderConfigFactory = Field(
        ...,
        default_factory=EmbedderConfigFactory,
        description="Embedder configuration for local key-similarity search.",
    )


class HierarchicalMarkdownMemoryConfig(BaseTextMemoryConfig):
    """Lossless hierarchical summary-tree memory stored as Markdown files.

    The filesystem directory structure mirrors the summary tree.  Leaves are
    raw chat chunks; internal nodes are LLM-generated summaries.  No database
    required — everything is ``.md`` files on disk.
    """

    memory_dir: str = Field(
        default=".memos/memory",
        description="Root directory of the memory filesystem tree.",
    )

    extractor_llm: LLMConfigFactory = Field(
        ...,
        default_factory=LLMConfigFactory,
        description="LLM used to summarise chunks into condensed nodes.",
    )

    embedder: EmbedderConfigFactory = Field(
        ...,
        default_factory=EmbedderConfigFactory,
        description="Local embedder for leaf and summary keys.",
    )

    fresh_tail_count: int = Field(
        default=32,
        description="Number of recent leaves always kept in _fresh/, never compacted.",
    )

    leaf_chunk_tokens: int = Field(
        default=2048,
        description="Maximum tokens per leaf chunk before splitting.",
    )

    condensed_target_tokens: int = Field(
        default=512,
        description="Target summary token count for condensed nodes.",
    )

    condensed_min_fanout: int = Field(
        default=4,
        description="Minimum children before a condensed node is created.",
    )

    context_threshold: float = Field(
        default=0.75,
        description="Fraction of leaf_chunk_tokens that triggers compaction.",
    )

    max_depth: int = Field(
        default=-1,
        description="Maximum condensation depth. -1 = unlimited.",
    )

    git_auto_commit: bool = Field(
        default=False,
        description="If True, auto-commit every compaction/mutation to git.",
    )


class TreeTextMemoryConfig(BaseTextMemoryConfig):
    """Tree text memory configuration class."""

    extractor_llm: LLMConfigFactory = Field(
        ...,
        default_factory=LLMConfigFactory,
        description="LLM configuration for the memory extractor",
    )
    dispatcher_llm: LLMConfigFactory = Field(
        ...,
        default_factory=LLMConfigFactory,
        description="LLM configuration for the memory dispatcher_llm in retrieve module",
    )
    embedder: EmbedderConfigFactory = Field(
        ...,
        default_factory=EmbedderConfigFactory,
        description="Embedder configuration for the memory embedding",
    )
    reranker: RerankerConfigFactory | None = Field(
        None,
        description="Reranker configuration (optional, defaults to cosine_local).",
    )
    graph_db: GraphDBConfigFactory = Field(
        ...,
        default_factory=GraphDBConfigFactory,
        description="Graph database configuration for the tree-memory storage",
    )
    internet_retriever: InternetRetrieverConfigFactory | None = Field(
        None,
        description="Internet retriever configuration (optional)",
    )

    reorganize: bool | None = Field(
        False,
        description="Optional description for this memory configuration.",
    )

    memory_size: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Maximum item counts per memory bucket, e.g.: "
            '{"WorkingMemory": 20, "LongTermMemory": 10000, "UserMemory": 10000}'
        ),
    )

    search_strategy: dict[str, Any] | None = Field(
        default=None,
        description=(
            'Set search strategy for this memory configuration.{"bm25": true, "cot": false}'
        ),
    )

    mode: str | None = Field(
        default="sync",
        description=("whether use asynchronous mode in memory add"),
    )
    include_embedding: bool | None = Field(
        default=False,
        description="Whether to include embedding in the memory retrieval",
    )


class SimpleTreeTextMemoryConfig(TreeTextMemoryConfig):
    """Simple tree text memory configuration class."""


class PreferenceTextMemoryConfig(BaseTextMemoryConfig):
    """Preference memory configuration class."""

    extractor_llm: LLMConfigFactory = Field(
        ...,
        default_factory=LLMConfigFactory,
        description="LLM configuration for the memory extractor",
    )
    vector_db: VectorDBConfigFactory = Field(
        ...,
        default_factory=VectorDBConfigFactory,
        description="Vector database configuration for the memory storage",
    )
    embedder: EmbedderConfigFactory = Field(
        ...,
        default_factory=EmbedderConfigFactory,
        description="Embedder configuration for the memory embedding",
    )
    reranker: RerankerConfigFactory | None = Field(
        None,
        description="Reranker configuration (optional).",
    )
    extractor: ExtractorConfigFactory = Field(
        ...,
        default_factory=ExtractorConfigFactory,
        description="Extractor configuration for the memory extracting",
    )
    adder: AdderConfigFactory = Field(
        ...,
        default_factory=AdderConfigFactory,
        description="Adder configuration for the memory adding",
    )
    retriever: RetrieverConfigFactory = Field(
        ...,
        default_factory=RetrieverConfigFactory,
        description="Retriever configuration for the memory retrieving",
    )


class MemFeedbackConfig(BaseMemoryConfig):
    """Memory feedback configuration class."""

    extractor_llm: LLMConfigFactory = Field(
        ...,
        default_factory=LLMConfigFactory,
        description="LLM configuration for the memory extractor",
    )
    embedder: EmbedderConfigFactory = Field(
        ...,
        default_factory=EmbedderConfigFactory,
        description="Embedder configuration for the memory embedding",
    )
    reranker: RerankerConfigFactory | None = Field(
        None,
        description="Reranker configuration (optional).",
    )
    graph_db: GraphDBConfigFactory = Field(
        ...,
        default_factory=GraphDBConfigFactory,
        description="Graph database configuration for the tree-memory storage",
    )
    reorganize: bool | None = Field(
        False,
        description="Optional description for this memory configuration.",
    )

    memory_size: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Maximum item counts per memory bucket, e.g.: "
            '{"WorkingMemory": 20, "LongTermMemory": 10000, "UserMemory": 10000}'
        ),
    )

    mem_reader: MemReaderConfigFactory = Field(
        ...,
        default_factory=MemReaderConfigFactory,
        description="MemReader configuration for the Feedback",
    )


# ─── 3. Global Memory Config Factory ──────────────────────────────────────────


class MemoryConfigFactory(BaseConfig):
    """Factory class for creating memory configurations."""

    backend: str = Field("uninitialized", description="Backend for memory")
    config: dict[str, Any] = Field({}, description="Configuration for the memory backend")

    backend_to_class: ClassVar[dict[str, Any]] = {
        "naive_text": NaiveTextMemoryConfig,
        "general_text": GeneralTextMemoryConfig,
        "markdown_text": MarkdownTextMemoryConfig,
        "markdown_tree_text": MarkdownTextMemoryConfig,
        "hierarchical_markdown": HierarchicalMarkdownMemoryConfig,
        "simple_tree_text": SimpleTreeTextMemoryConfig,
        "tree_text": TreeTextMemoryConfig,
        "pref_text": PreferenceTextMemoryConfig,
        "kv_cache": KVCacheMemoryConfig,
        "vllm_kv_cache": KVCacheMemoryConfig,  # Use same config as kv_cache
        "lora": LoRAMemoryConfig,
        "uninitialized": UninitializedMemoryConfig,
        "mem_feedback": MemFeedbackConfig,
    }

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, backend: str) -> str:
        """Validate the backend field."""
        if backend not in cls.backend_to_class:
            raise ConfigurationError(f"Invalid backend: {backend}")
        return backend

    @model_validator(mode="after")
    def create_config(self) -> "MemoryConfigFactory":
        config_class = self.backend_to_class[self.backend]
        self.config = config_class(**self.config)
        return self
