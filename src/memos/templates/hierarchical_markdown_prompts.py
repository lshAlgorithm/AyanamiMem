"""Prompt templates for hierarchical Markdown memory summarisation."""

# ── Leaf summary (depth 0 → 1) ───────────────────────────────────────────────

LEAF_SUMMARY_PROMPT = """\
You are a memory summariser.  Below is a raw conversation chunk between a user \
and an AI assistant.

<conversation>
${conversation}
</conversation>

${previous_context}

Produce a concise narrative summary (max ${target_tokens} tokens) that captures:
1. Key facts, decisions, and preferences expressed by the user.
2. Important recommendations or information provided by the assistant.
3. Any action items, plans, or commitments mentioned.

Also produce a short key (5-15 words) that describes the main topic.
Also produce a list of relevant tags (1-5 tags).

Return ONLY valid JSON:
{"key": "...", "summary": "...", "tags": ["...", "..."]}
"""

# ── Depth-1 condensation (depth 1 → 2) ───────────────────────────────────────

CONDENSED_D1_PROMPT = """\
You are a memory summariser.  Below are summaries of several related \
conversation segments.

<summaries>
${children_summaries}
</summaries>

${previous_context}

Produce a thematic summary (max ${target_tokens} tokens) that:
1. Groups related topics together.
2. Preserves key facts, numbers, names, and decisions.
3. Notes the overall arc or progression of the conversations.

Also produce a short key (5-15 words) describing the overarching theme.
Also produce a list of tags (1-5 tags).

Return ONLY valid JSON:
{"key": "...", "summary": "...", "tags": ["...", "..."]}
"""

# ── Depth 2+ condensation ────────────────────────────────────────────────────

CONDENSED_D2_PLUS_PROMPT = """\
You are a memory summariser working at a high level of abstraction.  \
Below are summaries of broad topic groups.

<summaries>
${children_summaries}
</summaries>

${previous_context}

Produce an abstract overview (max ${target_tokens} tokens) that:
1. Identifies major themes and patterns across topics.
2. Highlights the most durable and important facts.
3. Notes relationships between topic groups.

Also produce a short key (5-15 words) for this overview.
Also produce a list of tags (1-5 tags).

Return ONLY valid JSON:
{"key": "...", "summary": "...", "tags": ["...", "..."]}
"""

# ── Aggressive summary fallback ───────────────────────────────────────────────

AGGRESSIVE_SUMMARY_PROMPT = """\
You are a memory summariser under strict space constraints.  \
Below is content that must be compressed.

<content>
${content}
</content>

Produce the most compact summary possible (max ${target_tokens} tokens).  \
Keep ONLY:
- Concrete facts (names, numbers, dates, places)
- Firm decisions and preferences
- Action items

Drop all pleasantries, hedging, examples, and elaboration.

Also produce a short key (5-10 words).
Also produce 1-3 tags.

Return ONLY valid JSON:
{"key": "...", "summary": "...", "tags": ["...", "..."]}
"""

# ── Update summary (delta-aware re-summarisation) ────────────────────────────

UPDATE_SUMMARY_PROMPT = """\
You are a memory summariser.  A child node has been updated.  \
Below is the previous summary and the updated children.

<previous_summary>
${previous_summary}
</previous_summary>

<updated_children>
${children_summaries}
</updated_children>

Re-produce the summary (max ${target_tokens} tokens), incorporating changes \
from the updated children.  Preserve information from unchanged children.

Also produce an updated key (5-15 words).
Also produce updated tags (1-5 tags).

Return ONLY valid JSON:
{"key": "...", "summary": "...", "tags": ["...", "..."]}
"""

# ── Leaf key extraction (lightweight) ─────────────────────────────────────────

LEAF_KEY_PROMPT = """\
Given this conversation chunk, produce a short descriptive key (5-15 words) \
and 1-5 tags.

<conversation>
${conversation}
</conversation>

Return ONLY valid JSON:
{"key": "...", "tags": ["...", "..."]}
"""
