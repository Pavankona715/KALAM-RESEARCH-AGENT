"""
Context Builder
===============
Before every LLM call, this module assembles the full context:

    System Prompt
    + Conversation History    (from memory/DB)
    + Retrieved Documents     (from RAG pipeline)
    + Tool Outputs            (from previous agent steps)
    + Memory Results          (from long-term memory)
    = Messages[] sent to LLM

This is one of the most important modules — bad context assembly is
the #1 cause of poor agent performance.

Design principles:
- Context has a budget (max tokens). We must fit within it.
- Prioritize: recent conversation > retrieved docs > old history
- System prompt is always first, always present
- Never silently truncate — log what was dropped and why
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from backend.config.prompts import REACT_AGENT_SYSTEM_PROMPT
from backend.llm.base import Message
from backend.observability.logger import get_logger

logger = get_logger(__name__)

# Rough token estimate: 1 token ≈ 4 chars (conservative)
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Fast token estimate without calling the tokenizer."""
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class RetrievedDocument:
    """A document chunk retrieved from the vector store."""
    content: str
    source: str           # filename or URL
    score: float          # similarity score 0.0 - 1.0
    doc_id: str
    chunk_index: int = 0


@dataclass
class ContextConfig:
    """Controls how the context is assembled."""
    max_tokens: int = 12_000          # Total budget for the context
    system_prompt_tokens: int = 500   # Reserved for system prompt
    history_max_messages: int = 20    # Max conversation turns to include
    max_retrieved_docs: int = 5       # Max RAG results to include
    min_doc_score: float = 0.5        # Minimum similarity score for inclusion
    include_timestamps: bool = False  # Add timestamps to messages (verbose)


@dataclass
class ContextInput:
    """All raw inputs that the builder synthesizes into LLM messages."""
    user_message: str
    conversation_history: list[dict] = field(default_factory=list)
    # [{"role": "user"|"assistant", "content": "..."}]

    retrieved_docs: list[RetrievedDocument] = field(default_factory=list)
    tool_outputs: list[dict] = field(default_factory=list)
    # [{"tool": "web_search", "output": "..."}]

    memory_results: list[str] = field(default_factory=list)
    # Relevant snippets from long-term memory

    system_prompt_override: Optional[str] = None
    agent_type: str = "react"


@dataclass
class BuiltContext:
    """The assembled context ready to be sent to the LLM."""
    messages: list[Message]
    estimated_tokens: int
    retrieved_doc_ids: list[str]     # Track which docs were included
    truncated: bool = False          # True if context was truncated due to token limit


class ContextBuilder:
    """
    Assembles LLM context from all available sources.

    Usage:
        builder = ContextBuilder()
        context = builder.build(ContextInput(
            user_message="What is RAG?",
            conversation_history=[...],
            retrieved_docs=[...],
        ))
        response = await llm.complete(LLMRequest(messages=context.messages))
    """

    def __init__(self, config: Optional[ContextConfig] = None):
        self.config = config or ContextConfig()

    def build(self, inputs: ContextInput) -> BuiltContext:
        """
        Assemble the full context.

        Assembly order (priority high → low):
        1. System prompt (always included, never truncated)
        2. Memory results (high value, recent context)
        3. Retrieved documents (RAG context)
        4. Conversation history (older messages may be dropped)
        5. Tool outputs (intermediate results)
        6. Current user message (always included last)
        """
        messages: list[Message] = []
        total_tokens = 0
        remaining_budget = self.config.max_tokens
        truncated = False

        # ── 1. System Prompt ─────────────────────────────────────────────────
        system_prompt = inputs.system_prompt_override or self._build_system_prompt(inputs)
        messages.append(Message(role="system", content=system_prompt))
        system_tokens = estimate_tokens(system_prompt)
        total_tokens += system_tokens
        remaining_budget -= system_tokens

        # ── 2. Memory results (injected after system, before history) ────────
        if inputs.memory_results and remaining_budget > 500:
            memory_section = self._format_memory(inputs.memory_results)
            memory_tokens = estimate_tokens(memory_section)
            if memory_tokens < remaining_budget - 500:
                messages.append(Message(
                    role="system",
                    content=memory_section,
                ))
                total_tokens += memory_tokens
                remaining_budget -= memory_tokens

        # ── 3. Retrieved Documents ───────────────────────────────────────────
        doc_ids_included = []
        if inputs.retrieved_docs and remaining_budget > 1000:
            rag_section, included_docs = self._format_retrieved_docs(
                inputs.retrieved_docs,
                token_budget=min(remaining_budget // 3, 4000),
            )
            if rag_section:
                messages.append(Message(role="system", content=rag_section))
                rag_tokens = estimate_tokens(rag_section)
                total_tokens += rag_tokens
                remaining_budget -= rag_tokens
                doc_ids_included = [d.doc_id for d in included_docs]

        # ── 4. Conversation History ──────────────────────────────────────────
        history = self._trim_history(
            inputs.conversation_history,
            token_budget=remaining_budget - 1000,  # Reserve 1k for user message
        )

        if len(history) < len(inputs.conversation_history):
            truncated = True
            logger.debug(
                "context_history_truncated",
                original=len(inputs.conversation_history),
                kept=len(history),
            )

        for msg in history:
            messages.append(Message(role=msg["role"], content=msg["content"]))
            total_tokens += estimate_tokens(msg["content"])

        # ── 5. Tool Outputs (inject as assistant messages) ───────────────────
        for tool_output in inputs.tool_outputs:
            content = f"[Tool: {tool_output.get('tool', 'unknown')}]\n{tool_output.get('output', '')}"
            messages.append(Message(role="assistant", content=content))
            total_tokens += estimate_tokens(content)

        # ── 6. Current User Message (always last) ────────────────────────────
        messages.append(Message(role="user", content=inputs.user_message))
        total_tokens += estimate_tokens(inputs.user_message)

        logger.debug(
            "context_built",
            total_messages=len(messages),
            estimated_tokens=total_tokens,
            rag_docs_included=len(doc_ids_included),
            history_messages=len(history),
            truncated=truncated,
        )

        return BuiltContext(
            messages=messages,
            estimated_tokens=total_tokens,
            retrieved_doc_ids=doc_ids_included,
            truncated=truncated,
        )

    def _build_system_prompt(self, inputs: ContextInput) -> str:
        """Build the system prompt with dynamic context summary."""
        context_parts = []
        if inputs.retrieved_docs:
            context_parts.append(f"{len(inputs.retrieved_docs)} relevant documents retrieved")
        if inputs.conversation_history:
            context_parts.append(f"{len(inputs.conversation_history)} previous messages in context")
        if inputs.memory_results:
            context_parts.append(f"{len(inputs.memory_results)} memory items loaded")

        context_summary = (
            "\n".join(f"- {p}" for p in context_parts)
            if context_parts else "No prior context"
        )

        return REACT_AGENT_SYSTEM_PROMPT.format(
            context_summary=context_summary,
            current_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        )

    def _format_retrieved_docs(
        self,
        docs: list[RetrievedDocument],
        token_budget: int,
    ) -> tuple[str, list[RetrievedDocument]]:
        """Format retrieved documents into a context block."""
        # Filter by minimum score and sort by relevance
        filtered = [
            d for d in docs
            if d.score >= self.config.min_doc_score
        ]
        filtered.sort(key=lambda d: d.score, reverse=True)
        filtered = filtered[:self.config.max_retrieved_docs]

        if not filtered:
            return "", []

        sections = ["## Retrieved Knowledge\n"]
        included = []
        tokens_used = estimate_tokens(sections[0])

        for i, doc in enumerate(filtered):
            section = (
                f"### Source {i+1}: {doc.source} (relevance: {doc.score:.2f})\n"
                f"{doc.content}\n"
            )
            section_tokens = estimate_tokens(section)
            if tokens_used + section_tokens > token_budget:
                break
            sections.append(section)
            tokens_used += section_tokens
            included.append(doc)

        return "\n".join(sections), included

    def _format_memory(self, memory_items: list[str]) -> str:
        """Format long-term memory results into a context block."""
        items_text = "\n".join(f"- {item}" for item in memory_items[:5])
        return f"## Relevant Memory\n{items_text}"

    def _trim_history(
        self,
        history: list[dict],
        token_budget: int,
    ) -> list[dict]:
        """
        Trim conversation history to fit within token budget.
        Always keeps the most recent messages (drops oldest first).
        Respects max_messages limit.
        """
        # Apply message count limit
        limited = history[-self.config.history_max_messages:]

        # Apply token budget (iterate from most recent, backwards)
        kept = []
        tokens_used = 0

        for msg in reversed(limited):
            msg_tokens = estimate_tokens(msg.get("content", ""))
            if tokens_used + msg_tokens > token_budget:
                break
            kept.insert(0, msg)
            tokens_used += msg_tokens

        return kept