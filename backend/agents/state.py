"""
Agent State
===========
Defines the shared state that flows through every node in the LangGraph graph.

Design rules:
- State is immutable between nodes — each node returns a NEW state dict
- LangGraph merges return values with the existing state (reducer pattern)
- Lists use the `add_messages` reducer which appends rather than replaces
- All fields must be serializable (for checkpointing/replay)

The AgentState TypedDict is the single source of truth for what an
agent "knows" at any point during execution.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Complete state of a running agent.

    Fields with Annotated[list, add_messages] use LangGraph's built-in
    reducer — new messages are APPENDED to the list rather than replacing it.
    All other fields are replaced with the latest value on each update.
    """

    # Conversation messages (user, assistant, tool results)
    # add_messages reducer: new messages are appended, not replaced
    messages: Annotated[list[BaseMessage], add_messages]

    # Agent metadata
    session_id: str
    user_id: str
    agent_type: str

    # Execution tracking
    step_count: int               # Incremented each reasoning cycle
    max_steps: int                # Safety limit to prevent infinite loops

    # Tool tracking for the current step
    pending_tool_calls: list[dict]    # Tool calls requested by LLM
    tool_results: list[dict]          # Results from executed tools

    # Final output
    final_answer: Optional[str]       # Set when agent is done
    error: Optional[str]              # Set if agent fails

    # Retrieved documents for RAG context
    retrieved_docs: list[dict]        # Serialized RetrievedDocument objects

    # Run metadata (populated by orchestrator)
    run_id: Optional[str]
    metadata: dict[str, Any]


def create_initial_state(
    user_message: str,
    session_id: str,
    user_id: str,
    agent_type: str = "react",
    max_steps: int = 10,
    conversation_history: Optional[list[dict]] = None,
    retrieved_docs: Optional[list] = None,
    run_id: Optional[str] = None,
) -> AgentState:
    """
    Create the initial state for a new agent run.

    Args:
        user_message: The user's query
        session_id: Chat session ID for memory/history lookup
        user_id: User ID for permission checks
        agent_type: Which agent type is running
        max_steps: Maximum reasoning cycles before forced stop
        conversation_history: Prior messages to include as context
        run_id: Optional tracking ID for the agent run record
    """
    from langchain_core.messages import HumanMessage, AIMessage

    # Build initial message list from history + current message
    messages: list[BaseMessage] = []

    if conversation_history:
        for msg in conversation_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

    # Add the current user message
    messages.append(HumanMessage(content=user_message))

    # Serialize retrieved docs for state storage
    serialized_docs = []
    if retrieved_docs:
        for doc in retrieved_docs:
            if hasattr(doc, "content"):
                serialized_docs.append({
                    "content": doc.content,
                    "source": doc.source,
                    "score": doc.score,
                    "doc_id": doc.doc_id,
                    "chunk_index": doc.chunk_index,
                })
            elif isinstance(doc, dict):
                serialized_docs.append(doc)

    return AgentState(
        messages=messages,
        session_id=session_id,
        user_id=user_id,
        agent_type=agent_type,
        step_count=0,
        max_steps=max_steps,
        pending_tool_calls=[],
        tool_results=[],
        final_answer=None,
        error=None,
        retrieved_docs=serialized_docs,
        run_id=run_id,
        metadata={},
    )