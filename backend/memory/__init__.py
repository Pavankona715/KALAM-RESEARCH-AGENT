"""
Memory Package
==============
Short-term (Redis) and long-term (Qdrant) memory for agents.

Import from here:
    from backend.memory import MemoryManager, get_memory_manager
"""

from backend.memory.base import MemoryEntry, ConversationTurn
from backend.memory.short_term import ShortTermMemory, get_short_term_memory
from backend.memory.long_term import LongTermMemory, get_long_term_memory
from backend.memory.manager import MemoryManager, MemoryContext, get_memory_manager

__all__ = [
    "MemoryEntry",
    "ConversationTurn",
    "ShortTermMemory",
    "get_short_term_memory",
    "LongTermMemory",
    "get_long_term_memory",
    "MemoryManager",
    "MemoryContext",
    "get_memory_manager",
]