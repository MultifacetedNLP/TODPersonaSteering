"""Backward-compatibility shim — SGD multi-domain simulator.

Use :class:`llm_interaction.sgd_chat_simulator.SGDChatSimulator` directly.
This module re-exports ``MultiDomainInteractiveChatSession`` as an alias so
that existing import sites continue to work without change.
"""
from llm_interaction.sgd_chat_simulator import SGDChatSimulator as MultiDomainInteractiveChatSession

__all__ = ["MultiDomainInteractiveChatSession"]
