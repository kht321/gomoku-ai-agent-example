# deepseek_agent/__init__.py
"""
DeepSeek Gomoku agent package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Exposes `DeepSeekLLMAgent` at package level for convenience:
    from deepseek_agent import DeepSeekLLMAgent
"""

from .deepseek_llm_agent import DeepSeekLLMAgent

__all__ = ["DeepSeekLLMAgent"]
__version__ = "0.2.0"
