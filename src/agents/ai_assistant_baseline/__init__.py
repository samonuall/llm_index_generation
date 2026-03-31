"""
AI Assistant baseline agent.
Uses Claude Code, GitHub Copilot, or Gemini to generate preprocessing strategies.
"""

from .agent import AIAssistantBaseline

def create_agent(**kwargs):
    """Factory function for AI assistant baseline."""
    return AIAssistantBaseline(**kwargs)

__all__ = ['AIAssistantBaseline', 'create_agent']