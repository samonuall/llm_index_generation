"""Wrapper for AI assistant baseline evaluation."""
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from agent import AIAssistantBaseline

Preprocessor = AIAssistantBaseline