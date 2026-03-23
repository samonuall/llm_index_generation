"""
AI Assistant Baseline Agent.
One-shot LLM approach - asks AI coding assistants to generate chunking code.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import yaml
from litellm import completion

from src.evaluation.base import BasePreprocessor
from schema import Document, Chunk

# Import sibling modules
sys.path.insert(0, str(Path(__file__).parent))
from prompts import create_chunking_prompt
from code_executor import execute_chunking_code


class AIAssistantBaseline(BasePreprocessor):
    """
    Baseline using AI coding assistants via LiteLLM.
    Tests whether off-the-shelf LLMs can match custom iterative agents.
    """
    
    name = "ai_assistant_baseline"
    description = "One-shot LLM chunking code generation"
    
    def __init__(
        self,
        model: str = None,
        config_path: str = None,  # Will default to local config
        save_generated_code: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Default to config in same directory as this file
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Get model from config
        self.model = model or config.get('analysis_model')
        if not self.model:
            raise ValueError("No model specified! Set analysis_model in config.yaml")
        
        self.temperature = config.get('analysis_temperature', 1.0)
        self.max_tokens = config.get('max_tokens', 16000)
        self.api_base = config.get('api_base')
        
        if not self.api_base:
            raise ValueError("No api_base in config.yaml")
        
        self.save_generated_code = save_generated_code
        
        # Get API key
        self.api_key = os.getenv("LITELLM_API_KEY")
        if not self.api_key:
            raise ValueError("LITELLM_API_KEY not set in .env")
        
        print(f"🔧 Config loaded from: {config_path}")
        print(f"   Model: {self.model}")
        print(f"   Max tokens: {self.max_tokens}")
        print(f"   API base: {self.api_base}")
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM via LiteLLM proxy."""
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key,
                api_base=self.api_base,
            )
            
            finish_reason = response.choices[0].finish_reason
            print(f"🔧 Finish reason: {finish_reason}")
            
            if finish_reason == "length":
                print(f"⚠️  WARNING: Hit token limit! Increase max_tokens in config.yaml")
            
            content = response.choices[0].message.content
            print(f"🔧 Content length: {len(content) if content else 0}")
            
            return content or ""
            
        except Exception as e:
            print(f"❌ API Error: {e}")
            raise
    
    def preprocess(
        self,
        documents: List[Document],
        task_description: str = "Optimize document chunking for retrieval"
    ) -> List[Chunk]:
        """
        Use AI assistant to generate and apply chunking strategy.
        """
        print(f"🤖 AI Assistant Baseline: {self.model}")
        
        # Create prompt
        prompt = create_chunking_prompt(documents, task_description)
        print(f"📝 Prompt length: {len(prompt)} chars")
        
        # Get code from LLM
        print(f"📤 Querying LLM with max_tokens={self.max_tokens}...")
        response = self._call_llm(prompt)
        
        # Extract and save code
        code = self._extract_code(response)
        print(f"✅ Extracted code: {len(code)} chars")
        
        if len(code) == 0:
            print("❌ ERROR: Extracted code is empty!")
            raise ValueError("LLM returned empty code - increase max_tokens in config.yaml")
        
        if self.save_generated_code:
            self._save_code(code, task_description)
        
        # Execute chunking
        print(f"⚙️  Applying strategy to {len(documents)} documents...")
        chunks = execute_chunking_code(code, documents)
        
        avg_chunks = len(chunks) / len(documents) if documents else 0
        print(f"✅ Generated {len(chunks)} chunks ({avg_chunks:.1f} per doc)")
        
        return chunks
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from response with better handling."""
        if not response:
            return ""
        
        # Try markdown code blocks first
        if "```python" in response:
            parts = response.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0]
                return code.strip()
        
        if "```" in response:
            parts = response.split("```")
            if len(parts) >= 3:
                code = parts[1]
                return code.strip()
        
        # If no code blocks, assume entire response is code
        return response.strip()
    
    def _save_code(self, code: str, task: str):
        """Save generated code for inspection."""
        output_dir = Path("results") / "generated_strategies"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{self.model.replace('/', '_')}.py"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write(f"# Generated by {self.model}\n")
            f.write(f"# Task: {task}\n\n")
            f.write(code)
        
        print(f"💾 Saved strategy: {filepath}")
