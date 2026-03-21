"""
main.py – CLI entry point for LLM-driven preprocessing agents.

Usage:
    uv run python main.py --agent lite_llm_agent --loops 5 --split paper_retrieval_5000docs
    uv run python main.py --agent baseline --split paper_retrieval_5000docs
    uv run python main.py --agent ai_assistant --split paper_retrieval_5000docs --loops 3
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an LLM-driven preprocessing agent."
    )
    parser.add_argument(
        "--agent",
        default=None,
        choices=[
            "gemini_sdk",
            "lite_llm_agent",
            "test_agent",
            "analysis_code_agent",
            "baseline",
            "ai_assistant",
        ],
        help="Which agent to run (default: gemini_sdk)",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=3,
        help="Number of eval+improve loops (default: 3)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="tip_of_the_tongue_5000docs",
        help="CRUMB split name (default: tip_of_the_tongue_5000docs)",
    )
    parser.add_argument(
        "--no-query-text",
        action="store_true",
        default=False,
        help="Omit raw query text from the prompt (use when Gemini safety filters block content)",
    )
    parser.add_argument(
        "--enable_tracing",
        action="store_true",
        default=False,
        help="Enable tracing of LLM calls (only applies to lite_llm_agent)",
    )
    args = parser.parse_args()

    if args.agent == "gemini_sdk":
        from src.agents import GeminiSdkAgent
        agent = GeminiSdkAgent(include_query_text=not args.no_query_text)
    
    elif args.agent == "lite_llm_agent":
        from src.agents import LiteLLMAgent
        if args.enable_tracing:
            import mlflow
            mlflow.litellm.autolog()
        agent = LiteLLMAgent(include_query_text=not args.no_query_text)
    
    elif args.agent == "test_agent":
        from src.agents import LiteLLMAgent
        agent = LiteLLMAgent(include_query_text=not args.no_query_text, test_mode=True)
    
    elif args.agent == "analysis_code_agent":
        from src.agents.analysis_code_agent import AnalysisCodeAgent
        agent = AnalysisCodeAgent()
    
    elif args.agent == "baseline":
        # Import your baseline agent - adjust the import path as needed
        try:
            from src.agents.baseline import BaselineAgent
            agent = BaselineAgent()
        except ImportError:
            try:
                from src.preprocessing.baseline import BaselinePreprocessor
                # If it's a preprocessor, wrap it or use directly
                agent = BaselinePreprocessor()
            except ImportError:
                raise ImportError(
                    "Could not find baseline agent. Check if the file exists in "
                    "src/agents/baseline.py or src/preprocessing/baseline.py"
                )
    
    elif args.agent == "ai_assistant":
        # Import your AI assistant agent - adjust the import path as needed
        try:
            from src.agents.ai_assistant import AIAssistantAgent
            agent = AIAssistantAgent()
        except ImportError:
            try:
                from src.agents.ai_assistant_agent import AIAssistantAgent
                agent = AIAssistantAgent()
            except ImportError:
                raise ImportError(
                    "Could not find ai_assistant agent. Check if the file exists in "
                    "src/agents/ai_assistant.py or src/agents/ai_assistant_agent.py"
                )
    
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # Set the split on the agent
    agent.split = args.split
    agent.run(n_loops=args.loops)


if __name__ == "__main__":
    main()