"""
main.py – CLI entry point for LLM-driven preprocessing agents.

Usage:
    uv run python main.py --agent gemini_sdk --loops 5
    uv run python main.py --agent lite_llm_agent --loops 5 --split paper_retrieval
    uv run python main.py --agent baseline --split paper_retrieval
    uv run python main.py --agent ai_assistant --split paper_retrieval --loops 3

    # Analysis agent ablation conditions:
    uv run python main.py --agent analysis_code_agent --loops 3 --condition agent
    uv run python main.py --agent analysis_code_agent --loops 3 --condition agent_history
    uv run python main.py --agent analysis_code_agent --loops 3 --condition agent_contrastive

    # One-shot LLM baseline (no loops):
    uv run python main.py --agent one_shot
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
            "one_shot",
            "baseline",
            "ai_assistant",
        ],
        help="Which agent to run (default: gemini_sdk)",
    )
    parser.add_argument(
        "--condition",
        default="agent_contrastive",
        choices=["agent", "agent_history", "agent_contrastive", "agent_contrastive_no_history", "agent_noinput"],
        help="Ablation condition for analysis_code_agent (default: agent_contrastive)",
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
        default="tip_of_the_tongue",
        help="CRUMB split name (default: tip_of_the_tongue)",
    )
    parser.add_argument(
        "--no-query-text",
        action="store_true",
        default=False,
        help="Omit raw query text from the prompt (use when safety filters block content)",
    )
    parser.add_argument(
        "--task",
        default="sustainable_living",
        help="BRIGHT task/subset (used with gemini_sdk_bright / lite_llm_bright, default: sustainable_living)",
    )
    parser.add_argument(
        "--enable_tracing",
        action="store_true",
        default=False,
        help="Enable MLflow tracing of LLM calls (only applies to lite_llm_agent)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override LLM model for analysis_code_agent and one_shot (e.g. gemini/gemini-2.5-pro)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        dest="api_base",
        help="Override API base URL (omit to use provider default, e.g. for native Gemini API)",
    )
    parser.add_argument(
        "--max-distractors",
        type=int,
        default=9000,
        help="Max non-relevant docs to sample"
    )

    args = parser.parse_args()

    # Reset analysis_code_agent/preprocess.py to clean baseline at the start of every run.
    if args.agent in ("analysis_code_agent", "one_shot"):
        import shutil, pathlib
        repo_root = pathlib.Path(__file__).parent
        clean = repo_root / "src" / "agents" / "baseline" / "preprocess.py"
        target = repo_root / "src" / "agents" / "analysis_code_agent" / "preprocess.py"
        shutil.copy(clean, target)
        print(f"[main] Reset preprocess.py from clean baseline: {clean}")


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
        use_history = args.condition in ("agent_history", "agent_contrastive")
        use_contrastive = args.condition in ("agent_contrastive", "agent_contrastive_no_history")
        use_analysis = args.condition != "agent_noinput"
        agent = AnalysisCodeAgent(
            use_history=use_history,
            use_contrastive=use_contrastive,
            use_analysis=use_analysis,
            model=args.model,
            api_base=args.api_base,
        )
    
    elif args.agent == "one_shot":
        from src.agents.analysis_code_agent.one_shot_agent import run_one_shot
        run_one_shot(split=args.split, model=args.model, api_base=args.api_base, max_distractors=args.max_distractors)
        return

    elif args.agent == "baseline":
        try:
            from src.agents.baseline import BaselineAgent
            agent = BaselineAgent()
        except ImportError:
            try:
                from src.preprocessing.baseline import BaselinePreprocessor
                agent = BaselinePreprocessor()
            except ImportError:
                raise ImportError(
                    "Could not find baseline agent. Check if the file exists in "
                    "src/agents/baseline.py or src/preprocessing/baseline.py"
                )

    elif args.agent == "ai_assistant":
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

    agent.split = args.split
    agent.run(n_loops=args.loops)


if __name__ == "__main__":
    main()