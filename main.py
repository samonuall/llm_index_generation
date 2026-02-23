"""
main.py – CLI entry point for LLM-driven preprocessing agents.

Usage:
    uv run python main.py --agent gemini_sdk --loops 5
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an LLM-driven preprocessing agent."
    )
    parser.add_argument(
        "--agent",
        default="gemini_sdk",
        choices=["gemini_sdk", "lite_llm_agent"],
        help="Which agent to run (default: gemini_sdk)",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=3,
        help="Number of eval+improve loops (default: 3)",
    )
    parser.add_argument(
        "--no-query-text",
        action="store_true",
        default=False,
        help="Omit raw query text from the prompt (use when Gemini safety filters block content)",
    )
    args = parser.parse_args()

    if args.agent == "gemini_sdk":
        from src.agents.gemini_sdk.agent import GeminiSdkAgent
        agent = GeminiSdkAgent(include_query_text=not args.no_query_text)
    elif args.agent == "lite_llm_agent":
        from src.agents.lite_llm_agent.agent import LiteLLMAgent
        agent = LiteLLMAgent(include_query_text=not args.no_query_text)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    agent.run(n_loops=args.loops)


if __name__ == "__main__":
    main()
