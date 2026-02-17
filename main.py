"""
main.py – CLI entry point for LLM-driven preprocessing agents.

Usage:
    uv run python main.py --agent proposal_1 --loops 5
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an LLM-driven preprocessing agent."
    )
    parser.add_argument(
        "--agent",
        default="proposal_1",
        choices=["proposal_1"],
        help="Which agent to run (default: proposal_1)",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=3,
        help="Number of eval+improve loops (default: 3)",
    )
    args = parser.parse_args()

    if args.agent == "proposal_1":
        from src.agents.proposal_1.agent import Proposal1Agent
        agent = Proposal1Agent()
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    agent.run(n_loops=args.loops)


if __name__ == "__main__":
    main()
