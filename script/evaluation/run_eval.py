"""Run tau2-bench evaluation against one or more sglang endpoints.

Examples:
  # Dense-only eval (both agent and user on port 30000)
  uv run python script/evaluation/run_eval.py \
      --domain airline \
      --agent-url http://127.0.0.1:30000/v1 \
      --user-url http://127.0.0.1:30000/v1 \
      --save-dir data/simulations/qwen32b_dense_airline

  # Sparse agent + dense user (recommended for long-context domains)
  uv run python script/evaluation/run_eval.py \
      --domain banking_knowledge \
      --agent-url http://127.0.0.1:30001/v1 \
      --user-url http://127.0.0.1:30000/v1 \
      --evaluation-type ALL_WITH_NL_ASSERTIONS \
      --save-dir data/simulations/qwen32b_sparseA_denseU_banking
"""

import argparse
from pathlib import Path

from tau2 import TextRunConfig
from tau2.evaluator.evaluator import EvaluationType
from tau2.runner import get_tasks, run_tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", required=True, help="Domain name (airline, retail, banking_knowledge, telecom)")
    parser.add_argument("--task-split", default=None, help="Task split name (e.g., 'small' for telecom)")
    parser.add_argument("--model", default="openai/Qwen/Qwen3-32B", help="Model identifier (litellm format)")
    parser.add_argument("--agent-url", required=True, help="OpenAI-compatible endpoint for the agent")
    parser.add_argument("--user-url", required=True, help="OpenAI-compatible endpoint for the user simulator")
    parser.add_argument("--api-key", default="sk-local", help="API key (use 'sk-local' for local sglang)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--save-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--max-context-length", type=int, default=32000)
    parser.add_argument("--evaluation-type", default="ALL", choices=["ALL", "ALL_WITH_NL_ASSERTIONS"])
    parser.add_argument("--task-ids", nargs="*", default=None, help="Subset of task IDs to run")
    args = parser.parse_args()

    agent_args = {"api_base": args.agent_url, "api_key": args.api_key, "temperature": args.temperature}
    user_args = {"api_base": args.user_url, "api_key": args.api_key, "temperature": args.temperature}

    config = TextRunConfig(
        domain=args.domain,
        agent="llm_agent",
        user="user_simulator",
        llm_agent=args.model,
        llm_args_agent=agent_args,
        llm_user=args.model,
        llm_args_user=user_args,
        num_trials=args.num_trials,
        max_concurrency=args.max_concurrency,
        max_context_length=args.max_context_length,
    )
    tasks = get_tasks(args.domain, task_split_name=args.task_split, task_ids=args.task_ids)
    print(f"Running {len(tasks)} tasks from {args.domain}")

    results = run_tasks(
        config, tasks,
        save_path=args.save_dir / "results.json",
        save_dir=args.save_dir,
        evaluation_type=EvaluationType[args.evaluation_type],
    )
    n = len(results.simulations)
    avg_r = sum(s.reward_info.reward for s in results.simulations if s.reward_info) / n if n > 0 else 0
    print(f"Done: {n} sims, avg reward={avg_r:.2f}")


if __name__ == "__main__":
    main()
