"""Analyze tau2-bench trajectory: timing, token usage, and prefill/decode distributions.

Separates agent (assistant) LLM calls from user simulator LLM calls.
"""

import argparse
from pathlib import Path

import numpy as np

from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.data_model.simulation import Results
from tau2.metrics.agent_metrics import (
    compute_metrics,
    display_metrics,
    get_trajectory_token_stats,
)


def fmt_ts(ts: str | None) -> str:
    if ts is None:
        return "N/A"
    return ts.split("T")[1][:12]


def print_role_stats(label: str, msgs: list):
    """Print timing and token stats for a list of messages (agent or user)."""
    with_usage = [
        m for m in msgs
        if m.usage is not None and m.generation_time_seconds is not None
    ]
    if not with_usage:
        print(f"  {label}: no LLM calls with usage data")
        return

    prefills = np.array([m.usage["prompt_tokens"] for m in with_usage])
    decodes = np.array([m.usage["completion_tokens"] for m in with_usage])
    gen_times = np.array([m.generation_time_seconds for m in with_usage])

    print(f"  {label} ({len(with_usage)} requests):")
    header = f"    {'Metric':>20}  {'Avg':>8}  {'Median':>8}  {'Min':>8}  {'Max':>8}  {'P90':>8}  {'Total':>10}"
    print(header)
    print(f"    {'--------------------':>20}  {'--------':>8}  {'--------':>8}  {'--------':>8}  {'--------':>8}  {'--------':>8}  {'----------':>10}")
    for name, arr in [("Prefill tokens", prefills), ("Decode tokens", decodes), ("Generation time (s)", gen_times)]:
        fmt = ".0f" if "token" in name else ".2f"
        print(f"    {name:>20}  {arr.mean():{fmt}}  {np.median(arr):{fmt}}  {arr.min():{fmt}}  {arr.max():{fmt}}  {np.percentile(arr, 90):{fmt}}  {arr.sum():{fmt}}")

    # Prefill growth
    print(f"    Prefill growth:")
    for i, m in enumerate(with_usage):
        p = m.usage["prompt_tokens"]
        if i > 0:
            delta = p - with_usage[i - 1].usage["prompt_tokens"]
            sign = "+" if delta >= 0 else ""
            print(f"      Req {i}: {p:>6} tokens  ({sign}{delta})")
        else:
            print(f"      Req {i}: {p:>6} tokens")
    print()


def analyze(results_path: Path, output_path: Path | None = None):
    results = Results.load(results_path)

    lines: list[str] = []

    import sys
    import io

    # Capture all print output so we can both display and save
    old_stdout = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf

    for sim in results.simulations:
        print("=" * 80)
        print(f"Simulation: {sim.id}")
        print(f"  Task ID:            {sim.task_id}")
        print(f"  Termination:        {sim.termination_reason}")
        print(f"  Reward:             {sim.reward_info.reward if sim.reward_info else None}")
        print(f"  Duration:           {sim.duration:.2f}s")

        messages = sim.get_messages()
        assistant_msgs = [m for m in messages if isinstance(m, AssistantMessage)]
        user_msgs = [m for m in messages if isinstance(m, UserMessage)]
        other_count = len(messages) - len(assistant_msgs) - len(user_msgs)
        print(f"  Messages:           {len(messages)} total ({len(assistant_msgs)} assistant, {len(user_msgs)} user, {other_count} tool)")

        # --- Per-message table ---
        print()
        print("  Per-Message Breakdown:")
        print(f"  {'Turn':>4}  {'Role':>10}  {'Start':>14}  {'End':>14}  {'GenTime':>8}  {'Prefill':>8}  {'Decode':>8}")
        print(f"  {'----':>4}  {'----------':>10}  {'--------------':>14}  {'--------------':>14}  {'--------':>8}  {'--------':>8}  {'--------':>8}")
        for msg in messages:
            turn = msg.turn_idx if msg.turn_idx is not None else ""
            role = msg.role
            start = fmt_ts(getattr(msg, "start_time", None))
            end = fmt_ts(getattr(msg, "end_time", None))
            gen = f"{msg.generation_time_seconds:.2f}s" if getattr(msg, "generation_time_seconds", None) else "N/A"
            usage = getattr(msg, "usage", None)
            prefill = usage.get("prompt_tokens", "") if usage else ""
            decode = usage.get("completion_tokens", "") if usage else ""
            print(f"  {turn:>4}  {role:>10}  {start:>14}  {end:>14}  {gen:>8}  {prefill:>8}  {decode:>8}")

        # --- Separated stats ---
        print()
        print_role_stats("Assistant (Agent) LLM", assistant_msgs)
        print_role_stats("User (Simulator) LLM", user_msgs)

        # --- Token totals ---
        ts = get_trajectory_token_stats(sim)
        print(f"  Agent Token Totals: prefill={ts.total_prefill}, decode={ts.total_decode}, requests={ts.num_requests}")

    # --- Aggregate metrics ---
    print()
    print("=" * 80)
    print("Aggregate Metrics:")
    metrics = compute_metrics(results)
    display_metrics(metrics)

    # Restore stdout and get output
    sys.stdout = old_stdout
    output = buf.getvalue()
    print(output, end="")

    if output_path:
        output_path.write_text(output)
        print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze tau2-bench trajectories")
    parser.add_argument("results", type=str, help="Path to results.json")
    parser.add_argument("-o", "--output", type=str, default=None, help="Save analysis to file")
    args = parser.parse_args()
    out = Path(args.output) if args.output else None
    analyze(Path(args.results), out)
