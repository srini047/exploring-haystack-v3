"""
Linux Troubleshooting Agent with layered Human-in-the-Loop hooks
==================================================================

You describe a problem ("CPU is at 80%", "memory is full", etc.) and the
agent investigates by running real Linux shell commands via a tool. How
much control you keep over that execution is decided once, up front, by
picking a trust mode:

    [y] yes   -> confirm every command before it runs   (before_tool hook)
    [n] no    -> trust the agent fully, no prompts at all
    [a] auto  -> most cautious: confirm before every agent "step",
                 i.e. before the LLM is even called again  (before_llm hook)

This deliberately uses three different Haystack hook points so you can see
how they compose:

    before_llm  -> gate whether the agent is allowed to think/plan again
    before_tool -> gate whether a specific shell command is allowed to run
    after_run   -> summarize what happened once the agent finishes
"""

import subprocess

from haystack.components.agents import Agent
from haystack.components.agents.state import State
from haystack.dataclasses import ChatMessage
from haystack.hooks import hook
from haystack.hooks.human_in_the_loop import (
    AlwaysAskPolicy,
    NeverAskPolicy,
    BlockingConfirmationStrategy,
    ConfirmationHook,
    SimpleConsoleUI,
)
from haystack.tools import tool

from utils import cohere_llm_generator


@tool
def run_linux_command(
    command: str,
) -> str:
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        output = result.stdout.strip()
        if result.stderr.strip():
            output += f"\n[stderr]\n{result.stderr.strip()}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out after 30s."
    except Exception as exc:
        return f"Error running command: {exc}"


def make_before_llm_hook(ask_every_step: bool):
    step_counter = {"n": 0}

    @hook
    def before_llm(state: State) -> None:
        step_counter["n"] += 1
        if not ask_every_step:
            return
        answer = (
            input(
                f"\n[before_llm] step {step_counter['n']}: agent wants to think/plan "
                f"its next move. Continue? (y/n): "
            )
            .strip()
            .lower()
        )
        if answer != "y":
            raise RuntimeError("Stopped by user before an LLM step.")

    return before_llm


@hook
def log_before_tool(state: State) -> None:
    last_message = state.get("messages")[-1]
    if last_message.tool_calls:
        for call in last_message.tool_calls:
            print(f"[guardrail] about to call '{call.tool_name}' with {call.arguments}")


@hook
def after_run(state: State) -> None:
    messages = state.get("messages")
    tool_calls = sum(len(m.tool_calls or []) for m in messages if m.tool_calls)
    print(f"\n[after_run] session finished — {tool_calls} command(s) executed.")


def build_agent(mode: str) -> Agent:
    """
    mode:
      "manual" (y) -> confirm every command before execution (before_tool)
      "auto"   (n) -> trust the agent fully, never ask
      "strict" (a) -> confirm before every single agent step (before_llm),
                      on top of confirming every command too
    """
    tool_policy = NeverAskPolicy() if mode == "auto" else AlwaysAskPolicy()
    ask_every_step = mode == "strict"

    confirmation_strategy = BlockingConfirmationStrategy(
        confirmation_policy=tool_policy,
        confirmation_ui=SimpleConsoleUI(),
    )
    confirmation_hook = ConfirmationHook(
        confirmation_strategies={"*": confirmation_strategy},
    )

    return Agent(
        chat_generator=cohere_llm_generator(),
        tools=[run_linux_command],
        system_prompt=(
            "You are a Linux systems troubleshooting agent. Given a problem "
            "description (e.g. high CPU, high memory, disk full, slow network), "
            "diagnose it step by step using the run_linux_command tool. Always "
            "run read-only diagnostic commands first (top, ps, free, df, vmstat, "
            "iostat, journalctl, dmesg, ss/netstat) to gather evidence before "
            "proposing or running any destructive/fix command (kill, systemctl "
            "restart, rm, etc). Explain what each command's output tells you "
            "before moving to the next step, and clearly state your final "
            "diagnosis and recommended fix at the end."
        ),
        hooks={
            "before_llm": [make_before_llm_hook(ask_every_step)],
            "before_tool": [log_before_tool, confirmation_hook],
            "after_run": [after_run],
        },
    )


def select_mode() -> str:
    print("How much control do you want over the agent's commands?")
    print("  [y] yes  - confirm every command before it runs")
    print("  [n] no   - trust the agent fully, run everything automatically")
    print("  [a] auto - most granlar: confirmation required for every agent step")

    choice = input("Mode (y/n/a): ").strip().lower()

    return {"y": "manual", "n": "auto", "a": "strict"}.get(choice, "manual")


def main() -> None:
    mode = select_mode()
    agent = build_agent(mode)

    problem = input("\nDescribe the linux problem to troubleshoot: ").strip()
    result = agent.run(messages=[ChatMessage.from_user(problem)])

    print("\n--- Agent report ---")
    for msg in result["messages"]:
        if msg._role == "user":
            print("[USER]", end=" ")
        elif msg._role == "system":
            print("[SYSTEM]", end=" ")
        elif msg._role == "assistant":
            print("[ASSISTANT]", end=" ")
        elif msg._role == "tool":
            print("[TOOL]", end="")

        if msg.text:
            print(f"\n{msg.text}")


if __name__ == "__main__":
    main()
