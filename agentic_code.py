import asyncio
import subprocess
import textwrap
import json
import re
import httpx
from dataclasses import dataclass, field
from typing import Callable
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AgentRequest(BaseModel):
    query: str

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
LMSTUDIO_MODEL = "google/gemma-3-4b"
MAX_ITERATIONS = 6

@dataclass
class AgentState:
    task: str
    history: list = field(default_factory=list)   # full message log
    code: str = ""                                  # latest generated code
    last_output: str = ""                           # stdout + stderr
    last_exit_code: int = 0
    iteration: int = 0
    done: bool = False
    final_artifact: str = ""
    log: list = field(default_factory=list)


# ── Tool: run Python code in a subprocess ──────────────────────────────────────

def execute_code(code: str, timeout: int = 15) -> tuple[str, int]:
    """Run `code` in a fresh Python subprocess. Returns (output, exit_code)."""
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True, text=True, timeout=timeout
        )
        output = result.stdout + result.stderr
        return output.strip() or "(no output)", result.returncode
    except subprocess.TimeoutExpired:
        return "ERROR: timed out after {timeout}s", 1
    except Exception as e:
        return f"ERROR: {e}", 1


def agent_print(state: AgentState, msg: str):
    print(msg)
    state.log.append(msg)

# ── LMStudio call ──────────────────────────────────────────────────────────────

async def call_gemma(
    client: httpx.AsyncClient,
    messages: list,
    max_tokens: int = 1024,
) -> str:
    resp = await client.post(
        LMSTUDIO_URL,
        json={
            "model": LMSTUDIO_MODEL,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "stream": False
        },
        timeout=httpx.Timeout(300.0),
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ── Prompt builders ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an autonomous coding agent. You operate in a loop:
    1. Think about what to do next.
    2. Write Python code to do it (inside ```python ... ``` fences).
    3. Observe the result of running that code.
    4. Repeat or declare done.

    Rules:
    - Always put runnable code inside ```python ... ``` fences.
    - If you are done and satisfied, output exactly: DONE: <one-line summary>
    - Keep each code block self-contained and runnable.
    - If the previous run had an error, fix it — do not repeat the same mistake.
""")

def plan_prompt(task: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Task: {task}\n\nPlan your approach, then write the first code block."},
    ]

def reflect_prompt(state: AgentState) -> list:
    """Build the full conversation so Gemma can reflect and decide next step."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += state.history   # all prior turns
    messages.append({
        "role": "user",
        "content": (
            f"Execution result (exit code {state.last_exit_code}):\n"
            f"```\n{state.last_output}\n```\n\n"
            f"Iteration {state.iteration}/{MAX_ITERATIONS}. "
            "If the task is complete, reply with DONE: <summary>. "
            "Otherwise, fix the issue and write the next code block."
        )
    })
    return messages


# ── Extract code from Gemma's reply ───────────────────────────────────────────

def extract_code(text: str) -> str | None:
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None

def is_done(text: str) -> tuple[bool, str]:
    if text.strip().startswith("DONE:"):
        return True, text.split("DONE:", 1)[1].strip()
    return False, ""


# ── Main agent loop ────────────────────────────────────────────────────────────

async def run_agent(task: str, on_step: Callable | None = None) -> AgentState:
    """
    Run the agentic loop for `task`.
    `on_step` is an optional callback(state) called after each iteration
    so you can stream progress to a UI.
    """
    state = AgentState(task=task)

    async with httpx.AsyncClient() as client:

        # ── PLAN ──────────────────────────────────────────────────────────────
        agent_print(state, f"\n[Agent] Task: {task}")
        reply = await call_gemma(client, plan_prompt(task))
        agent_print(state, f"[Agent] Plan reply:\n{reply}\n")
        state.history.append({
            "role": "user",
            "content": f"Task: {task}\n\nPlan your approach, then write the first code block."
        })
        state.history.append({
            "role": "assistant",
            "content": reply
        })

        # Check if Gemma somehow finished in one shot
        done, summary = is_done(reply)
        if done:
            state.done = True
            state.final_artifact = summary
            return state

        # ── LOOP: GENERATE → EXECUTE → REFLECT ────────────────────────────────
        while state.iteration < MAX_ITERATIONS and not state.done:
            state.iteration += 1
            agent_print(state, f"\n[Agent] Iteration {state.iteration}\n")

            # GENERATE: extract code from latest reply
            code = extract_code(reply)
            if not code:
                agent_print(state, f"[Agent] No code block found — asking Gemma to try again\n")
                state.history.append({
                    "role": "user",
                    "content": "Your last reply contained no ```python``` code block. Please write one."
                })
                reply = await call_gemma(client, state.history)
                state.history.append({"role": "assistant", "content": reply})
                continue

            state.code = code
            agent_print(state, f"[Agent] Running code:\n{code}\n")

            # EXECUTE
            output, exit_code = execute_code(code)
            state.last_output = output
            state.last_exit_code = exit_code
            agent_print(state, f"[Agent] Output (exit {exit_code}):\n{output}\n")

            if on_step:
                on_step(state)

            # REFLECT
            reflect_msgs = reflect_prompt(state)
            reply = await call_gemma(client, reflect_msgs)
            agent_print(state, f"[Agent] Reflect reply:\n{reply}\n")

            # Store the exchange in history
            state.history.append({
                "role": "user",
                "content": f"Output (exit {exit_code}):\n```\n{output}\n```"
            })
            state.history.append({"role": "assistant", "content": reply})

            # Check if done
            done, summary = is_done(reply)
            if done:
                state.done = True
                state.final_artifact = summary
                break

            # If error-free and no new code, we're probably stuck
            if exit_code == 0 and not extract_code(reply):
                state.done = True
                state.final_artifact = reply
                break

    if not state.done:
        agent_print(state, f"[Agent] Max iterations reached.")
        state.final_artifact = state.last_output  # best we have

    return state


# ── Drop-in usage example ──────────────────────────────────────────────────────

async def main():
    task = "Write a Python function that finds all prime numbers up to N using a sieve, then test it with N=50 and print the results."

    def show_step(state: AgentState):
        print(f"  → step {state.iteration}: exit={state.last_exit_code}")

    result = await run_agent(task, on_step=show_step)
    print("\n=== FINAL ===")
    print(f"Done: {result.done}")
    print(f"Summary: {result.final_artifact}")
    print(f"Last code:\n{result.code}")


@app.post("/run_agentic")
async def run_agentic(request: AgentRequest):
    result = await run_agent(request.query)
    return {
        "done": result.done,
        "summary": result.final_artifact,
        "last_code": result.code,
        "last_output": result.last_output,
        "full_log": "\n".join(result.log),
    }


if __name__ == "__main__":
    #asyncio.run(main())
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)