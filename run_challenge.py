#!/usr/bin/env python3
"""Single-command bootstrap for the Magicpin challenge.

Starts the bot, waits for it to become healthy, and then runs the judge.
It normalizes the provider configuration to Gemini first so stale Groq model
values do not poison the runtime in a fresh shell or deployment job.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent
BOT_URL = "http://localhost:8080"
HEALTH_URL = f"{BOT_URL}/v1/healthz"


def normalize_runtime_env() -> dict:
    env = os.environ.copy()

    # Force deterministic Gemini-first runtime values for both the bot and judge.
    env["BOT_URL"] = BOT_URL
    env["LLM_PROVIDER"] = "gemini"
    env["LLM_MODEL"] = "gemini-3.5-flash-lite"
    env["GEMINI_MODEL"] = "gemini-3.5-flash-lite"
    env["TEST_SCENARIO"] = "full_pipeline"
    env["GEMINI_API_KEY"] = env.get("GEMINI_API_KEY") or env.get("GEMINI_API_KEY_1", "")

    # Keep the existing Groq config in place for fallback scenarios
    if not env.get("GROQ_MODEL"):
        env["GROQ_MODEL"] = "openai/gpt-oss-120b"

    return env


def stop_existing_bot_processes() -> None:
    """Best-effort cleanup for stale bot processes from previous runs."""
    commands = [
        ["pkill", "-f", "python bot.py"],
        ["pkill", "-f", "uvicorn bot:app --host 127.0.0.1 --port 8080"],
    ]
    for cmd in commands:
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except FileNotFoundError:
            pass


def wait_for_bot(timeout_seconds: int = 60) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            req = urllib.request.Request(HEALTH_URL, method="GET")
            with urllib.request.urlopen(req, timeout=3) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def run_bot(env: dict) -> subprocess.Popen:
    bot_log = ROOT / ".bot.log"
    with open(bot_log, "ab") as log_handle:
        proc = subprocess.Popen(
            [sys.executable, "bot.py"],
            cwd=str(ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return proc


def main() -> int:
    stop_existing_bot_processes()
    env = normalize_runtime_env()
    os.environ.update(env)

    print("[bootstrap] Starting bot...")
    bot_proc = run_bot(env)
    try:
        if not wait_for_bot():
            print("[bootstrap] Bot did not become healthy within 60s.", file=sys.stderr)
            return 1

        print("[bootstrap] Bot is healthy. Running full judge_simulator.main()...")
        import importlib
        judge_module = importlib.import_module("judge_simulator")
        return int(judge_module.main())
    finally:
        try:
            bot_proc.terminate()
            bot_proc.wait(timeout=10)
        except Exception:
            try:
                bot_proc.kill()
                bot_proc.wait(timeout=5)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
