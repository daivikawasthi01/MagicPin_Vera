#!/usr/bin/env python3
"""
Keepalive Pinger for Vera Bot.
Pings GET /v1/healthz every 5 minutes to prevent free instances (Render/Koyeb)
from going to sleep.
"""
import os
import time
import urllib.request
import datetime

BOT_URL = os.environ.get("BOT_URL", "http://localhost:8080").rstrip("/")
HEALTH_URL = f"{BOT_URL}/v1/healthz"
INTERVAL_SECONDS = int(os.environ.get("PING_INTERVAL", 300))  # 5 minutes default


def ping():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        req = urllib.request.Request(HEALTH_URL, headers={"User-Agent": "Vera-KeepAlive/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            status = resp.status
            data = resp.read().decode("utf-8")
            print(f"[{now}] Ping OK ({status}): {data[:80]}...")
    except Exception as exc:
        print(f"[{now}] Ping Failed / Cold Start: {exc}")


def main():
    print(f"Starting keepalive pinger for {HEALTH_URL} every {INTERVAL_SECONDS}s...")
    while True:
        ping()
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
