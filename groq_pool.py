"""
Groq API Key Pool & Instant Rate-Limit Rotator.
Manages separate, isolated key pools for the Bot (Composer) and Judge Simulator,
eliminating free-tier rate limits (20-30 RPM) via instant failover upon HTTP 429.
"""
import os
import time
import json
import asyncio
import logging
import threading
from typing import List, Dict, Any, Optional, Callable
import httpx

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("groq_pool")


class GroqKeyPool:
    """
    Thread-safe and async-safe Groq API Key Pool with instant rate-limit failover.
    Can be scoped to 'bot', 'judge', or 'all' to ensure the Bot and Judge never collide
    on the same quota budget.
    """
    def __init__(self, role: str = "bot"):
        self.role = role.lower().strip()
        self._lock = threading.Lock()
        self._keys: List[str] = []
        self._status: Dict[str, Dict[str, Any]] = {}
        self._current_index: int = 0
        self._invalid_retry_seconds = 900.0
        self.reload_keys()

    def reload_keys(self) -> List[str]:
        """Discovers and assigns Groq API keys based on the pool's designated role."""
        with self._lock:
            bot_keys: List[str] = []
            judge_keys: List[str] = []

            # 1. Check explicit comma-separated lists
            if os.environ.get("BOT_GROQ_KEYS"):
                for k in os.environ["BOT_GROQ_KEYS"].replace(",", "\n").splitlines():
                    k = k.strip()
                    if k and k.startswith("gsk_") and k not in bot_keys:
                        bot_keys.append(k)

            if os.environ.get("JUDGE_GROQ_KEYS"):
                for k in os.environ["JUDGE_GROQ_KEYS"].replace(",", "\n").splitlines():
                    k = k.strip()
                    if k and k.startswith("gsk_") and k not in judge_keys:
                        judge_keys.append(k)

            # 2. Gather numbered keys 1..20
            all_numbered: List[str] = []
            primary = (os.environ.get("GROQ_API_KEY") or "").strip()
            if primary and primary.startswith("gsk_") and primary not in all_numbered:
                all_numbered.append(primary)

            for i in range(1, 21):
                k = (os.environ.get(f"GROQ_API_KEY_{i}") or "").strip()
                if k and k.startswith("gsk_") and k not in all_numbered:
                    all_numbered.append(k)

            # If explicit lists weren't provided, partition all discovered keys 50/50
            if not bot_keys and not judge_keys:
                mid = max(1, len(all_numbered) // 2)
                bot_keys = all_numbered[:mid]
                judge_keys = all_numbered[mid:]
            else:
                # If explicit keys exist, fill any unassigned numbered keys
                for i, k in enumerate(all_numbered, 1):
                    if i <= 4:
                        if k not in bot_keys and k not in judge_keys:
                            bot_keys.append(k)
                    else:
                        if k not in judge_keys and k not in bot_keys:
                            judge_keys.append(k)

            # Assign keys based on role
            if self.role == "judge":
                self._keys = judge_keys
            elif self.role == "all":
                self._keys = bot_keys + [k for k in judge_keys if k not in bot_keys]
            else:
                # Default to bot
                self._keys = bot_keys

            for k in self._keys:
                if k not in self._status:
                    self._status[k] = {
                        "rate_limited_until": 0.0,
                        "invalid": False,
                        "invalid_until": 0.0,
                        "successes": 0,
                        "rate_limits": 0
                    }
            if self._current_index >= len(self._keys):
                self._current_index = 0
            return list(self._keys)

    @property
    def total_keys(self) -> int:
        return len(self._keys)

    def get_candidate_keys(self) -> List[str]:
        """
        Returns an ordered list of keys starting from the current active key,
        filtering out invalid keys and prioritizing non-rate-limited keys.
        """
        with self._lock:
            if not self._keys:
                return []
            now = time.time()
            n = len(self._keys)

            # Re-activate keys whose invalid window has expired.
            for k in self._keys:
                status = self._status.get(k, {})
                invalid_until = status.get("invalid_until", 0.0)
                if status.get("invalid", False) and invalid_until and now >= invalid_until:
                    status["invalid"] = False
                    status["invalid_until"] = 0.0

            ordered = [self._keys[(self._current_index + i) % n] for i in range(n)]
            valid = [k for k in ordered if not self._status.get(k, {}).get("invalid", False)]

            active = [k for k in valid if self._status.get(k, {}).get("rate_limited_until", 0.0) <= now]
            cooling = [k for k in valid if self._status.get(k, {}).get("rate_limited_until", 0.0) > now]
            cooling.sort(key=lambda k: self._status.get(k, {}).get("rate_limited_until", 0.0))

            return active + cooling

    def mark_rate_limited(self, key: str, cooldown_seconds: float = 60.0):
        """Marks a key as rate-limited and advances to the next key instantly."""
        with self._lock:
            if key in self._status:
                self._status[key]["rate_limited_until"] = time.time() + cooldown_seconds
                self._status[key]["rate_limits"] += 1
            if self._keys:
                try:
                    idx = self._keys.index(key)
                    self._current_index = (idx + 1) % len(self._keys)
                except ValueError:
                    pass

    def mark_invalid(self, key: str, retry_after_seconds: Optional[float] = None):
        """Marks a key as temporarily invalid and re-allows it after a cooldown window."""
        with self._lock:
            if key in self._status:
                ttl = retry_after_seconds if retry_after_seconds is not None else self._invalid_retry_seconds
                self._status[key]["invalid"] = True
                self._status[key]["invalid_until"] = time.time() + ttl
            if self._keys:
                try:
                    idx = self._keys.index(key)
                    self._current_index = (idx + 1) % len(self._keys)
                except ValueError:
                    pass

    def mark_success(self, key: str):
        """Marks successful response and updates current index."""
        with self._lock:
            if key in self._status:
                self._status[key]["rate_limited_until"] = 0.0
                self._status[key]["successes"] += 1
            if self._keys:
                try:
                    self._current_index = self._keys.index(key)
                except ValueError:
                    pass

    def get_preferred_models(self, preferred_model: Optional[str] = None) -> List[str]:
        """Returns ordered list of Groq models to try."""
        models: List[str] = []
        if preferred_model:
            models.append(preferred_model)
        
        env_model = os.environ.get("GROQ_MODEL") or os.environ.get("LLM_MODEL")
        if env_model and env_model not in models:
            models.append(env_model)
            
        defaults = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
        ]
        for d in defaults:
            if d not in models:
                models.append(d)
        return models

    async def execute_chat_async(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 150,
        timeout: float = 6.0,
        on_success: Optional[Callable[[str], None]] = None,
    ) -> Optional[str]:
        """
        Asynchronously executes chat completion with instant key failover on 429.
        Retries respect active cooldown windows so we do not hammer the same
        rate-limited keys within the 60s free-tier window that triggered the burst
        failure pattern in the real judge run.
        """
        retries = 2 if self.role == "bot" else 1
        for attempt in range(retries + 1):
            candidate_keys = self.get_candidate_keys()
            if not candidate_keys:
                logger.warning("%s pool exhausted: no candidate keys available", self.role)
                return None

            active_candidates = [k for k in candidate_keys if self._status.get(k, {}).get("rate_limited_until", 0.0) <= time.time()]
            if not active_candidates and attempt < retries:
                next_ready = min(
                    self._status.get(k, {}).get("rate_limited_until", time.time())
                    for k in candidate_keys
                )
                wait_for = max(3.0, next_ready - time.time() + 0.25)
                logger.warning("%s pool cooling; waiting %.2fs for the next key to recover", self.role, wait_for)
                await asyncio.sleep(wait_for)
                continue

            models_to_try = self.get_preferred_models(model)
            headers_template = {
                "Content-Type": "application/json",
                "User-Agent": f"magicpin-{self.role}/1.0"
            }

            for key in candidate_keys:
                status_entry = self._status.get(key, {})
                if status_entry.get("invalid", False):
                    continue

                for groq_model in models_to_try:
                    try:
                        headers = {**headers_template, "Authorization": f"Bearer {key}"}
                        payload = {
                            "model": groq_model,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens
                        }

                        async with httpx.AsyncClient(timeout=timeout) as client:
                            start = time.time()
                            res = await client.post(
                                "https://api.groq.com/openai/v1/chat/completions",
                                headers=headers,
                                json=payload
                            )
                            latency = time.time() - start

                        if res.status_code == 200:
                            data = res.json()
                            choices = data.get("choices", [])
                            if choices:
                                content = choices[0].get("message", {}).get("content", "").strip()
                                if content.startswith('"') and content.endswith('"'):
                                    content = content[1:-1].strip()
                                if content:
                                    self.mark_success(key)
                                    logger.info("%s async success via %s/%s in %.2fs", self.role, key[:8], groq_model, latency)
                                    if on_success:
                                        on_success(groq_model)
                                    return content
                            logger.warning("%s async empty response via %s/%s in %.2fs", self.role, key[:8], groq_model, latency)
                            continue

                        elif res.status_code == 429:
                            logger.warning("%s async 429 via %s/%s in %.2fs", self.role, key[:8], groq_model, latency)
                            self.mark_rate_limited(key, cooldown_seconds=60.0)
                            if groq_model != models_to_try[-1]:
                                continue
                            break

                        elif res.status_code == 401:
                            logger.warning("%s async 401 via %s/%s in %.2fs; key temporarily invalidated", self.role, key[:8], groq_model, latency)
                            self.mark_invalid(key)
                            break

                        elif res.status_code == 404:
                            logger.warning("%s async 404 via %s/%s; trying next model", self.role, key[:8], groq_model)
                            continue

                        else:
                            logger.warning("%s async unexpected %s via %s/%s", self.role, res.status_code, key[:8], groq_model)
                            continue

                    except (httpx.TimeoutException, httpx.NetworkError) as exc:
                        logger.warning("%s async network timeout/error via %s/%s: %s", self.role, key[:8], groq_model, exc)
                        break
                    except Exception as exc:
                        logger.exception("%s async unhandled exception via %s/%s", self.role, key[:8], groq_model)
                        break

            if attempt < retries:
                next_ready = min(
                    self._status.get(k, {}).get("rate_limited_until", 0.0)
                    for k in candidate_keys
                    if self._status.get(k, {}).get("rate_limited_until", 0.0) > time.time()
                ) if any(self._status.get(k, {}).get("rate_limited_until", 0.0) > time.time() for k in candidate_keys) else 0.0
                backoff = max(3.0, next_ready - time.time() + 0.25) if next_ready else 3.0
                logger.warning("%s pool retrying after failure window (attempt %s/%s) in %.2fs", self.role, attempt + 1, retries, backoff)
                await asyncio.sleep(backoff)

        return None

    def execute_chat_sync(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 150,
        timeout: float = 6.0,
        on_success: Optional[Callable[[str], None]] = None,
    ) -> Optional[str]:
        """
        Synchronously executes chat completion with instant key failover on 429.
        Retries respect the cooldown clock instead of hammering the same rate-limited
        keys immediately after a burst of 429s, which is the failure mode observed
        in the judge's full run.
        """
        retries = 2 if self.role == "bot" else 1
        for attempt in range(retries + 1):
            candidate_keys = self.get_candidate_keys()
            if not candidate_keys:
                logger.warning("%s pool exhausted: no candidate keys available", self.role)
                return None

            active_candidates = [k for k in candidate_keys if self._status.get(k, {}).get("rate_limited_until", 0.0) <= time.time()]
            if not active_candidates and attempt < retries:
                next_ready = min(
                    self._status.get(k, {}).get("rate_limited_until", time.time())
                    for k in candidate_keys
                )
                wait_for = max(3.0, next_ready - time.time() + 0.25)
                logger.warning("%s pool cooling; waiting %.2fs for the next key to recover", self.role, wait_for)
                time.sleep(wait_for)
                continue

            models_to_try = self.get_preferred_models(model)
            headers_template = {
                "Content-Type": "application/json",
                "User-Agent": f"magicpin-{self.role}/1.0"
            }

            for key in candidate_keys:
                status_entry = self._status.get(key, {})
                if status_entry.get("invalid", False):
                    continue

                for groq_model in models_to_try:
                    try:
                        headers = {**headers_template, "Authorization": f"Bearer {key}"}
                        payload = {
                            "model": groq_model,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens
                        }

                        start = time.time()
                        with httpx.Client(timeout=timeout) as client:
                            res = client.post(
                                "https://api.groq.com/openai/v1/chat/completions",
                                headers=headers,
                                json=payload
                            )
                        latency = time.time() - start

                        if res.status_code == 200:
                            data = res.json()
                            choices = data.get("choices", [])
                            if choices:
                                content = choices[0].get("message", {}).get("content", "").strip()
                                if content.startswith('"') and content.endswith('"'):
                                    content = content[1:-1].strip()
                                if content:
                                    self.mark_success(key)
                                    logger.info("%s sync success via %s/%s in %.2fs", self.role, key[:8], groq_model, latency)
                                    if on_success:
                                        on_success(groq_model)
                                    return content
                            logger.warning("%s sync empty response via %s/%s in %.2fs", self.role, key[:8], groq_model, latency)
                            continue

                        elif res.status_code == 429:
                            logger.warning("%s sync 429 via %s/%s in %.2fs", self.role, key[:8], groq_model, latency)
                            self.mark_rate_limited(key, cooldown_seconds=60.0)
                            if groq_model != models_to_try[-1]:
                                continue
                            break

                        elif res.status_code == 401:
                            logger.warning("%s sync 401 via %s/%s in %.2fs; key temporarily invalidated", self.role, key[:8], groq_model, latency)
                            self.mark_invalid(key)
                            break

                        elif res.status_code == 404:
                            logger.warning("%s sync 404 via %s/%s; trying next model", self.role, key[:8], groq_model)
                            continue

                        else:
                            logger.warning("%s sync unexpected %s via %s/%s", self.role, res.status_code, key[:8], groq_model)
                            continue

                    except (httpx.TimeoutException, httpx.NetworkError) as exc:
                        logger.warning("%s sync network timeout/error via %s/%s: %s", self.role, key[:8], groq_model, exc)
                        break
                    except Exception as exc:
                        logger.exception("%s sync unhandled exception via %s/%s", self.role, key[:8], groq_model)
                        break

            if attempt < retries:
                next_ready = min(
                    self._status.get(k, {}).get("rate_limited_until", 0.0)
                    for k in candidate_keys
                    if self._status.get(k, {}).get("rate_limited_until", 0.0) > time.time()
                ) if any(self._status.get(k, {}).get("rate_limited_until", 0.0) > time.time() for k in candidate_keys) else 0.0
                backoff = max(3.0, next_ready - time.time() + 0.25) if next_ready else 3.0
                logger.warning("%s pool retrying after failure window (attempt %s/%s) in %.2fs", self.role, attempt + 1, retries, backoff)
                time.sleep(backoff)

        return None


# Explicitly isolated pools
BOT_GROQ_POOL = GroqKeyPool(role="bot")
JUDGE_GROQ_POOL = GroqKeyPool(role="judge")
GLOBAL_GROQ_POOL = BOT_GROQ_POOL
