"""
Gemini API Key Pool & Instant Rate-Limit Rotator.
Manages key rotation for both Bot (Composer) and Judge Simulator with Gemini Flash 3.5 Lite.
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

logger = logging.getLogger("gemini_pool")


class GeminiKeyPool:
    """
    Thread-safe and async-safe Gemini API Key Pool with instant rate-limit failover.
    Can be scoped to 'bot', 'judge', or 'all' to ensure the Bot and Judge never collide
    on the same quota budget.
    """
    def __init__(self, role: str = "bot"):
        self.role = role.lower().strip()
        self._lock = threading.Lock()
        self._keys: List[str] = []
        self._status: Dict[str, Dict[str, Any]] = {}
        self._current_index = 0
        self._invalid_retry_seconds = 600.0
        self.reload_keys()

    def reload_keys(self) -> List[str]:
        with self._lock:
            all_keys: List[str] = []

            # 1. Check explicit comma-separated lists
            for env_name in ("GEMINI_KEYS", "GEMINI_API_KEYS", "GEMINI_POOL_KEYS"):
                raw = os.environ.get(env_name, "")
                if raw:
                    for k in raw.replace(",", "\n").splitlines():
                        k = k.strip()
                        if k and k not in all_keys:
                            all_keys.append(k)

            # 2. Numbered keys GEMINI_API_KEY_1 .. 20
            for i in range(1, 21):
                k = (os.environ.get(f"GEMINI_API_KEY_{i}") or "").strip()
                if k and k not in all_keys:
                    all_keys.append(k)

            # 3. Named keys
            for key_env in ("GEMINI_API_KEY", "PRIMARY_LLM_KEY", "SECONDARY_LLM_KEY", "LLM_API_KEY"):
                primary = (os.environ.get(key_env) or "").strip()
                if primary and (primary.startswith("AIzaSy") or len(primary) > 25) and primary not in all_keys:
                    all_keys.append(primary)

            # Partition keys if multiple keys available
            if len(all_keys) >= 2:
                mid = max(1, len(all_keys) // 2)
                bot_keys = all_keys[:mid]
                judge_keys = all_keys[mid:]
            else:
                bot_keys = all_keys
                judge_keys = all_keys

            if self.role == "judge":
                self._keys = judge_keys if judge_keys else all_keys
            elif self.role == "bot":
                self._keys = bot_keys if bot_keys else all_keys
            else:
                self._keys = all_keys

            for k in self._keys:
                if k not in self._status:
                    self._status[k] = {
                        "rate_limited_until": 0.0,
                        "invalid": False,
                        "invalid_until": 0.0,
                        "successes": 0,
                        "rate_limits": 0,
                    }

            if self._current_index >= len(self._keys):
                self._current_index = 0
            return list(self._keys)

    @property
    def total_keys(self) -> int:
        return len(self._keys)

    def get_candidate_keys(self) -> List[str]:
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

            return active + cooling if active or cooling else list(self._keys)

    def mark_rate_limited(self, key: str, cooldown_seconds: float = 30.0):
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
        models: List[str] = []
        if preferred_model:
            models.append(preferred_model)
        env_model = os.environ.get("GEMINI_MODEL") or os.environ.get("LLM_MODEL")
        if env_model and env_model not in models:
            models.append(env_model)
        defaults = [
            "gemini-3.5-flash-lite",
            "gemini-3.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
        ]
        for model in defaults:
            if model not in models:
                models.append(model)
        return models

    def _extract_content(self, data: Dict[str, Any]) -> Optional[str]:
        candidates = data.get("candidates", [])
        if not candidates:
            return None
        candidate = candidates[0]
        content_obj = candidate.get("content", {})
        parts = content_obj.get("parts", [])
        if not parts:
            return None
        text = parts[0].get("text", "").strip()
        return text if text else None

    async def execute_chat_async(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        timeout: float = 3.5,
        on_success: Optional[Callable[[str], None]] = None,
    ) -> Optional[str]:
        """
        Asynchronously executes chat completion against Google Gemini API with instant key failover on 429.
        """
        candidate_keys = self.get_candidate_keys()
        if not candidate_keys:
            logger.warning("%s gemini pool exhausted: no keys available", self.role)
            return None

        models_to_try = self.get_preferred_models(model)

        # Format Gemini messages
        system_instruction = None
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = {"parts": [{"text": content}]}
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
            else:
                contents.append({"role": "user", "parts": [{"text": content}]})

        if not contents and system_instruction:
            contents.append({"role": "user", "parts": [{"text": "Please proceed."}]})

        for key in candidate_keys[:4]:
            status = self._status.get(key, {})
            if status.get("invalid", False):
                continue

            for g_model in models_to_try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{g_model}:generateContent?key={key}"
                payload: Dict[str, Any] = {
                    "contents": contents,
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": max_tokens,
                    }
                }
                if system_instruction:
                    payload["systemInstruction"] = system_instruction

                try:
                    start = time.time()
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        res = await client.post(
                            url,
                            json=payload,
                            headers={"Content-Type": "application/json"}
                        )
                    latency = time.time() - start

                    if res.status_code == 200:
                        data = res.json()
                        content = self._extract_content(data)
                        if content:
                            self.mark_success(key)
                            logger.info("%s gemini async success via %s/%s in %.2fs", self.role, key[:8], g_model, latency)
                            if on_success:
                                on_success(g_model)
                            return content
                        continue
                    elif res.status_code == 429:
                        logger.warning("%s gemini async 429 via %s/%s in %.2fs", self.role, key[:8], g_model, latency)
                        self.mark_rate_limited(key, cooldown_seconds=30.0)
                        break
                    elif res.status_code in {400, 401, 403}:
                        logger.warning("%s gemini async %s via %s/%s: %s", self.role, res.status_code, key[:8], g_model, res.text[:200])
                        self.mark_invalid(key)
                        break
                    elif res.status_code == 404:
                        logger.warning("%s gemini model %s 404; trying next model", self.role, g_model)
                        continue
                    else:
                        continue
                except Exception as exc:
                    logger.warning("%s gemini async error via %s/%s: %s", self.role, key[:8], g_model, exc)
                    break

        return None

    def execute_chat_sync(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        timeout: float = 12.0,
        on_success: Optional[Callable[[str], None]] = None,
    ) -> Optional[str]:
        """
        Synchronously executes chat completion against Google Gemini API with instant key failover on 429.
        """
        retries = 2
        for attempt in range(retries + 1):
            candidate_keys = self.get_candidate_keys()
            if not candidate_keys:
                logger.warning("%s gemini pool exhausted: no keys available", self.role)
                return None

            models_to_try = self.get_preferred_models(model)

            # Format Gemini messages
            system_instruction = None
            contents = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    system_instruction = {"parts": [{"text": content}]}
                elif role == "assistant":
                    contents.append({"role": "model", "parts": [{"text": content}]})
                else:
                    contents.append({"role": "user", "parts": [{"text": content}]})

            if not contents and system_instruction:
                contents.append({"role": "user", "parts": [{"text": "Please proceed."}]})

            for key in candidate_keys:
                status = self._status.get(key, {})
                if status.get("invalid", False):
                    continue

                for g_model in models_to_try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{g_model}:generateContent?key={key}"
                    payload: Dict[str, Any] = {
                        "contents": contents,
                        "generationConfig": {
                            "temperature": temperature,
                            "maxOutputTokens": max_tokens,
                        }
                    }
                    if system_instruction:
                        payload["systemInstruction"] = system_instruction

                    try:
                        start = time.time()
                        with httpx.Client(timeout=timeout) as client:
                            res = client.post(
                                url,
                                json=payload,
                                headers={"Content-Type": "application/json"}
                            )
                        latency = time.time() - start

                        if res.status_code == 200:
                            data = res.json()
                            content = self._extract_content(data)
                            if content:
                                self.mark_success(key)
                                logger.info("%s gemini sync success via %s/%s in %.2fs", self.role, key[:8], g_model, latency)
                                if on_success:
                                    on_success(g_model)
                                return content
                            continue
                        elif res.status_code == 429:
                            logger.warning("%s gemini sync 429 via %s/%s in %.2fs", self.role, key[:8], g_model, latency)
                            self.mark_rate_limited(key, cooldown_seconds=30.0)
                            break
                        elif res.status_code in {400, 401, 403}:
                            logger.warning("%s gemini sync %s via %s/%s: %s", self.role, res.status_code, key[:8], g_model, res.text[:200])
                            self.mark_invalid(key)
                            break
                        elif res.status_code == 404:
                            logger.warning("%s gemini model %s 404; trying next model", self.role, g_model)
                            continue
                        else:
                            continue
                    except Exception as exc:
                        logger.warning("%s gemini sync error via %s/%s: %s", self.role, key[:8], g_model, exc)
                        break

            if attempt < retries:
                time.sleep(1.0)

        return None


# Isolated Pools for Bot and Judge
BOT_GEMINI_POOL = GeminiKeyPool(role="bot")
JUDGE_GEMINI_POOL = GeminiKeyPool(role="judge")
GEMINI_POOL = BOT_GEMINI_POOL
