### 1. Infrastructure & Hosting (The Live Endpoint)

* **Hosting Platform Selection:** Deploy to a cloud service that guarantees no idle-sleeping (e.g., Render Paid Tier, Fly.io, Railway, AWS EC2, or GCP Cloud Run with minimum instances set to 1). *Free tiers that sleep after 15 minutes of inactivity will lose in-memory state and fail the warmup check*.


* **Domain & HTTPS:** Secure a persistent public HTTPS URL (e.g., `[https://vera-bot.up.railway.app](https://vera-bot.up.railway.app)`) pointing directly to your root application.


* **Network & Timeout Configuration:** Configure your cloud reverse proxy (Nginx/Cloudflare) to allow HTTP POST payloads up to 1 MB and set worker read timeouts to $\ge 35\text{s}$ so reverse proxies don't terminate requests before the 30s deadline.



---

### 2. LLM Provider & Budget Governance

* **API Key Provisioning:** Set up API keys for your chosen frontier model provider (OpenAI, Anthropic, Gemini, or Groq).


* **Hard Spending Limits & Alerts:** Set a strict monthly spending cap (e.g., $10–$20) on your LLM provider dashboard to prevent runaway billing during 10 req/sec stress-testing passes.


* **Model Rate-Limit Verification:** Ensure your LLM account tier supports at least 60–100 Requests Per Minute (RPM) to handle Phase 2 ticks without hitting provider `429 Too Many Requests` limits.

---

### 3. Local Environment & Data Prep

* **Dataset Generation:** Run the dataset expansion script before starting development:
```bash
python3 dataset/generate_dataset.py --seed-dir dataset --out expanded

```


This creates the 50 merchants, 200 customers, 100 triggers, and the 30 canonical test pairs needed for benchmarking.


* **Environment Variables (`.env`):** Store secrets securely locally and in your deployment pipeline. Never commit keys to Git:
```env
LLM_PROVIDER=openai
LLM_API_KEY=sk-...
PORT=8080
LOG_LEVEL=INFO

```



---

### 4. Evaluation & Benchmarking Setup

* **Local Simulator Environment:** Configure `judge_simulator.py` locally to point to your development server (`http://localhost:8080`) to test behavioral scenario compliance (`auto_reply_hell`, `intent_transition`, `hostile`).


* **Custom Benchmarking Harness:** Create a lightweight evaluation script (`bench.py`) that loads `expanded/test_pairs.json`, invokes your `compose()` function in-process, and exports scores and outputs to a local CSV/JSON file to track improvements across prompt iterations.



---

### 5. Observability & Post-Submission Monitoring

* **Structured JSON Logging:** Implement standard stdout JSON logging for every incoming request and outgoing response:
* Log endpoint name, `context_id`, `version`, latency (ms), and emitted `rationale`.




* **Uptime Monitoring:** Set up a free external ping service (e.g., UptimeRobot, Better Stack) targeting `GET /v1/healthz` every 1–2 minutes. This alerts you immediately if your bot crashes or drops offline during or after evaluation.



---

### 6. Submission Artifacts Checklist

* **Git Version Control:** Initialize a clean Git repository with logical phase commits. Include a `.gitignore` excluding `.env`, `expanded/`, `__pycache__`, and temporary benchmarking logs.


* **Fallback Files (`submission.jsonl` & `bot.py`):** Pre-generate `submission.jsonl` (30 lines produced by your canonical harness) and confirm `bot.py` cleanly exports `compose()`.


* **1-Page `README.md`:** Draft a concise overview detailing:
1. System architecture & prompt strategy.


2. Primary model choice & rationale.


3. Trade-offs made (e.g., in-memory storage vs. database).

After making sure of these, only then read plan.md