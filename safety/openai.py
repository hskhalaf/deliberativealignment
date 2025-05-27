# --- openai.py ---
import os, json, yaml, asyncio, aiohttp, pandas as pd, re

URL = "https://go.apis.huit.harvard.edu/ais-openai-direct-limited-schools/v1/chat/completions"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HEADERS = {"Content-Type": "application/json",
           "api-key": yaml.safe_load(open(os.path.join(ROOT, "secrets.yaml")))["api_key"]}

DEFAULT_CFG  = {"temperature": 1.2, "max_tokens": 1_800}
MAX_RETRIES, TIMEOUT = 3, 45

# ------------------------------------------------------------------ helpers
def _strip_fence(txt: str | None) -> str | None:
    if not txt:
        return None
    m = re.match(r"```[\w]*\n([\s\S]*?)```", txt.strip())
    if m:
        return m.group(1).strip()
    j = re.search(r'(\[[\s\S]*\])', txt.strip())
    return j.group(1).strip() if j else txt.strip()

# ------------------------------------------------------------------ low‑level POST
async def _post_request(session: aiohttp.ClientSession, payload: dict) -> str | None:
    delay = 2
    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(URL, headers=HEADERS, json=payload, timeout=TIMEOUT) as r:
                r.raise_for_status()
                data = await r.json()          # ←‑‑‑ await here!
                return data["choices"][0]["message"]["content"]
        except asyncio.TimeoutError:
            print("Timeout.")
        except Exception as e:
            print("Request failed:", e)
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(delay)
                delay *= 2
    return None

# ------------------------------------------------------------------ one prompt
async def _gen_single(prompt: str, opts: dict, session: aiohttp.ClientSession) -> dict:
    payload = {
        "model": opts.get("model_id", "gpt-4o"),
        "messages": [{"role": "user", "content": prompt}],
        **DEFAULT_CFG,
        "max_tokens": opts.get("max_length", 1_800),
    }
    raw   = _strip_fence(await _post_request(session, payload))
    parsed = None
    if raw:
        try:
            parsed = json.loads(raw) if isinstance(json.loads(raw), list) else None
        except json.JSONDecodeError:
            pass
    return {"raw": raw, "json": parsed}

# ------------------------------------------------------------------ many prompts
async def _gen_many(prompts: list[str], opts: dict) -> pd.DataFrame:
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        tasks   = [_gen_single(p, opts, session) for p in prompts]
        results = await asyncio.gather(*tasks)
    return pd.DataFrame(results)

def gen_prompts(prompts: list[str], opts: dict) -> pd.DataFrame:
    """`prompts` must be a **list** of user‑prompt strings."""
    print(f"Processing {len(prompts)} prompts with model {opts['model_id']}.")
    return asyncio.run(_gen_many(prompts, opts))
