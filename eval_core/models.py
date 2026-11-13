import itertools
import json
import os

import requests

_key_cycle = None


def load_gemini_keys(path="API_keys.json"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            keys = json.load(f).get("GEMINI_API_KEYS", [])
        if keys:
            return keys
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return [env_key]
    raise RuntimeError("No Gemini key found. Add API_keys.json or set GEMINI_API_KEY.")


def _next_key():
    global _key_cycle
    if _key_cycle is None:
        _key_cycle = itertools.cycle(load_gemini_keys())
    return next(_key_cycle)


def ask_gemini(prompt, system, temperature=0.3, model="gemini-2.5-flash", api_key=None):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key or _next_key())
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system,
        ),
    )
    return response.text or ""


def ask_ollama(prompt, system, model, temperature=0.2,
               host="http://localhost:11434", timeout=300):
    reply = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=timeout,
    )
    reply.raise_for_status()
    return reply.json()["message"]["content"]
