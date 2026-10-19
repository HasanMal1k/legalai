"""
app/config.py

Paths, model names and secrets for the app. Secrets come from the environment; on Windows a key set with
`setx NAME "value"` is also read from the user's environment in the registry, so the app works without
restarting the editor or terminal it was started from. Keys are never written to disk or logged.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
APP_DATA = DATA / "app"                       # the app's own files: caches, the question and feedback database
EMBED_STORE = DATA / "embeddings" / "kanon-2-embedder"

EMBED_MODEL = "kanon-2-embedder"
RERANK_MODEL = "kanon-2-reranker"
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")

ANSWER_COLLECTION = ("current", "unconfirmed")   # chunk `serve` values questions are answered from
HISTORY_COLLECTION = ("historical",)            # shown only as "what the law was before"


def secret(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
            return value or None
    except (ImportError, OSError):
        return None
