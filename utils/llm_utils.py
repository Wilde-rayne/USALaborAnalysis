import requests
from functools import lru_cache

from utils.constants import (
    OLLAMA_URL,
    OLLAMA_API_PATH,
    MODEL_NAME,
    DEFAULT_TIMEOUT
)

@lru_cache(maxsize=256)
def generate_insight(prompt: str, timeout: int = DEFAULT_TIMEOUT, active_tab: str = None) -> str:
    """
    Retrieval-augmented chat: fetches top-3 context snippets (from data and documentation) for the active tab,
    then calls Ollama with them as system context alongside the user prompt.
    """
    from utils.embeddings import retrieve_context, get_relevant_context


    if active_tab:
        context = get_relevant_context(active_tab, top_k=3)
    else:
        context = retrieve_context(prompt, top_k=3)
    context = context.strip() if context else ""


    system_msg = "You are a data assistant helping with Midwest BLS data."
    if context:
        system_msg += " Here is the relevant context:\n\n" + context
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt}
    ]


    payload = {"model": MODEL_NAME, "messages": messages}

    try:
        resp = requests.post(
            f"{OLLAMA_URL.rstrip('/')}{OLLAMA_API_PATH}",
            json=payload,
            timeout=timeout
        )
        resp.raise_for_status()
        data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            return "[AI] No response choices returned."

        return choices[0].get("message", {}).get("content", "[AI] No content returned.")
    except requests.RequestException as e:
        return f"[AI] Error generating insight: {e}"
