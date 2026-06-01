import re
import urllib.parse

import requests


def web_search(query: str) -> str:
    """Performs a web search to find relevant information about a given query. Returns the top results as numbered snippets."""
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            snippets = re.findall(
                r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
                response.text,
                re.DOTALL,
            )
            if snippets:
                results = []
                for i, snip in enumerate(snippets[:5]):
                    clean_snip = re.sub(r"<[^>]+>", "", snip).strip()
                    results.append(f"{i + 1}. {clean_snip}")
                return "\n\n".join(results)
            return "No results found."
        return f"Search failed with status code {response.status_code}."
    except Exception as e:
        return f"Search encountered an error: {str(e)}"


def fetch_webpage(url: str) -> str:
    """Fetches the text content of a webpage at the given URL, stripping all HTML markup. Returns up to 4000 characters."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            text = response.text
            # Remove scripts and styles
            text = re.sub(
                r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL
            )
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            # Remove all other HTML tags
            clean_text = re.sub(r"<[^>]+>", "", text)
            # Normalize whitespace
            clean_text = re.sub(r"\s+", " ", clean_text).strip()
            return clean_text[:4000]
        return f"Webpage fetch failed with status code {response.status_code}."
    except Exception as e:
        return f"Webpage fetch encountered an error: {str(e)}"
