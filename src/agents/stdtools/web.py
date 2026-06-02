import re
import urllib.parse

import requests


def web_search(query: str) -> str:
    """Performs a web search to find relevant information about a given query. Returns the top results as numbered snippets."""
    import random

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    ]

    headers = {
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://duckduckgo.com/",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    # Helper to parse DuckDuckGo HTML snippets across Standard and Lite interfaces
    def parse_html_snippets(html_text: str) -> list:
        # 1. Standard HTML version snippets
        snips = re.findall(
            r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
            html_text,
            re.DOTALL,
        )
        # 2. Fallback to Lite version snippets
        if not snips:
            snips = re.findall(
                r'<td[^>]*class="[^"]*result-snippet[^"]*"[^>]*>(.*?)</td>',
                html_text,
                re.DOTALL,
            )
        return snips

    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        # If rate limited (202) or fails, fallback to Lite version via POST
        if response.status_code == 202 or response.status_code != 200:
            lite_url = "https://lite.duckduckgo.com/lite/"
            data = {"q": query}
            headers["User-Agent"] = random.choice([ua for ua in user_agents if ua != headers["User-Agent"]])
            headers["Referer"] = "https://lite.duckduckgo.com/"
            response = requests.post(lite_url, headers=headers, data=data, timeout=10)

        if response.status_code == 200:
            snippets = parse_html_snippets(response.text)
            if snippets:
                results = []
                for i, snip in enumerate(snippets[:5]):
                    clean_snip = re.sub(r"<[^>]+>", "", snip).strip()
                    clean_snip = clean_snip.replace("&amp;", "&").replace("&quot;", '"').replace("&#x27;", "'")
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
            text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            # Remove all other HTML tags
            clean_text = re.sub(r"<[^>]+>", "", text)
            # Normalize whitespace
            clean_text = re.sub(r"\s+", " ", clean_text).strip()
            return clean_text[:4000]
        return f"Webpage fetch failed with status code {response.status_code}."
    except Exception as e:
        return f"Webpage fetch encountered an error: {str(e)}"
