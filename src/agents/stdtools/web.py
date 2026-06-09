import json
import random
import urllib.parse

import requests
from bs4 import BeautifulSoup


def web_search(query: str) -> str:
    """Performs a web search to find relevant information about a given query. Returns the top results as JSON."""

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

    def parse_standard_results(soup: BeautifulSoup) -> list[tuple[str, str]]:
        """Parse results from the standard DuckDuckGo HTML interface."""
        results = []
        for result_div in soup.select(".result"):
            link_tag = result_div.select_one("a.result__a")
            snippet_tag = result_div.select_one("a.result__snippet")
            url = (
                link_tag["href"]
                if link_tag and link_tag.has_attr("href")
                else ""
            )
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
            if url or snippet:
                results.append((url, snippet))
        return results

    def parse_lite_results(soup: BeautifulSoup) -> list[tuple[str, str]]:
        """Parse results from the DuckDuckGo Lite interface."""
        results = []
        link_tags = soup.select("a.result-link")
        snippet_tags = soup.select("td.result-snippet")
        for link_tag, snippet_tag in zip(link_tags, snippet_tags):
            url = link_tag["href"] if link_tag.has_attr("href") else ""
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
            if url or snippet:
                results.append((url, snippet))
        return results

    def parse_html_results(html_text: str) -> list[tuple[str, str]]:
        """Parse DuckDuckGo HTML results using BeautifulSoup, with Lite fallback."""
        soup = BeautifulSoup(html_text, "html.parser")
        results = parse_standard_results(soup)
        if not results:
            results = parse_lite_results(soup)
        return results

    search_url = (
        f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    )
    try:
        response = requests.get(search_url, headers=headers, timeout=10)

        # If rate limited (202) or fails, fallback to Lite version via POST
        if response.status_code == 202 or response.status_code != 200:
            lite_url = "https://lite.duckduckgo.com/lite/"
            data = {"q": query}
            headers["User-Agent"] = random.choice(
                [ua for ua in user_agents if ua != headers["User-Agent"]]
            )
            headers["Referer"] = "https://lite.duckduckgo.com/"
            response = requests.post(
                lite_url, headers=headers, data=data, timeout=10
            )

        if response.status_code == 200:
            parsed = parse_html_results(response.text)
            if parsed:
                results = []
                for i, (link, snip) in enumerate(parsed[:5]):
                    entry = {"rank": i + 1, "snippet": snip}
                    if link:
                        entry["url"] = link
                    results.append(entry)
                return json.dumps(results, indent=2)
            return "No results found."
        return f"Search failed with status code {response.status_code}."
    except Exception as e:
        return f"Search encountered an error: {str(e)}"


def fetch_webpage(url: str) -> str:
    """Fetches the text content of a webpage at the given URL, stripping all HTML markup, while preserving links (absolute URLs) for ease of traversal. Returns up to 4000 characters."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            # Remove scripts and styles
            for tag in soup(["script", "style"]):
                tag.decompose()

            # Format <a> tags to include their absolute href URLs
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if href:
                    absolute_href = urllib.parse.urljoin(url, href)
                    text = a.get_text().strip()
                    if text:
                        a.replace_with(f" [{text}]({absolute_href}) ")
                    else:
                        a.replace_with(f" ({absolute_href}) ")

            clean_text = soup.get_text(separator=" ", strip=True)
            return clean_text[:4000]
        return f"Webpage fetch failed with status code {response.status_code}."
    except Exception as e:
        return f"Webpage fetch encountered an error: {str(e)}"
