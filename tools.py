import logging
from typing import List, Union
from tavily import TavilyClient  
logger = logging.getLogger("react_agent")

def make_tavily_client() -> "TavilyClient":
    if TavilyClient is None:
        raise RuntimeError(
            "The 'tavily-python' package is not installed. Please run: pip install -r requirements.txt"
        )
    # Key is read from env by TavilyClient; we validate existence upstream to avoid leaking secrets here.
    return TavilyClient()


def tavily_search(client: "TavilyClient", query: str, max_results: int = 5) -> str:
    """
    Run a Tavily search and return an observation string containing ONLY:
    answer (top-level string from Tavily's response, when available)
    title (per result)
    content (per result)
    url (per result)
    """
    logger.debug("TOOL CALL: name=Tavily.search params=%s", {"query": query, "max_results": max_results})

    try:
        raw = client.search(query=query, max_results=max_results, include_raw_content=False, include_answer=True)
    except TypeError:
        # Some versions might use positional args
        raw = client.search(query, max_results=max_results, include_raw_content=False, include_answer=True)

    results: List[dict] = []
    answer: str = ""
    if isinstance(raw, dict):
        if isinstance(raw.get("results"), list):
            results = raw["results"]
        elif isinstance(raw.get("result"), dict):
            results = [raw["result"]]
        answer = (raw.get("answer") or "").strip()
    elif isinstance(raw, list):
        results = raw

    logger.debug("TOOL RESULT: name=Tavily.search result_count=%d", len(results))

    lines = ["Search Results:"]
#    if answer:
#        lines.append(f"Answer: {answer}")

    for idx, r in enumerate(results, start=1):
        title = (r.get("title")).strip()
        url = (r.get("url")).strip()
        content = (r.get("content")).strip()
        item = f"{idx}. title:{title}\ncontent:{content}\nURL: {url}"
        lines.append(item)
    if len(lines) == 1:
        lines.append("(No results returned)")

    observation = "\n\n".join(lines)
    return observation


def tavily_extract(client: "TavilyClient", url_or_urls: Union[str, List[str]]) -> str:
    """
    Use Tavily Extract to fetch full raw content from specific URL(s) and format as an observation string.
    Accepts either a single URL (string) or a list of URLs.

    Simplified version: rely on Tavily SDK supporting `extract(urls=List[str])`.
    """
    urls: List[str] = [url_or_urls] if isinstance(url_or_urls, str) else list(url_or_urls)

    logger.debug(
        "TOOL CALL: name=Tavily.extract params=%s",
        {"urls": urls},
    )

    # Simplified: direct call with keyword argument; assumes up-to-date SDK
    raw = client.extract(urls=urls)

    results: List[dict] = []
    failed_count = 0
    if isinstance(raw, dict):
        if isinstance(raw.get("results"), list):
            results = raw["results"]
        if isinstance(raw.get("failed_results"), list):
            failed_count = len(raw["failed_results"])
    elif isinstance(raw, list):
        results = raw

    logger.debug(
        "TOOL RESULT: name=Tavily.extract result_count=%d failed_count=%d",
        len(results),
        failed_count,
    )

    lines = ["Extract Results:"]
    for idx, r in enumerate(results, start=1):
        url = (r.get("url") or r.get("link") or "").strip()
        raw_content = (r.get("raw_content") or "").strip()
        # Fallback to other content fields if raw_content missing
        if not raw_content:
            raw_content = (r.get("content") or r.get("snippet") or r.get("description") or "").strip()
        images = r.get("images") or []
        favicon = (r.get("favicon") or "").strip()
        item = f"{idx}. URL: {url}\nImages: {len(images)}\nFavicon: {favicon}\nRaw Content:\n{raw_content}"
        lines.append(item)

    if len(lines) == 1:
        lines.append("(No results returned)")

    observation = "\n\n".join(lines)
    return observation
