import logging
from typing import List, Union
from tavily import TavilyClient  
import os
from dotenv import load_dotenv
from summary import summarize_content
load_dotenv(override=True)
import tiktoken
logger = logging.getLogger("react_agent")

def make_tavily_client() -> "TavilyClient":
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def tavily_search(client: "TavilyClient", query: str, max_results: int = 5) -> str:
    """
    Run a Tavily search and return an observation string containing ONLY:
    answer (top-level string from Tavily's response, when available)
    title (per result)
    content (per result)
    url (per result)
    """
    logger.debug("TOOL CALL: name=Tavily.search params=%s", {"query": query, "max_results": max_results})
    raw = client.search(
        query=query, 
        max_results=max_results, 
        include_raw_content=False, 
        include_answer=False,
        search_depth="basic",
        )

    results: List[dict] = []
    if isinstance(raw, dict):
        if isinstance(raw.get("results"), list):
            results = raw["results"]
        elif isinstance(raw.get("result"), dict):
            results = [raw["result"]]
    elif isinstance(raw, list):
        results = raw

    # Filter out results whose URL contains blocked keywords
    exclude_keywords = ["huggingface.co","huggingface.com","arxiv.org"]
    filtered_results: List[dict] = []
    for r in results:
        url_lower = ((r.get("url") or "").strip().lower())
        if any(kw in url_lower for kw in exclude_keywords):
            continue
        filtered_results.append(r)

    logger.debug(
        "TOOL RESULT: name=Tavily.search raw_count=%d filtered_count=%d blocked_keywords=%s",
        len(results),
        len(filtered_results),
        ",".join(exclude_keywords),
    )

    lines = ["Search Results:"]
    # if answer:
    #     lines.append(f"Answer: {answer}")

    for idx, r in enumerate(filtered_results, start=1):
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
    # Create the encoder once and reuse for all items
    enc = tiktoken.get_encoding("o200k_base")
    for idx, r in enumerate(results, start=1):
        url = (r.get("url") or r.get("link") or "").strip()
        raw_content = (r.get("raw_content") or "").strip()
        # Safely compute token count and summarize when content is too long
        try:
            token_count = len(enc.encode(raw_content))
        except Exception:
            token_count = 0 if not raw_content else len(raw_content)
        if token_count > 4000:
            raw_content = summarize_content(raw_content)
        item = f"{idx}. URL: {url}\nRaw Content:\n{raw_content}"
        lines.append(item)

    if len(lines) == 1:
        lines.append("(No results returned)")

    observation = "\n\n".join(lines)
    return observation

if __name__ == "__main__":
    client = make_tavily_client()
    query = "huggingface"
    observation = tavily_search(client, query)
    print(observation)
