import os
import re
import sys
import argparse
import logging
import datetime
import json
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv  
load_dotenv()

# Define module-level logger for this module
logger = logging.getLogger("react_agent")

# New module imports
from llm_api import call_llm  # type: ignore
from tools import make_tavily_client, tavily_search, tavily_extract  # type: ignore
from summary import summarize_content  # type: ignore

SYSTEM_PROMPT = ("""You are an autonomous agent that uses web search and page access to solve the user's question.Given the question and the history context, generate the thought as well as the next action (only one action). The completed thought should contain analysis of available information and planning for future steps. Enclose the thought within <plan> </plan> tags. 

Allowed actions:
1. Search with a search engine, e.g. <search> the search query </search>
2. Access a URL found earlier, e.g. <access> the url to access </access>
3. Provide a final answer, e.g. <answer> the answer(should be a short answer) </answer>

Guidelines:
1. Double-check previous conclusions and identified facts using searches from different perspectives.
2. Try alternative approaches (different queries, different sources) when progress stalls.
3. Search results provide URL snippets,you can access those URLs to retrieve full content.
4. Always put the next action after the thought.
5. Choose exactly one action.
6. Use English for your search queries.

Thought: ... // the thought to be completed

Next Action: ... // the next action to be completed
"""
)


def parse_agent_reply(reply: str) -> Tuple[str, Optional[str]]:
    """
    Parse the agent's reply according to the new prompt protocol with tags.
    Returns a tuple (kind, payload):
    kind == "search": payload is the search query string inside <search>...</search>
    kind == "access": payload is the URL string inside <access>...</access>
    kind == "final": payload is the answer string inside <answer>...</answer>
    kind == "stop": payload is the raw reply (legacy Stop signal)
    kind == "other": payload is the raw reply
    """
    if not reply:
        return ("other", reply)

    # New tag-based parsing
    m_ans = re.search(r"(?is)<\s*answer\s*>\s*(.*?)\s*<\s*/\s*answer\s*>", reply)
    if m_ans:
        ans = m_ans.group(1).strip()
        if ans:
            return ("final", ans)

    m_search = re.search(r"(?is)<\s*search\s*>\s*(.*?)\s*<\s*/\s*search\s*>", reply)
    if m_search:
        query = m_search.group(1).strip()
        if query:
            return ("search", query)

    m_access = re.search(r"(?is)<\s*access\s*>\s*(.*?)\s*<\s*/\s*access\s*>", reply)
    if m_access:
        url = m_access.group(1).strip()
        if url:
            return ("access", url)

    # Backward-compatibility with prior protocol
    m_final = re.search(r"(?is)final\s+answer\s*:\s*(.*)", reply)
    if m_final:
        final = m_final.group(1).strip()
        if final:
            return ("final", final)

    if re.search(r"(?im)^\s*stop\s*$", reply):
        return ("stop", reply.strip())

    return ("other", reply.strip())


def _truncate_for_log(text: str, limit: int = 500) -> str:
    """Truncate long text for terminal logging to keep output readable."""
    try:
        s = (text or "").strip()
    except Exception:
        s = str(text)
    if len(s) <= limit:
        return s
    return s[:limit] + " ... [truncated]"


def _strip_url(u: str) -> str:
    try:
        return u.strip().strip("[]()<>\"'`,; ")
    except Exception:
        return u


def normalize_access_payload(payload: str):
    """
    Normalize the content inside <access>...</access> to a single URL string.
    - Removes surrounding quotes/brackets like []()<> if present
    - If a JSON array or text contains multiple URLs, only the first http(s) URL is returned
    """
    s = (payload or "").strip()

    # Remove symmetric surrounding quotes if any
    while len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'", "`"):
        s = s[1:-1].strip()

    # If wrapped in a single pair of brackets, peel once
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")) or (s.startswith("<") and s.endswith(">")):
        inner = s[1:-1].strip()
        if inner:
            s = inner

    # If JSON array, take the first URL-like item
    if s.startswith("[") and s.endswith("]"):
        try:
            data = json.loads(s)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str) and item.strip():
                        return _strip_url(item)
                    if isinstance(item, dict):
                        val = item.get("url") or item.get("link")
                        if isinstance(val, str) and val.strip():
                            return _strip_url(val)
        except Exception:
            pass

    # Regex extract: take the first http(s) URL found
    m = re.search(r'https?://[^\s)\]>\\"\'`]+', s)
    if m:
        return _strip_url(m.group(0))

    # Last chance: stripped URL-like
    s2 = _strip_url(s)
    if s2.startswith("http://") or s2.startswith("https://"):
        return s2

    return s


def _format_payload(payload: object) -> str:
    """Return a clean, human-readable string for request/response payloads."""
    try:
        if isinstance(payload, (dict, list)):
            return json.dumps(payload, ensure_ascii=False, indent=2)
        text = str(payload)
        return text
    except Exception:
        return str(payload)

def log_tag_summaries(log_data: List[Dict], reply: str) -> None:
    """Restore legacy-style logging: extract <plan>, and chosen action tags.
    Appends entries of types: plan, search, access, answer (if present).
    """
    if not reply:
        return
    try:
        m_plan = re.search(r"(?is)<\s*plan\s*>\s*(.*?)\s*<\s*/\s*plan\s*>", reply)
        if m_plan:
            content = m_plan.group(1).strip()
            if content:
                log_data.append({"type": "plan", "content": content})
                logger.info("[PLAN] %s", content)

        m_search = re.search(r"(?is)<\s*search\s*>\s*(.*?)\s*<\s*/\s*search\s*>", reply)
        if m_search:
            q = m_search.group(1).strip()
            if q:
                log_data.append({"type": "search", "content": q})
                logger.info("[SEARCH] %s", q)

        m_access = re.search(r"(?is)<\s*access\s*>\s*(.*?)\s*<\s*/\s*access\s*>", reply)
        if m_access:
            u = m_access.group(1).strip()
            if u:
                log_data.append({"type": "access", "content": u})
                logger.info("[ACCESS] %s", u)

        m_ans = re.search(r"(?is)<\s*answer\s*>\s*(.*?)\s*<\s*/\s*answer\s*>", reply)
        if m_ans:
            a = m_ans.group(1).strip()
            if a:
                log_data.append({"type": "answer", "content": a})
                logger.info("[ANSWER] %s", a)
    except Exception:
        # Do not fail logging on parsing issues
        pass


def react_answer(question: str, model: Optional[str] = None, provider: Optional[str] = None, use_summary: bool = True) -> str:
    """
    Core ReAct loop: let the LLM decide to Search via Tavily until it produces a Final Answer and Stop.
    The loop no longer has a step limit; termination is controlled by the model.
    """
    provider = (provider or "").lower()
    logger.debug("LLM provider=%s", provider)

    # All providers use the unified call_llm path; no client objects are needed
    oa_client = None
    tv_client = make_tavily_client()
    history_text = ""
    system_text = SYSTEM_PROMPT

    messages: List[Dict] = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": question.strip()},
    ]

    # Resolve model name from CLI only (ignore environment variables for model id)
    active_model_name = model 

    # Configure per-model token limits (simplified):
    def _resolve_token_limit(model_id: str):
        limits_by_model_id = {
            "gpt-5": 400000,
            "gpt-4.1": 1000000,
            "gemini-2.5-flash": 1000000,
            "gemini-2.5-pro": 1000000,
            "qwen3-235b-a22b-thinking-2507":128000,
            "qwen3-235b-a22b-instruct-2507":128000,
            "kimi-k2-0905-preview": 260000,
            "us.anthropic.claude-sonnet-4-20250514-v1:0": 200000,
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0": 200000,
            "glm-4.5": 128000,
            "deepseek-chat": 128000, #deepseek v3.1 terminus
            "deepseek-reasoner": 128000, #deepseek v3.1 terminus thinking mode
            "alibaba/tongyi-deepresearch-30b-a3b": 128000,
            "grok-4-0709": 256000,
        }
        key = (model_id or "").lower()
        return limits_by_model_id.get(key, f"Token limit not mapped for model id: {model_id}")

    limit_or_hint = _resolve_token_limit(active_model_name)
    if isinstance(limit_or_hint, int):
        token_limit: Optional[int] = limit_or_hint
        threshold_tokens: Optional[int] = int(token_limit * 0.9)
    else:
        logger.debug(str(limit_or_hint))
        token_limit = None
        threshold_tokens = None
    # Enforce budget with a cumulative threshold gate before starting a new LLM round
    running_total_tokens: int = 0

    # Initialize JSON logging: use the active_model_name for all providers
    model_id = active_model_name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.getcwd(), "log")
    os.makedirs(log_dir, exist_ok=True)
    # Sanitize model_id to be Windows filename-safe (replace characters like : <> : \ / | ? * with '-')
    safe_model_id = re.sub(r'[<>:"/\\|?*]+', '-', model_id)
    log_filename = os.path.join(log_dir, f"{safe_model_id}_{timestamp}.json")
    log_data: List[Dict] = []
    # Restore legacy log structure: record user query immediately
    log_data.append({"type": "user_query", "content": question.strip()})
    logger.info("[USER_QUERY] %s", question.strip())
    
    step = 0
    pending_summary = None  # defer summarization: replace raw observation after next LLM thought

    while True:
        step += 1
        # Keep step count internally; display is emitted per API call
        
        # Budget gate before making the next model call
        if threshold_tokens is not None and running_total_tokens >= threshold_tokens:
            logger.info("[TOKEN_BUDGET_REACHED] running=%d threshold=%d", running_total_tokens, threshold_tokens)
            log_data.append({"type": "token_budget_reached", "step": step, "running_total_tokens": running_total_tokens, "threshold_tokens": threshold_tokens})
            try:
                with open(log_filename, "w", encoding="utf-8") as f:
                    json.dump(log_data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return "已达到 token 预算阈值，停止继续对话。"
        
        # 调用 LLM
        reply, usage = call_llm(oa_client, messages, model=model, provider=provider)
        # 恢复旧式日志：记录计划与所选动作标签
        log_tag_summaries(log_data, reply)
        
        # Compute step total tokens for budget enforcement using provider-reported usage
        step_total_tokens = -1
        if usage is not None:
            try:
                step_total_tokens = int(usage.get("total_tokens", 0))
            except Exception:
                step_total_tokens = -1
 
        if step_total_tokens >= 0:
            # 记录本步 token 使用，并仅保存最近一次的 token 值
            log_data.append({"type": "token_usage", "step": step, "total_tokens": step_total_tokens})
            running_total_tokens = step_total_tokens
            logger.info("[TOKEN_USAGE] step=%d total_tokens=%d", step, step_total_tokens)

        # After model has seen the raw observation in the previous step, optionally replace it with a summary
        # Only summarize observations originating from the extract/access tool
        if use_summary and pending_summary and pending_summary.get("kind") == "access":
            try:
                summarized = summarize_content(pending_summary.get("raw_text", "") or "", source_url=pending_summary.get("source_url"))
            except Exception:
                summarized = pending_summary.get("raw_text", "") or ""
            # Build replacement block for Access only
            new_block = f"\n[Access]\nURL: {pending_summary.get('url')}\n{summarized}\n"
            # Replace in history_text
            history_text = history_text.replace(pending_summary.get("history_raw_block", ""), new_block, 1)
            # Update the corresponding message content
            idx = pending_summary.get("message_index")
            if isinstance(idx, int) and 0 <= idx < len(messages):
                messages[idx]["content"] = f"Observation:\n{summarized}"
            # 恢复旧式日志：记录已应用的摘要
            log_data.append({
                "type": "summary_applied",
                "observation": summarized,
                "kind": "access",
                "url": pending_summary.get("url"),
            })
            logger.info("[SUMMARY_APPLIED] kind=access url=%s\n%s", pending_summary.get("url"), _truncate_for_log(summarized))
            pending_summary = None

        kind, payload = parse_agent_reply(reply)
        if kind == "final" and isinstance(payload, str):
            # 记录最终答案并保存日志（旧版结构）
            log_data.append({"type": "final", "content": payload})
            logger.info("[FINAL] %s", payload)
            try:
                with open(log_filename, "w", encoding="utf-8") as f:
                    json.dump(log_data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return payload
        elif kind == "search" and isinstance(payload, str):
            query = payload
            try:
                search_results = tavily_search(tv_client, query)
            except Exception as e:
                logger.debug("Search failed: %s", e)
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": f"Search failed: {e}. Try a different query or provide an answer if enough information is available."})
                continue
            # Append observation
            if isinstance(search_results, str):
                observation = search_results
            else:
                # Fallback formatting if an unexpected structure is returned
                try:
                    import json as _json
                    observation = _json.dumps(search_results, ensure_ascii=False, indent=2)
                except Exception:
                    observation = str(search_results)
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"Observation:\n{observation}"})
            continue
        elif kind == "access" and isinstance(payload, str):
            url = normalize_access_payload(payload)
            try:
                extracted = tavily_extract(tv_client, url)
            except Exception as e:
                logger.debug("Access failed: %s", e)
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": f"Access failed: {e}. Try a different URL or provide an answer if enough information is available."})
                continue
            # Append observation (raw; may summarize after next step)
            raw_text = (extracted or "").strip()
            url_display = url
            obs_block = f"\n[Access]\nURL: {url_display}\n{raw_text}\n"
            # 恢复旧式日志：记录访问原始观察
            log_data.append({
                "type": "access_observation_raw",
                "url": url_display,
                "raw_text": raw_text,
            })
            logger.info("[ACCESS_OBSERVATION_RAW] url=%s\n%s", url_display, _truncate_for_log(raw_text))
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"Observation:\n{raw_text}"})
            history_text += obs_block
            # Defer summarization until after model has reacted once to the raw observation
            pending_summary = {
                "kind": "access",
                "url": url_display,
                "raw_text": raw_text,
                "source_url": url,
                "history_raw_block": obs_block,
                "message_index": len(messages) - 1,
            }
            continue
        else:
            # Model deviated; nudge it to follow the protocol (noisy content suppressed)
            guidance = (
                "Please follow the protocol strictly: \n"
                "1) choose exactly ONE action tag: <search>, <access>, or <answer>;  \n"
                "2) if answering, put the final answer inside <answer>...</answer>. \n"
                "3) if searching, put the query inside <search>...</search>. \n"
                "4) if accessing, put the URL inside <access>...</access>."
            )
            # Do not record guidance in API-only logging mode
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": guidance})
            continue


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="ReAct agent with Tavily and LLM choice")
    parser.add_argument("--provider", "-p", default="openai", help="LLM provider to use")
    parser.add_argument("--model_id", "-id", default="gpt-5", help="Model identifier to use")
    parser.add_argument("--query", "-q", help="User query string to run through the agent")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug logging output (default: off)")
    parser.add_argument("--summary", dest="use_summary", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable summarization of extracted content (default: on)")
    args = parser.parse_args(argv[1:])

    # Configure logging to stdout (info-level by default; debug if --debug)
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")

    # Quiet third-party verbose loggers (expanded)
    for noisy in (
        "openai",
        "httpx",
        "httpcore",
        "anthropic",
        "botocore",
        "boto3",
        "s3transfer",
        "urllib3",
    ):
        try:
            logging.getLogger(noisy).setLevel(logging.WARNING)
        except Exception:
            pass


    # Collect query and invoke the agent (provider and model id from CLI; no env fallback)
    if args.query:
        question = args.query.strip()
    else:
        # Backward compatibility: allow passing question as positional or fall back to input()
        if len(argv) > 1:
            question = " ".join(argv[1:]).strip()
        else:
            question = input("Enter your question: ")

    react_answer(
        question,
        model=args.model_id,
        provider=args.provider,
        use_summary=args.use_summary,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))