import os
import re
import sys
import argparse
import logging
import datetime
import json
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv  
from llm_api import call_llm  # type: ignore
from tools import make_tavily_client, tavily_search, tavily_extract  # type: ignore
from summary import summarize_content  # type: ignore
from token_limit import get_token_limit  # type: ignore

load_dotenv(override=True)

# Define module-level logger for this module
logger = logging.getLogger("react_agent")

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
"""
)


def parse_agent_reply(reply: str) -> Tuple[str, Optional[str], Dict[str, str]]:
    """
    Parse the agent's reply according to the new prompt protocol with tags.
    Returns a tuple (kind, payload, all_tags):
    kind == "search": payload is the search query string inside <search>...</search>
    kind == "access": payload is the URL string inside <access>...</access>
    kind == "final": payload is the answer string inside <answer>...</answer>
    kind == "plan": payload is the plan content inside <plan>...</plan>
    all_tags: dict containing all extracted tags for logging
    """
    if not reply:
        return ("other", reply, {})

    # Extract all tags once to avoid duplicate parsing
    all_tags = {}
    
    # Extract plan tag
    m_plan = re.search(r"(?is)<\s*plan\s*>\s*(.*?)\s*<\s*/\s*plan\s*>", reply)
    if m_plan:
        all_tags["plan"] = m_plan.group(1).strip()

    # New tag-based parsing - check action tags first
    m_ans = re.search(r"(?is)<\s*answer\s*>\s*(.*?)\s*<\s*/\s*answer\s*>", reply)
    if m_ans:
        ans = m_ans.group(1).strip()
        if ans:
            all_tags["answer"] = ans
            return ("answer", ans, all_tags)

    m_search = re.search(r"(?is)<\s*search\s*>\s*(.*?)\s*<\s*/\s*search\s*>", reply)
    if m_search:
        query = m_search.group(1).strip()
        if query:
            all_tags["search"] = query
            return ("search", query, all_tags)

    m_access = re.search(r"(?is)<\s*access\s*>\s*(.*?)\s*<\s*/\s*access\s*>", reply)
    if m_access:
        url = m_access.group(1).strip()
        if url:
            all_tags["access"] = url
            return ("access", url, all_tags)
    
    # If we have a plan tag but no action tag, return plan
    if "plan" in all_tags:
        return ("plan", all_tags["plan"], all_tags)
    
    # Default fallback for unrecognized replies
    return ("other", reply, all_tags)


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


def _format_payload(payload: object) -> str:
    """Return a clean, human-readable string for request/response payloads."""
    try:
        if isinstance(payload, (dict, list)):
            return json.dumps(payload, ensure_ascii=False, indent=2)
        text = str(payload)
        return text
    except Exception:
        return str(payload)


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
    limit_or_hint = get_token_limit(model)
    if isinstance(limit_or_hint, int):
        threshold_tokens: Optional[int] = int(limit_or_hint * 0.9)
    else:
        error_msg = limit_or_hint
        print(error_msg, file=sys.stderr)
        logger.error(error_msg)
        sys.exit(1)
    # Enforce budget with a cumulative threshold gate before starting a new LLM round
    running_total_tokens: int = 0

    # Initialize JSON logging: use the active_model_name for all providers
    model_id = model
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
        logger.info("-" * 50)
        
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
        # 解析回复并获取所有标签
        kind, payload, all_tags = parse_agent_reply(reply)
        
        # Compute step total tokens for budget enforcement using provider-reported usage
        if usage is not None:
            try:
                step_total_tokens = int(usage.get("total_tokens", 0))
            except Exception:
                error_msg = f"错误：无法解析模型 '{model}' 的 token 使用信息。{usage}"
                print(error_msg, file=sys.stderr)
                logger.error(error_msg)
                sys.exit(1)
 
        if step_total_tokens >= 0:
            # 记录本步 token 使用，并仅保存最近一次的 token 值
            log_data.append({"type": "token_usage", "step": step, "total_tokens": step_total_tokens})
            running_total_tokens = step_total_tokens
            logger.info("[TOKEN_USAGE] step=%d total_tokens=%d", step, step_total_tokens)

        # Always log PLAN content if present, even when an action tag is included
        plan_text = all_tags.get("plan")
        if plan_text:
            log_data.append({"type": "plan", "content": plan_text})
            logger.info("[PLAN] %s", plan_text)

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

        if kind == "plan" and isinstance(payload, str):
            # 已记录 PLAN；Plan标签不是动作，继续循环等待动作标签
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": "Please provide your next action: <search>, <access>, or <answer>."})
            continue
        elif kind == "answer" and isinstance(payload, str):
            # 记录最终答案并保存日志（旧版结构）
            log_data.append({"type": "answer", "content": payload})
            logger.info("[ANSWER] %s", payload)
            try:
                with open(log_filename, "w", encoding="utf-8") as f:
                    json.dump(log_data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return payload
        elif kind == "search" and isinstance(payload, str):
            query = payload
            # Log the SEARCH tag and record in log data
            logger.info("[SEARCH] %s", query)
            log_data.append({"type": "search", "query": query})
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
            url = _strip_url(payload)
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
            # Log the ACCESS tag and record in log data
            logger.info("[ACCESS] %s", url_display)
            log_data.append({"type": "access", "url": url_display})
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
            # Check if plan tag was provided in all_tags
            has_plan = "plan" in all_tags and all_tags["plan"]
            
            if not has_plan:
                # If no plan tag, provide guidance that includes plan requirement
                guidance = (
                    "Please follow the protocol strictly: \n"
                    "1) First, provide your thought and planning within <plan>...</plan> tags. This is required.\n"
                    "2) Then choose exactly ONE action tag: <search>, <access>, or <answer>;  \n"
                    "3) if answering, put the final answer inside <answer>...</answer>. \n"
                    "4) if searching, put the query inside <search>...</search>. \n"
                    "5) if accessing, put the URL inside <access>...</access>."
                )
            else:
                # If plan tag exists but no valid action tag, focus on action requirement
                guidance = (
                    "You provided a plan, but please also choose exactly ONE action tag: \n"
                    "1) <search>query</search> to search for information \n"
                    "2) <access>url</access> to access a specific URL \n"
                    "3) <answer>final answer</answer> to provide your final answer"
                )
            
            # Record guidance in both logger and log data
            logger.info("[GUIDANCE] %s", guidance)
            log_data.append({"type": "guidance", "content": guidance})
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