import os
import re
import sys
import argparse
import logging
import shutil
import datetime
import json
import csv
import time
import threading
import uuid
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv  
from service.prompt import SYSTEM_PROMPT
from service.llm_api import call_llm  # type: ignore
from service.tools import make_tavily_client, tavily_search, tavily_extract  # type: ignore
from service.token_limit import get_token_limit  # type: ignore
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# Define module-level logger for this module
logger = logging.getLogger("react_agent")

def _log_step_separator() -> None:
    # Print two dashed lines sized to the current terminal width
    width = shutil.get_terminal_size(fallback=(80, 24)).columns
    line = "-" * max(10, width)
    logger.info(line)
    logger.info(line)
 

def _strip_url(u: str) -> str:
    try:
        return u.strip().strip("[]()<>\"'`,; ")
    except Exception:
        return u

def _safe_model_subdir(model_id: Optional[str]) -> str:
    """Return a Windows filename-safe subdirectory name for a model id."""
    mid = (model_id or "unknown-model")
    return re.sub(r'[<>:"/\\|?*]+', '-', mid)

def _load_processed_problems(model_id: Optional[str]) -> set:
    """
    Read existing JSON logs under log/<model-id>/ and collect queries that
    have already been processed. Handles initial case where the directory or
    files do not exist.
    """
    processed: set = set()
    base_log_dir = os.path.join(os.path.dirname(__file__), "log")
    safe_mid = _safe_model_subdir(model_id)
    model_log_dir = os.path.join(base_log_dir, safe_mid)

    if not os.path.isdir(model_log_dir):
        # Initial case: no logs yet
        return processed

    try:
        for name in os.listdir(model_log_dir):
            if not name.lower().endswith(".json"):
                continue
            fpath = os.path.join(model_log_dir, name)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Logs are stored as a list of entries; be robust to dict
                if isinstance(data, list):
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        # Prefer explicit 'query', fallback to 'user_query'
                        if item.get("type") == "query" and isinstance(item.get("content"), str):
                            processed.add(item["content"].strip())
                        elif item.get("type") == "user_query" and isinstance(item.get("content"), str):
                            processed.add(item["content"].strip())
                elif isinstance(data, dict):
                    # Fallback: single-object log formats
                    q = None
                    if isinstance(data.get("query"), str):
                        q = data.get("query")
                    elif isinstance(data.get("user_query"), str):
                        q = data.get("user_query")
                    if q:
                        processed.add(q.strip())
            except Exception:
                # Skip unreadable or malformed log files silently
                continue
    except Exception:
        # If listing fails, treat as no processed problems
        return processed

    return processed


def react_answer(question: str, model: Optional[str] = None, provider: Optional[str] = None, retries: int = 3) -> str:
    """
    Core ReAct loop: let the LLM decide to Search via Tavily until it produces a Final Answer and Stop.
    The loop no longer has a step limit; termination is controlled by the model.
    """
    logger.debug("LLM provider=%s", provider)

    # Identify current thread and generate a strong unique run_id for this task
    thread_id = threading.get_ident()
    run_id = uuid.uuid4().hex

    # All providers use the unified call_llm path; no client objects are needed
    oa_client = None
    tv_client = make_tavily_client()
    system_text = SYSTEM_PROMPT

    messages: List[Dict] = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": question.strip()},
    ]

    limit_or_hint = get_token_limit(model)
    if isinstance(limit_or_hint, int):
        threshold_tokens: Optional[int] = int(limit_or_hint * 0.9)
    else:
        error_msg = limit_or_hint
        print(error_msg, file=sys.stderr)
        logger.error(error_msg)
        sys.exit(1)
    # Enforce budget with a cumulative threshold gate before starting a new LLM round
    total_tokens=0

    # Initialize JSON logging: use the active_model_name for all providers
    model_id = model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Write logs under the project directory regardless of current working dir
    log_dir = os.path.join(os.path.dirname(__file__), "log")
    os.makedirs(log_dir, exist_ok=True)
    # Sanitize model_id to be Windows filename-safe (replace characters like : <> : \ / | ? * with '-')
    safe_model_id = re.sub(r'[<>:"/\\|?*]+', '-', model_id)
    # Use shortened run_id (first 4 chars) in file name to keep names compact
    log_filename = os.path.join(log_dir, safe_model_id, f"{timestamp}-{run_id[:4]}.json")
    # Ensure the model-specific subdirectory exists (avoid silent write failures)
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    log_data: List[Dict] = []
    log_data.append({"type": "user_query", "content": question.strip()})
    logger.info("[USER_QUERY] %s", question.strip())
    
    step = 0
    # When limits are hit, switch to forced final-answer mode
    force_final_answer = False

    while True:
        step += 1
        # Separator lines to delineate parallel LLM call outputs
        _log_step_separator()

        # If we hit max steps or token budget, request immediate final answer
        if step >= 80 or total_tokens >= threshold_tokens:
            logger.info("[FINAL_ANSWER_REQUEST] step=%d total_tokens=%d threshold_tokens=%d", step, total_tokens, threshold_tokens)
            log_data.append({"type": "final_answer_request", "reason": "max_steps_or_tokens_reached", "step": step, "total_tokens": total_tokens, "threshold_tokens": threshold_tokens})
            messages.append({
                "role": "user",
                "content": (
                    "You have reached the maximum step count or token count. Provide your final answer immediately, "
                    "enclosed within <answer>...</answer> tags, using all prior information. "
                ),
            })
            force_final_answer = True
        # Proceed to LLM call (normal or forced final answer)
        
        # Call the LLM for the next step with retry logic
        attempt = 0
        while True:
            try:
                reply, usage = call_llm(oa_client, messages, model=model, provider=provider)
                break
            except Exception as e:
                attempt += 1
                err_str = str(e)
                logger.warning("[LLM_CALL_FAILED] attempt=%d/%d error=%s", attempt, retries, e)
                log_data.append({"type": "llm_error", "step": step, "attempt": attempt, "error": err_str})
                if attempt >= max(0, int(retries)):
                    logger.error("[LLM_CALL_ABORT] exceeded max retries=%d", retries)
                    # Do not persist the JSON log on abort; exit immediately
                    sys.exit(1)
                # Backoff before the next attempt
                time.sleep(min(10, 1 + attempt * 2))
        # Inline tag parsing: require <plan> then exactly one action tag (unless forcing final answer)
        m_plan = re.search(r"(?is)<\s*plan\s*>\s*(.*?)\s*<\s*/\s*plan\s*>", reply or "")
        plan_text = m_plan.group(1).strip() if m_plan else ""
        m_ans = re.search(r"(?is)<\s*answer\s*>\s*(.*?)\s*<\s*/\s*answer\s*>", reply or "")
        answer_text = m_ans.group(1).strip() if m_ans else ""
        m_search = re.search(r"(?is)<\s*search\s*>\s*(.*?)\s*<\s*/\s*search\s*>", reply or "")
        search_query = m_search.group(1).strip() if m_search else ""
        m_access = re.search(r"(?is)<\s*access\s*>\s*(.*?)\s*<\s*/\s*access\s*>", reply or "")
        access_text = m_access.group(1).strip() if m_access else ""
        current_tokens = int(usage.get("total_tokens", 0))

        # Record token usage (do not include thread/run identifiers in persisted log data)
        log_data.append({"type": "token_usage", "step": step, "total_tokens": current_tokens})
        total_tokens = current_tokens
        logger.info("[QUERY] %s", question)
        logger.info("[TOKEN_USAGE] step=%d total_tokens=%d", step, current_tokens)

        # If plan missing, guide the model to include it unless forcing final answer
        if not plan_text and not force_final_answer:
            guidance = (
                "Please follow the protocol strictly: \n"
                "provide your thought and planning within <plan>...</plan> tags. This is required.\n"
            )
            logger.info("[GUIDANCE] %s", guidance)
            log_data.append({"type": "guidance", "content": guidance})
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": guidance})
            continue
        # Log plan when present
        log_data.append({"type": "plan", "content": plan_text})
        logger.info("[PLAN] %s", plan_text)
        messages.append({"role": "assistant", "content": reply})

        # Determine action: enforce exactly one (unless forcing final answer)
        actions = []
        if answer_text:
            actions.append(("answer", answer_text))
        if search_query:
            actions.append(("search", search_query))
        if access_text:
            actions.append(("access", access_text))
        # Merge action validation: in forced mode require exactly one action and it must be <answer>
        invalid_action = False
        if force_final_answer:
            invalid_action = (len(actions) != 1 or actions[0][0] != "answer")
        else:
            invalid_action = (len(actions) != 1)
        if invalid_action:
            if force_final_answer:
                guidance = (
                    "Provide the FINAL ANSWER now enclosed within <answer>...</answer> tags only. "
                    "Do not use <plan>, <search>, or <access>."
                )
            else:
                guidance = (
                    "Please follow the protocol strictly, choose exactly ONE action tag: \n"
                    "1) <search>query</search> to search for information \n"
                    "2) <access>url</access> to access a specific URL \n"
                    "3) <answer>final answer</answer> to provide your final answer"
                )
            logger.info("[GUIDANCE] %s", guidance)
            log_data.append({"type": "guidance", "content": guidance})
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": guidance})
            continue
        kind, payload = actions[0]

        if kind == "answer" and isinstance(payload, str):
            log_data.append({"type": "answer", "content": payload})
            logger.info("[ANSWER] %s", payload)
            # Persist logs for this query; if the primary path fails, abort
            try:
                with open(log_filename, "w", encoding="utf-8") as f:
                    json.dump(log_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error("Failed to write log file '%s': %s", log_filename, e)
                sys.exit(1)
            return payload
        elif kind == "search" and isinstance(payload, str):
            query = payload
            # Log the SEARCH tag and record in log data
            logger.info("[SEARCH] %s", query)
            log_data.append({"type": "search", "query": query})
            search_results = tavily_search(tv_client, query)
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
            # Append only the observation; assistant reply already appended after <plan>
            messages.append({"role": "user", "content": f"{observation}"})
            continue
        elif kind == "access" and isinstance(payload, str):
            url = _strip_url(payload)
            extracted = tavily_extract(tv_client, url)
            # Append observation (raw; may summarize after next step)
            raw_text = (extracted or "").strip()
            url_display = url
            # Log the ACCESS tag and record in log data
            logger.info("[ACCESS] %s", url_display)
            log_data.append({"type": "access", "url": url_display})
            log_data.append({
                "type": "access_observation_raw",
                "url": url_display,
                "raw_text": raw_text,
            })
            logger.info("[ACCESS_OBSERVATION_RAW] url=%s\n%s", url_display, raw_text)
            # Append only the observation; assistant reply already appended after <plan>
            messages.append({"role": "user", "content": f"{raw_text}"})
            continue
        # No other branches; action guidance handled above.


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Search agent with tools")
    parser.add_argument("--provider", default="openai", help="LLM provider to use")
    parser.add_argument("--model-id", default="gpt-5", help="Model identifier to use")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries for failed LLM API calls")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug logging output (default: off)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker threads to process queries")
    args = parser.parse_args(argv[1:])

    # Configure logging to stdout (info-level by default; debug if --debug)
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")

    # Collect queries and invoke the agent. If no CLI argument is provided,
    # load problems from the CSV at the specified absolute path.
    queries: List[str] = []
    processed_problems = _load_processed_problems(args.model_id)
    default_csv = os.path.join(os.path.dirname(__file__), "eval", "browse_comp_sample.csv")
    try:
        with open(default_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = (row.get("problem") or "").strip()
                if q:
                    if q in processed_problems:
                        logger.info("[SKIP_ALREADY_PROCESSED] %s", q[:120])
                        continue
                    queries.append(q)
        if not queries:
            print(f"No new problems to process in CSV: {default_csv}", file=sys.stderr)
            logger.error("No new problems to process in CSV: %s", default_csv)
            sys.exit(1)
    except Exception as e:
        print(f"Error reading default CSV '{default_csv}': {e}", file=sys.stderr)
        logger.error("Error reading default CSV '%s': %s", default_csv, e)
        sys.exit(1)

    # Process queries in parallel using a thread pool; support Ctrl+C to interrupt promptly
    executor = ThreadPoolExecutor(max_workers=max(1, int(args.workers)))
    try:
        futures = [
            executor.submit(
                react_answer,
                question,
                model=args.model_id,
                provider=args.provider,
                retries=args.retries,
            )
            for question in queries
        ]
        for fut in as_completed(futures):
            try:
                _ = fut.result()
            except Exception as e:
                logger.error("[WORKER_ERROR] %s", e)
        return 0
    except KeyboardInterrupt:
        logger.warning("[INTERRUPTED] Received Ctrl+C; canceling pending tasks...")
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        # Exit with 130 (conventional code for SIGINT)
        sys.exit(130)
    finally:
        # Ensure executor is torn down
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main(sys.argv))