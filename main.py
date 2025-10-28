import os
import re
import sys
import argparse
import logging
import datetime
import json
import csv
import time
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv  
from prompt import SYSTEM_PROMPT
from llm_api import call_llm  # type: ignore
from tools import make_tavily_client, tavily_search, tavily_extract  # type: ignore
from token_limit import get_token_limit  # type: ignore

load_dotenv()

# Define module-level logger for this module
logger = logging.getLogger("react_agent")

 

def _strip_url(u: str) -> str:
    try:
        return u.strip().strip("[]()<>\"'`,; ")
    except Exception:
        return u


def react_answer(question: str, model: Optional[str] = None, provider: Optional[str] = None, retries: int = 3) -> str:
    """
    Core ReAct loop: let the LLM decide to Search via Tavily until it produces a Final Answer and Stop.
    The loop no longer has a step limit; termination is controlled by the model.
    """
    provider = (provider or "").lower()
    logger.debug("LLM provider=%s", provider)

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
    log_filename = os.path.join(log_dir, safe_model_id, f"{timestamp}.json")
    # Ensure the model-specific subdirectory exists (avoid silent write failures)
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    log_data: List[Dict] = []
    log_data.append({"type": "user_query", "content": question.strip()})
    logger.info("[USER_QUERY] %s", question.strip())
    
    step = 0

    while True:
        step += 1
        # Keep step count internally; display is emitted per API call
        logger.info("-" * 50)
        
        # Budget gate before making the next model call
        if total_tokens >= threshold_tokens:
            logger.info("[TOKEN_BUDGET_REACHED] total=%d threshold=%d", total_tokens, threshold_tokens)
            log_data.append({"type": "token_budget_reached", "step": step, "total_tokens": total_tokens, "threshold_tokens": threshold_tokens})
            try:
                with open(log_filename, "w", encoding="utf-8") as f:
                    json.dump(log_data, f, ensure_ascii=False, indent=2)
            except Exception:
                logger.error("error: fail to write log file")
                sys.exit(1)
            return "token consumption reach the threshold, stop the conversation."
        
        # Call the LLM for the next step with retry logic
        attempt = 0
        last_error: Optional[Exception] = None
        while True:
            try:
                reply, usage = call_llm(oa_client, messages, model=model, provider=provider)
                break
            except Exception as e:
                attempt += 1
                last_error = e
                logger.warning("[LLM_CALL_FAILED] attempt=%d/%d error=%s", attempt, retries, e)
                log_data.append({"type": "llm_error", "step": step, "attempt": attempt, "error": str(e)})
                if attempt >= max(0, int(retries)):
                    logger.error("[LLM_CALL_ABORT] exceeded max retries=%d", retries)
                    # Persist the log before aborting
                    try:
                        with open(log_filename, "w", encoding="utf-8") as f:
                            json.dump(log_data, f, ensure_ascii=False, indent=2)
                    except Exception as write_err:
                        logger.error("Failed to write log file '%s': %s", log_filename, write_err)
                        sys.exit(1)
                    # Abort the run
                    sys.exit(1)
                # Backoff before the next attempt
                time.sleep(min(10, 1 + attempt * 2))
        # Inline tag parsing: require <plan> then exactly one action tag
        m_plan = re.search(r"(?is)<\s*plan\s*>\s*(.*?)\s*<\s*/\s*plan\s*>", reply or "")
        plan_text = m_plan.group(1).strip() if m_plan else ""
        m_ans = re.search(r"(?is)<\s*answer\s*>\s*(.*?)\s*<\s*/\s*answer\s*>", reply or "")
        answer_text = m_ans.group(1).strip() if m_ans else ""
        m_search = re.search(r"(?is)<\s*search\s*>\s*(.*?)\s*<\s*/\s*search\s*>", reply or "")
        search_query = m_search.group(1).strip() if m_search else ""
        m_access = re.search(r"(?is)<\s*access\s*>\s*(.*?)\s*<\s*/\s*access\s*>", reply or "")
        access_text = m_access.group(1).strip() if m_access else ""
        current_tokens = int(usage.get("total_tokens", 0))

        log_data.append({"type": "token_usage", "step": step, "total_tokens": current_tokens})
        total_tokens = current_tokens
        logger.info("[TOKEN_USAGE] step=%d total_tokens=%d", step, current_tokens)

        # If plan missing, guide the model to include it
        if not plan_text:
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

        # Determine action: enforce exactly one
        actions = []
        if answer_text:
            actions.append(("answer", answer_text))
        if search_query:
            actions.append(("search", search_query))
        if access_text:
            actions.append(("access", access_text))
        if len(actions) != 1:
            guidance = (
                "Please follow the protocol strictly,choose exactly ONE action tag: \n"
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
            messages.append({"role": "user", "content": f"Observation:\n{observation}"})
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
            messages.append({"role": "user", "content": f"Observation:\n{raw_text}"})
            continue
        # No other branches; action guidance handled above.


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Search agent with tools")
    parser.add_argument("--provider", default="openai", help="LLM provider to use")
    parser.add_argument("--model-id", default="gpt-5", help="Model identifier to use")
    parser.add_argument("--query", help="User query string to run through the agent")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries for failed LLM API calls")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug logging output (default: off)")
    args = parser.parse_args(argv[1:])

    # Configure logging to stdout (info-level by default; debug if --debug)
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")

    # Collect queries and invoke the agent. If no CLI argument is provided,
    # load problems from the CSV at the specified absolute path.
    queries: List[str] = []
    if args.query:
        queries = [args.query.strip()]
    else:
        default_csv = os.path.join(os.path.dirname(__file__), "eval", "browse_comp_sample.csv")
        try:
            with open(default_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    q = (row.get("problem") or "").strip()
                    if q:
                        queries.append(q)
            if not queries:
                print(f"No problems found in CSV: {default_csv}", file=sys.stderr)
                logger.error("No problems found in CSV: %s", default_csv)
                sys.exit(1)
        except Exception as e:
            print(f"Error reading default CSV '{default_csv}': {e}", file=sys.stderr)
            logger.error("Error reading default CSV '%s': %s", default_csv, e)
            sys.exit(1)

    for question in queries:
        react_answer(
            question,
            model=args.model_id,
            provider=args.provider,
            retries=args.retries,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))