import os
import re
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

# Ensure project root is importable when running this file directly from the subfolder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("react_agent")

def _safe_model_id_from_filename(filename: str) -> str:
    """Extract model id prefix from filenames like 'model_YYYYMMDD_HHMMSS.json'."""
    base = os.path.basename(filename)
    m = re.match(r"^(.+?)_\d{8}_\d{6}\.json$", base)
    if m:
        return m.group(1)
    return os.path.splitext(base)[0]

def load_eval_map(csv_path: str) -> Dict[str, str]:
    """Load problem->answer map from the BrowseComp decoded CSV.
    Assumes headers include 'problem' and 'answer'.
    """
    import csv
    mapping: Dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            problem = (row.get("problem") or "").strip()
            answer = (row.get("answer") or "").strip()
            if problem:
                mapping[problem] = answer
    return mapping


def read_log_file(path: str) -> List[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to read log %s: %s", path, e)
        return []


def extract_log_summary(log_entries: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
    """Get (user_query, final_answer) from legacy-style log entries."""
    user_query: Optional[str] = None
    answer: Optional[str] = None
    for it in log_entries:
        t = (it.get("type") or "").lower()
        if t == "user_query" and not user_query:
            user_query = (it.get("content") or "").strip()
        elif t == "answer" and not answer:
            answer = (it.get("content") or "").strip()
        # Early exit if both found
        if user_query and answer:
            break
    return user_query, answer


def build_reviewer_messages(
    model_id: str,
    user_query: Optional[str],
    expected_answer: Optional[str],
    final_answer: Optional[str],
    full_log: List[Dict],
) -> List[Dict[str, str]]:
    """Compose messages for full analysis (only when quick judge is not CORRECT).
    The model returns plain analysis text; no JSON from the model is required.
    """
    system_prompt = (
        "You are an expert evaluator reviewing browsing-agent test logs. "
        "Provide very detailed and deep analysis of the browsing-agent's performance. "
        "you should focus on the agent planning or thought trajectory and analysis why it fails to get the expected answer. "
        "Return only an analysis text, covering:\n"
        "- Whether the final answer matches the expected answer (and why).\n"
        "- If incorrect, a detailed failure analysis.\n"
        "- Concrete citations to log content; no generalities.\n"
    )

    user_payload = {
        "model_id": model_id,
        "user_query": user_query,
        "expected_answer": expected_answer,
        "final_answer": final_answer,
        "log_entries": full_log
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    return messages


def build_quick_judge_messages(
    expected_answer: str,
    final_answer: str,
    user_query: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Compose minimal messages to judge correctness only.

    The model must respond exactly with 'CORRECT' or 'INCORRECT'.
    Avoids sending the full log to reduce token usage.
    """
    system_prompt = (
        "You are a strict grader. Compare the final_answer to expected_answer. "
        "Judge semantic equivalence, ignoring superficial formatting differences. "
        "Respond with ONLY 'CORRECT' or 'INCORRECT'. No explanations."
    )

    user_payload = {
        "user_query": user_query,
        "expected_answer": expected_answer,
        "final_answer": final_answer,
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]
    return messages


def _list_json_files(dir_path: str) -> List[str]:
    try:
        return [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.lower().endswith(".json")
        ]
    except Exception:
        return []


def _get_log_keywords() -> List[str]:
    """Resolve log filename filter keywords from env.

    Reads comma-separated `REVIEW_LOG_KEYWORDS` from environment/.env.
    """
    raw = os.getenv("REVIEW_LOG_KEYWORDS", "").strip()
    return [s.strip().lower() for s in raw.split(",") if s.strip()]


def _find_log_by_query(log_dir: str, query: str) -> List[Dict]:
    """Return log entries for the first log whose user_query matches."""
    for lp in _list_json_files(log_dir):
        entries = read_log_file(lp)
        uq, _fa = extract_log_summary(entries)
        if (uq or "").strip() == (query or "").strip():
            return entries
    return []


def _write_review(out_dir: str, base_name: str, review_obj: Dict[str, str]) -> None:
    """Write the review object immediately to the review directory."""
    out_name = base_name + "-review.json"
    out_path = os.path.join(out_dir, out_name)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(review_obj, f, ensure_ascii=False, indent=2)
        logger.info("Wrote review: %s", out_path)
        logger.info("token usage: %s", review_obj.get("token_usage"))
        logger.info("___"*50)
    except Exception as e:
        logger.error("Failed to write review %s: %s", out_path, e)


def run_reviewer() -> None:
    """Run LLM reviewer without CLI args using project defaults and requirements.

    - Loads expected answers from `eval/browse_comp_test_set_decoded.csv` (columns: problem, answer).
    - Requires JSON test logs under `log/` (legacy list-of-entries format).
    - For each log: extract `user_query` and final answer; look up expected answer by `user_query` in CSV.
    - Perform a quick judge (expects exactly 'CORRECT' or 'INCORRECT'). If correct, save immediately.
    - Otherwise, run a full analysis using the complete log, and save immediately.
    """
    # Logging setup (no CLI)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cwd = os.getcwd()
    eval_dir = os.path.join(cwd, "eval")
    log_dir = os.path.join(cwd, "log")
    out_dir = os.path.join(cwd, "review")
    csv_path = os.path.join(eval_dir, "browse_comp_test_set_decoded.csv")
    # Configure OpenAI model via env var; defaults to a fast, capable model
    model_name = "gpt-5"

    os.makedirs(out_dir, exist_ok=True)

    # Load expected answers
    eval_map = load_eval_map(csv_path)

    # OpenAI provider via official API
    def _call_openai(messages: List[Dict[str, str]], model: str) -> Tuple[str, Optional[Dict[str, int]]]:
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.openai.com/v1")
            raw_resp = client.chat.completions.with_raw_response.create(
                model=model,
                messages=messages,
            )
        except Exception as e:
            raise SystemExit(f"Error: LLM API request failed: {e}")
        # Parse structured response and validate content
        resp = raw_resp.parse()
        msg = resp.choices[0].message
        content = (getattr(msg, "content", None) or "").strip()
        u = getattr(resp, "usage", None)
        usage: Optional[Dict[str, int]] = None
        if u:
            pt = int(getattr(u, "prompt_tokens", 0))
            ct = int(getattr(u, "completion_tokens", 0))
            tt = int(getattr(u, "total_tokens", pt + ct))
            usage = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}
        return content, usage

    # Ensure required inputs exist: CSV in eval and JSON logs in log
    if not os.path.isfile(csv_path):
        raise SystemExit(f"Error: CSV not found at {csv_path}.")

    log_json_files = _list_json_files(log_dir)
    # Filter to only logs whose filenames contain any configured keywords
    keywords = _get_log_keywords()
    log_json_files = [
        p for p in log_json_files
        if any(k in os.path.basename(p).lower() for k in keywords)
    ]
    if not log_json_files:
        raise SystemExit(
            f"Error: no log JSON files in {log_dir} matching keywords: {', '.join(keywords)}."
        )

    # Review directly from logs
    for log_path in sorted(log_json_files):
        logger.info("Reviewing log %s", os.path.basename(log_path))
        entries = read_log_file(log_path)
        user_query, final_answer = extract_log_summary(entries)
        if not user_query:
            logger.warning("Skipping %s: missing user_query.", log_path)
            continue
        expected_answer = eval_map.get(user_query)
        if not expected_answer:
            logger.error("Expected answer not found for query; skipping. Query: %s", user_query)
            continue
        model_id = _safe_model_id_from_filename(log_path)

        # Quick judge
        judge_messages = build_quick_judge_messages(
            expected_answer=expected_answer,
            final_answer=(final_answer or ""),
            user_query=user_query,
        )
        try:
            judge_content, _jusage = _call_openai(judge_messages, model=model_name)
            normalized = (judge_content or "").strip().upper()
            if normalized == "CORRECT":
                review_obj = {"model_id": model_id, "user_query": user_query, "review": "CORRECT", "token_usage": _jusage}
                _write_review(out_dir, os.path.splitext(os.path.basename(log_path))[0], review_obj)
                continue
        except Exception as e:
            review_obj = {"model_id": model_id, "user_query": user_query, "review": f"LLM quick judge failed: {e}", "token_usage": None}
            _write_review(out_dir, os.path.splitext(os.path.basename(log_path))[0], review_obj)
            continue

        # Full analysis
        messages = build_reviewer_messages(
            model_id=model_id,
            user_query=user_query,
            expected_answer=expected_answer,
            final_answer=final_answer,
            full_log=entries,
        )
        try:
            content, _usage = _call_openai(messages, model=model_name)
            review_obj = {"model_id": model_id, "user_query": user_query, "review": content, "token_usage": _usage}
        except Exception as e:
            review_obj = {"model_id": model_id, "user_query": user_query, "review": f"LLM review failed: {e}", "token_usage": None}
        _write_review(out_dir, os.path.splitext(os.path.basename(log_path))[0], review_obj)
    return


if __name__ == "__main__":
    run_reviewer()