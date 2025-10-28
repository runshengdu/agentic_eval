"""
Download the BrowseComp test set CSV and also produce a human-readable (decrypted) CSV.
Uses the same decrypt logic as eval/browsecomp_eval.py.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import sys
from pathlib import Path
from urllib.request import urlopen

# URL sourced from eval/browsecomp_eval.py
DATASET_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"


def derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the password using SHA256 (same as browsecomp_eval.py)."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64-encoded ciphertext with XOR (same as browsecomp_eval.py)."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


def download_csv(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as resp:  # nosec - public CSV file
        data = resp.read()
    dest.write_bytes(data)


def decode_csv(src: Path, dest: Path) -> None:
    """Read the encrypted CSV and write a new CSV containing only decoded, human-readable fields."""
    with src.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")
        # Only keep decoded, user-friendly columns
        fieldnames = ["problem", "answer", "problem_topic"]

        rows_out: list[dict[str, str]] = []
        for row in reader:
            canary = row.get("canary", "")
            try:
                dec_problem = decrypt(row.get("problem", ""), canary)
            except Exception:
                dec_problem = ""
            try:
                dec_answer = decrypt(row.get("answer", ""), canary)
            except Exception:
                dec_answer = ""
            rows_out.append({
                "problem": dec_problem,
                "answer": dec_answer,
                "problem_topic": row.get("problem_topic", ""),
            })

    with dest.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)


def main() -> int:
    eval_dir = Path(__file__).parent
    raw_path = eval_dir / "browse_comp.csv"
    decoded_path = eval_dir / "browse_comp_decoded.csv"

    try:
        # Always (re)download the raw CSV to ensure freshness
        download_csv(DATASET_URL, raw_path)
        print(f"Saved raw dataset to: {raw_path}")

        # Produce a human-readable decoded version
        decode_csv(raw_path, decoded_path)
        print(f"Saved decoded dataset to: {decoded_path}")
        return 0
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())