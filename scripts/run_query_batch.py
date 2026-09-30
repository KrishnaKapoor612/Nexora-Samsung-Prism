"""Run each line from a query file against the local API and save every result.

From the repository root, start the API in one terminal, then run:
    python scripts/run_query_batch.py

The default input is student_kit/input.txt and the default output is
evaluation_results.jsonl. Each output line is a complete, independently
readable JSON record and is flushed immediately so completed responses survive
if a later request fails or the run is interrupted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent.parent


def _queries(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _post_query(url: str, query: str, timeout: float) -> tuple[int | None, object]:
    payload = json.dumps({"query": query}, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.status
            body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        status = exc.code
        body = exc.read().decode("utf-8", errors="replace")
    except (URLError, TimeoutError, OSError) as exc:
        return None, {"error": str(exc)}

    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, {"error": "API response was not JSON", "body": body}


def _check_health(query_url: str, timeout: float) -> tuple[bool, str]:
    health_url = query_url.removesuffix("/v1/troubleshoot").rstrip("/") + "/health"
    try:
        with urlopen(health_url, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8", errors="replace"))
            if response.status == 200 and body.get("status") == "ok":
                return True, health_url
            return False, f"{health_url} returned HTTP {response.status}: {body}"
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        return False, f"Cannot reach {health_url}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send queries sequentially to the troubleshooting API and save responses."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "student_kit" / "input.txt",
        help="UTF-8 text file with one query per line (default: student_kit/input.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL results file (default: a timestamped file in the project root)",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/v1/troubleshoot",
        help="Troubleshooting endpoint URL",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Maximum seconds to wait for each query",
    )
    args = parser.parse_args()

    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    if args.output is None:
        output_path = ROOT / f"evaluation_results_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    else:
        output_path = args.output if args.output.is_absolute() else ROOT / args.output
    if not input_path.is_file():
        print(f"Query file not found: {input_path}", file=sys.stderr)
        return 2
    queries = _queries(input_path)
    if not queries:
        print(f"No non-empty queries in: {input_path}", file=sys.stderr)
        return 2

    healthy, health_message = _check_health(args.url, min(args.timeout, 10.0))
    if not healthy:
        print(f"API health check failed. Start the server first. {health_message}", file=sys.stderr)
        return 3
    print(f"API is healthy at {health_message}; running {len(queries)} queries sequentially.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"answer": 0, "no_match": 0, "http_error": 0}
    started = time.time()
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for index, query in enumerate(queries, start=1):
            status, result = _post_query(args.url, query, args.timeout)
            if status is None or status < 200 or status >= 300:
                actual = "http_error"
                counts[actual] += 1
            elif isinstance(result, dict):
                contexts = result.get("response", {}).get("contexts", [])
                actual = "answer" if contexts else "no_match"
                counts[actual] += 1
            else:
                actual = "http_error"
                counts[actual] += 1

            record = {
                "index": index,
                "query": query,
                # Fill this with "answer" or "no_match" from a trusted
                # reference set before using the file for precision/recall.
                "expected_outcome": None,
                "actual_outcome": actual,
                "http_status": status,
                "result": result,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            print(f"[{index}/{len(queries)}] {actual} (HTTP {status}) - {query}")

    print(f"\nSaved {len(queries)} query results to: {output_path}")
    print(
        "Actual outcomes: "
        f"answer={counts['answer']}, no_match={counts['no_match']}, "
        f"http_error={counts['http_error']}"
    )
    print(f"Elapsed: {time.time() - started:.1f}s")
    print("Precision/recall need expected_outcome labels from a trusted reference set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
