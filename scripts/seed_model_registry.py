#!/usr/bin/env python3
import json
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROXY_DIR = os.path.join(ROOT, "proxy")
if PROXY_DIR not in sys.path:
    sys.path.insert(0, PROXY_DIR)

from services.model_registry import sync_registry_from_env_checks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed model registry from explicit checks JSON.")
    parser.add_argument(
        "--checks-file",
        required=True,
        help="Path to JSON file with an array of model checks compatible with sync_registry_from_env_checks",
    )
    args = parser.parse_args()

    with open(args.checks_file, "r", encoding="utf-8") as f:
        checks = json.load(f)
    if not isinstance(checks, list):
        raise SystemExit("checks-file must contain a JSON array")

    report = sync_registry_from_env_checks(checks)
    print(json.dumps({"status": "ok", "report": report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
