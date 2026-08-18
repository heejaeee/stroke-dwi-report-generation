"""Replace the GitHub owner placeholder in public repository metadata."""

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = (ROOT / "README.md", ROOT / "CITATION.cff")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("owner", help="GitHub user or organization name")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.owner):
        raise SystemExit("Owner may contain only letters, numbers, dot, underscore, and hyphen.")

    for path in FILES:
        text = path.read_text(encoding="utf-8")
        updated = text.replace("<ORG_OR_USER>", args.owner)
        path.write_text(updated, encoding="utf-8", newline="\n")
        print(f"updated: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
