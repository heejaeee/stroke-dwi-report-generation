import argparse
import json
import re
from pathlib import Path


CT_ONLY_PATTERN = re.compile(
    r"^\s*no\s+definite\s+evidence\s+of\s+acute\s+infarction\s+"
    r"on\s+this\s+ct\s+scan\s*[.]?\s*$",
    re.IGNORECASE,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "filter": "exclude CT-only negative targets from DWI report generation",
        "source_dir": str(source_dir),
        "out_dir": str(out_dir),
        "splits": {},
        "removed": [],
    }

    for split in ["train", "val", "test"]:
        input_path = source_dir / f"{split}_raw_overlay_crop_messages.jsonl"
        output_path = out_dir / input_path.name
        kept = []
        removed = []

        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                target = str(record.get("target", "")).strip()
                if CT_ONLY_PATTERN.fullmatch(target):
                    removed.append(
                        {
                            "split": split,
                            "id": record.get("id"),
                            "patient_id": record.get("patient_id"),
                            "target": target,
                        }
                    )
                else:
                    kept.append(record)

        with output_path.open("w", encoding="utf-8") as handle:
            for record in kept:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary["splits"][split] = {
            "input": len(kept) + len(removed),
            "kept": len(kept),
            "removed": len(removed),
        }
        summary["removed"].extend(removed)

    (out_dir / "report_target_filter_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
