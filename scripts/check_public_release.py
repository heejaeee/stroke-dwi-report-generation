"""Fail when a public release contains common private-data artifacts."""

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".jsonl", ".csv", ".toml"}
PRIVATE_SUFFIXES = {".dcm", ".dicom", ".nii", ".nrrd", ".mha", ".mhd", ".xlsx", ".xls", ".pt", ".pth", ".ckpt", ".safetensors"}
PATTERNS = {
    "Linux user or mounted-data path": re.compile(r"/(?:home|data|mnt)/[^\s\"']+", re.I),
    "Windows user path": re.compile(r"[A-Z]:\\Users\\", re.I),
    "UNC or private-network path": re.compile(r"\\\\10\.|\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    "workspace owner path": re.compile(r"(?:Workspace|Users)[\\/][^\\/\s]+", re.I),
}


def main():
    findings = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        lower = path.name.lower()
        if any(lower.endswith(suffix) for suffix in PRIVATE_SUFFIXES):
            findings.append(f"private file type: {path.relative_to(ROOT)}")
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{name}: {path.relative_to(ROOT)}")
    if findings:
        print("Public-release check failed:")
        print("\n".join(f"- {finding}" for finding in findings))
        return 1
    print("Public-release check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
