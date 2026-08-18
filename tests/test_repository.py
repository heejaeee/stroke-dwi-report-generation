import py_compile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_public_python_sources_compile(self):
        paths = list((ROOT / "src").glob("*.py")) + list((ROOT / "scripts").glob("*.py"))
        for path in paths:
            with self.subTest(path=path.name):
                py_compile.compile(str(path), doraise=True)

    def test_example_identifiers_are_synthetic(self):
        for path in (ROOT / "examples").rglob("*"):
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="ignore")
                self.assertTrue("SYNTHETIC" in text or path.suffix.lower() == ".png")


if __name__ == "__main__":
    unittest.main()
