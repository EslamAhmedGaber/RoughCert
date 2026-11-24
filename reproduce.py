"""Rebuild and independently check the tables and figures from the deposited results."""
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    (root / "audit").mkdir(exist_ok=True)
    (root / "manuscript/tables").mkdir(parents=True, exist_ok=True)
    scripts = (
        "build_v3_tables.py",
        "build_v3_figures.py",
        "audit_v3_tables.py",
        "audit_v3_figures.py",
    )
    for script in scripts:
        subprocess.run([sys.executable, "-B", str(root / "scripts" / script)], cwd=root, check=True)


if __name__ == "__main__":
    main()
