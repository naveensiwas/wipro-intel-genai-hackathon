"""Root launcher for the Streamlit app.

Usage:
    python app_launcher.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent
    app_entry = project_root / "src" / "app" / "main.py"

    if not app_entry.exists():
        print(f"Error: app entry file not found at {app_entry}", file=sys.stderr)
        return 1

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_entry)]
    return subprocess.call(cmd, cwd=str(project_root))


if __name__ == "__main__":
    raise SystemExit(main())
