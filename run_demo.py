"""PyCharm friendly entry point: select the .venv interpreter and click Run."""
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    project = Path(__file__).resolve().parent
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(project / "app.py"), "--server.address", "127.0.0.1", "--browser.gatherUsageStats", "false"], cwd=project))
