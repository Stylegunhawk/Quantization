import csv
import subprocess
import sys
from pathlib import Path

DIR = Path(__file__).parent


def _wav_is_valid(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def test_kokoro_runs_and_logs():
    subprocess.run([sys.executable, str(DIR / "run_kokoro.py")], check=True, cwd=DIR)
    assert _wav_is_valid(DIR / "output_kokoro.wav")
    rows = list(csv.DictReader((DIR / "results.csv").open()))
    assert any(r["model"] == "kokoro-82m" for r in rows)


def test_piper_runs_and_logs():
    subprocess.run([sys.executable, str(DIR / "run_piper.py")], check=True, cwd=DIR)
    assert _wav_is_valid(DIR / "output_piper.wav")
    rows = list(csv.DictReader((DIR / "results.csv").open()))
    assert any(r["model"] == "piper" for r in rows)


def test_inflect_runs_and_logs():
    subprocess.run([sys.executable, str(DIR / "run_inflect.py")], check=True, cwd=DIR)
    assert _wav_is_valid(DIR / "output_inflect.wav")
    rows = list(csv.DictReader((DIR / "results.csv").open()))
    assert any(r["model"] == "inflect-micro-v2" for r in rows)


if __name__ == "__main__":
    test_kokoro_runs_and_logs()
    test_piper_runs_and_logs()
    test_inflect_runs_and_logs()
    print("smoke test passed")
