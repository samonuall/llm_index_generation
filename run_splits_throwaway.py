#!/usr/bin/env python3
"""Throwaway: run main.py over a list of splits, resetting state between each.

Edit SPLITS and MAIN_ARGS below. Run with:
    uv run python run_splits_throwaway.py

All terminal output (this script's prints + main.py subprocess output) is also
mirrored to LOG_FILE so a long batch run can be reviewed later.
"""
import atexit
import builtins
import datetime
import shutil
import subprocess
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent

# ── EDIT ME ──────────────────────────────────────────────────────────────────
SPLITS = [
    "stack_exchange",
    "clinical_trial",
    "paper_retrieval",
    "legal_qa",
    "tip_of_the_tongue",
    "set_operation_entity_retrieval",
    "theorem_retrieval",
    "code_retrieval",
]

# Args passed to main.py for every split (split itself is appended automatically)
MAIN_ARGS = [
    "--agent", "analysis_code_agent",
    "--loops", "3",
    "--condition", "agent",
]
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CODE   = ROOT / "src" / "agents" / "DEFAULT_CODE_COPY_FROM_HERE.txt"
PREPROCESS_PY  = ROOT / "src" / "agents" / "analysis_code_agent" / "preprocess.py"
BM25_CACHE     = ROOT / ".bm25_cache"
AGENT_BM25_CACHE = ROOT / "src" / "agents" / "analysis_code_agent" / ".bm25_cache"
SERVER_PORT    = 8765

# ── output mirroring ─────────────────────────────────────────────────────────
LOG_DIR = ROOT / "run_splits_logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"run_splits_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"

_log_handle = open(LOG_FILE, "a", buffering=1)  # line-buffered text mode
atexit.register(_log_handle.close)

_orig_print = builtins.print

def _tee_print(*args, **kwargs):
    """Replacement for builtins.print that mirrors output to LOG_FILE."""
    _orig_print(*args, **kwargs)
    kwargs.pop("file", None)
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs, file=_log_handle)

builtins.print = _tee_print


def _tee_subprocess(cmd: list[str], cwd: pathlib.Path) -> int:
    """Run `cmd`, streaming combined stdout/stderr to both terminal and LOG_FILE."""
    proc = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        _log_handle.write(line)
        _log_handle.flush()
    return proc.wait()
# ─────────────────────────────────────────────────────────────────────────────


def reset_preprocess() -> None:
    shutil.copyfile(DEFAULT_CODE, PREPROCESS_PY)
    print(f">>> reset {PREPROCESS_PY.relative_to(ROOT)} <- DEFAULT_CODE_COPY_FROM_HERE.txt")


def clean_bm25_cache() -> None:
    for d in (BM25_CACHE, AGENT_BM25_CACHE):
        if d.exists():
            shutil.rmtree(d)
            print(f">>> removed {d.relative_to(ROOT)}/")


def kill_server_port() -> None:
    try:
        out = subprocess.check_output(
            ["lsof", "-i", f":{SERVER_PORT}", "-t"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return
    for pid in out.splitlines():
        subprocess.run(["kill", "-9", pid.strip()], check=False)
        print(f">>> killed leftover pid {pid.strip()} on port {SERVER_PORT}")


def split_data_exists(split: str) -> bool:
    base = ROOT / "data" / split
    return all((base / f).exists() for f in (
        "documents.jsonl", "validation_queries.jsonl", "evaluation_queries.jsonl",
    ))


def run_split(split: str) -> int:
    cmd = ["uv", "run", "python", "main.py", *MAIN_ARGS, "--split", split]
    print(f">>> {' '.join(cmd)}")
    return _tee_subprocess(cmd, ROOT)


def main() -> None:
    print(f">>> logging output to {LOG_FILE.relative_to(ROOT)}")
    failed: list[str] = []
    for split in SPLITS:
        print(f"\n{'='*60}\n  SPLIT: {split}\n{'='*60}")

        if not split_data_exists(split):
            print(f"!!! data/{split}/ missing — skipping. Download with:")
            print(f"    uv run python src/evaluation/scripts/get_data.py --split {split}")
            failed.append(f"{split} (no data)")
            continue

        kill_server_port()
        clean_bm25_cache()
        reset_preprocess()

        rc = run_split(split)
        if rc != 0:
            print(f"!!! split '{split}' failed (rc={rc}) — continuing")
            failed.append(f"{split} (rc={rc})")

    # Final reset so the working tree is left clean
    kill_server_port()
    reset_preprocess()

    print(f"\n{'='*60}\n  DONE\n{'='*60}")
    if failed:
        print("Failed/skipped:")
        for f in failed:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
