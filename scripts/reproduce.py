"""Run the quick JumpGrad smoke audit and rebuild the frozen showcase visuals."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SMOKE_COMMAND = (
    sys.executable,
    str(ROOT / "scripts/run_jumpgrad_end_to_end.py"),
    "--smoke",
)
RENDER_COMMAND = (
    sys.executable,
    str(ROOT / "scripts/render_jumpgrad_visuals.py"),
)


def run_command(command: tuple[str, ...], label: str) -> None:
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"{label} failed", file=sys.stderr)
        if result.stdout:
            print(result.stdout, file=sys.stderr, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        raise SystemExit(result.returncode)


def main() -> None:
    run_command(SMOKE_COMMAND, "JumpGrad smoke audit")
    print("Smoke audit: PASS")
    print("  Direct AD physics gradient: zero")
    print("  CRN-FD physics gradient: nonzero")
    print("  End-to-end controller gradient: finite and nonzero")
    print("  One optimizer update: completed")

    run_command(RENDER_COMMAND, "Showcase rendering")
    print("Showcase rendering: PASS")
    print("Visuals: outputs/jumpgrad_visuals/")


if __name__ == "__main__":
    main()
