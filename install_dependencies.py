#!/usr/bin/env python3
"""Install the Python environment for the MethodsNET text analysis materials."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import venv
from pathlib import Path


if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10 or newer is required for these course materials.")


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(command: list) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"\n+ {printable}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def module_available(python: Path, module_name: str) -> bool:
    code = (
        "import importlib.util, sys; "
        f"sys.exit(0 if importlib.util.find_spec({module_name!r}) else 1)"
    )
    return subprocess.run([python, "-c", code], cwd=ROOT).returncode == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a local .venv and install the dependencies for the "
            "advanced text analysis summer school notebooks."
        )
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Install only the base notebook dependencies, without torch/transformers.",
    )
    parser.add_argument(
        "--no-venv",
        action="store_true",
        help="Install into the currently active Python environment instead of .venv.",
    )
    parser.add_argument(
        "--skip-spacy-model",
        action="store_true",
        help="Skip downloading en_core_web_sm. The notebooks still run with a blank tokenizer.",
    )
    parser.add_argument(
        "--skip-kernel",
        action="store_true",
        help="Skip registering the environment as a Jupyter kernel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.no_venv:
        python = Path(sys.executable)
        print(f"Using active Python environment: {python}")
    else:
        if not VENV_DIR.exists():
            print(f"Creating virtual environment: {VENV_DIR}")
            venv.EnvBuilder(with_pip=True).create(VENV_DIR)
        python = venv_python()
        if not python.exists():
            raise SystemExit(f"Could not find the virtual-environment Python at {python}")
        print(f"Using virtual environment: {VENV_DIR}")

    requirements = (
        ROOT / "materials" / "requirements.txt"
        if args.base_only
        else ROOT / "materials" / "requirements-transformers.txt"
    )

    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    run([python, "-m", "pip", "install", "-r", requirements])

    if args.skip_spacy_model:
        print("\nSkipping spaCy language model download.")
    elif module_available(python, "en_core_web_sm"):
        print("\nspaCy model en_core_web_sm is already installed.")
    else:
        run([python, "-m", "spacy", "download", "en_core_web_sm"])

    if args.skip_kernel:
        print("\nSkipping Jupyter kernel registration.")
    else:
        kernel_name = "methodsnet-2026-base" if args.base_only else "methodsnet-2026"
        display_name = (
            "Python (MethodsNET 2026 base)"
            if args.base_only
            else "Python (MethodsNET 2026)"
        )
        run(
            [
                python,
                "-m",
                "ipykernel",
                "install",
                "--user",
                "--name",
                kernel_name,
                "--display-name",
                display_name,
            ]
        )

    print("\nInstallation complete.")
    if args.no_venv:
        print("Start Jupyter with: jupyter lab materials/notebooks")
    elif os.name == "nt":
        print(r"Activate with: .venv\Scripts\activate")
        print("Then start Jupyter with: jupyter lab materials/notebooks")
    else:
        print("Activate with: source .venv/bin/activate")
        print("Then start Jupyter with: jupyter lab materials/notebooks")


if __name__ == "__main__":
    main()
