#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from packaging.version import Version

EXPECTED_VLLM_VERSION = "0.12.0"


def resolve_installed_vllm_root(package_root: str | None) -> Path:
    if package_root is not None:
        root = Path(package_root).resolve()
        if not (root / "__init__.py").exists():
            raise RuntimeError(f"{root} does not look like a vllm package root")
        return root

    import vllm

    installed_version = Version(vllm.__version__)
    if Version(installed_version.public) != Version(EXPECTED_VLLM_VERSION):
        raise RuntimeError(
            f"Expected vllm=={EXPECTED_VLLM_VERSION}, found vllm=={vllm.__version__}"
        )
    return Path(vllm.__file__).resolve().parent


def resolve_overlay_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    candidates = [
        script_dir / "vllm_overlay" / "vllm",
    ]
    for overlay_root in candidates:
        if overlay_root.exists():
            return overlay_root

    searched = ", ".join(str(path) for path in candidates)
    raise RuntimeError(f"Missing overlay directory. Checked: {searched}")


def install_overlay(overlay_root: Path, installed_root: Path) -> list[str]:
    copied_files: list[str] = []
    for src in sorted(overlay_root.rglob("*")):
        if not src.is_file():
            continue
        relpath = src.relative_to(overlay_root)
        dst = installed_root / relpath
        if not dst.exists():
            raise RuntimeError(f"Target file does not exist in installed vllm: {dst}")
        shutil.copy2(src, dst)
        copied_files.append(str(relpath))
    return copied_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install the local vLLM entropy overlay on top of a vllm 0.12.0-based install."
    )
    parser.add_argument("--package-root", help="Path to the installed `vllm` package root.")
    args = parser.parse_args()

    overlay_root = resolve_overlay_root()
    installed_root = resolve_installed_vllm_root(args.package_root)
    copied_files = install_overlay(overlay_root, installed_root)

    print("Installed vLLM entropy overlay:")
    for relpath in copied_files:
        print(f"  - {relpath}")


if __name__ == "__main__":
    main()
