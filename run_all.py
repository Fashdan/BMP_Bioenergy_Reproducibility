from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def copy_required_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def run(cmd: list[str], cwd: Path) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the BMP analysis from the raw source table.")
    parser.add_argument("--output-dir", default="reproduction_output", help="New clean output directory.")
    parser.add_argument("--quick-check", action="store_true", help="Run a reduced smoke test with one outer fold.")
    parser.add_argument("--full", action="store_true", help="Run the complete reproducible workflow. This is also the default.")
    parser.add_argument("--grid-jobs", type=int, default=None, help="Parallel jobs for the inner grid search.")
    args = parser.parse_args()
    if args.quick_check and args.full:
        raise SystemExit("Use either --quick-check or --full, not both.")
    release_root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = release_root / output_dir
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_required_file(release_root / "data" / "source_provenance" / "table_complete.csv", output_dir / "table_complete.csv")
    copy_required_file(release_root / "scripts" / "run_model_validation_and_explainability.py", output_dir / "scripts" / "run_model_validation_and_explainability.py")
    copy_required_file(release_root / "scripts" / "run_two_stage_target_sensitivity.py", output_dir / "scripts" / "run_two_stage_target_sensitivity.py")
    cmd = [sys.executable, str(output_dir / "scripts" / "run_model_validation_and_explainability.py")]
    if args.grid_jobs is not None:
        cmd += ["--grid-jobs", str(args.grid_jobs)]
    if args.quick_check:
        cmd += ["--n-repeats", "1", "--max-outer", "1", "--scenario", "C_BMP_DM_all_features", "--only-nested", "--skip-shap", "--skip-lignin-stats"]
    run(cmd, cwd=output_dir)
    if not args.quick_check:
        run([sys.executable, str(output_dir / "scripts" / "run_two_stage_target_sensitivity.py")], cwd=output_dir)
    metadata = {
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "mode": "quick-check" if args.quick_check else "full",
        "output_dir": str(output_dir),
        "source_table": str((output_dir / "table_complete.csv").resolve()),
        "principle": "Scripts start from table_complete.csv and do not read prior result tables as analytical inputs.",
    }
    (output_dir / "reproduction_run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if args.quick_check:
        print("Status: PASS")


if __name__ == "__main__":
    main()
