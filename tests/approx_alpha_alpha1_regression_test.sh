#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python3 - <<'PY'
import importlib.util
from pathlib import Path
import subprocess
import tempfile

module_path = Path("experiments/approx_alpha/alpha_sweep.py")
spec = importlib.util.spec_from_file_location("alpha_sweep", module_path)
alpha_sweep = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(alpha_sweep)

config = Path("configs/deep1m_disk.config")
gt = Path("experiments/range_gt/output/deep1m_0.2632_gt.json")
prod_bin = Path("build/approx_alpha_search_stats_prod")
breakdown_bin = Path("build/approx_alpha_search_stats_breakdown")
for path in (config, gt, prod_bin, breakdown_bin):
    if not path.exists():
        raise SystemExit(f"missing required path: {path}")


def run(binary: Path, cfg: Path) -> str:
    proc = subprocess.run(
        [str(binary.resolve()), "--config", str(cfg.resolve()), "--gt", str(gt.resolve())],
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"{binary} failed:\n{proc.stderr}")
    return proc.stdout


def compare(name: str, binary: Path, fields: list[str]) -> None:
    with tempfile.TemporaryDirectory(prefix=f"approx_alpha_{name}.") as tmp:
        explicit = Path(tmp) / "deep1m_alpha1.config"
        alpha_sweep.write_temp_config(config, "1.0", explicit)
        default_values = alpha_sweep.parse_run_output(run(binary, config), fields)
        explicit_values = alpha_sweep.parse_run_output(run(binary, explicit), fields)
        mismatches = [
            key
            for key in fields
            if key not in {"qps", "search_time"}
            and default_values[key] != explicit_values[key]
        ]
        if mismatches:
            raise SystemExit(
                f"{name} default config differs from explicit approx_alpha=1.0: "
                + ", ".join(mismatches)
            )


compare("production", prod_bin, alpha_sweep.PROD_FIELDS)
compare("breakdown", breakdown_bin, alpha_sweep.BREAKDOWN_FIELDS)
PY
