#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

test -f configs/deep1m_disk.config
test -f experiments/range_gt/output/deep1m_0.2632_gt.json

python3 - <<'PY'
import importlib.util
import json
import os
from pathlib import Path
import tempfile

module_path = Path("experiments/approx_alpha/alpha_sweep.py")
spec = importlib.util.spec_from_file_location("alpha_sweep", module_path)
alpha_sweep = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(alpha_sweep)


def write_probe(path: Path, prune: int, timing: int) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "if sys.argv[1:] == ['--print-build-info']:\n"
        "    print(json.dumps({\n"
        f"        'DJ_ENABLE_PRUNE_BREAKDOWN_STATS': {prune},\n"
        f"        'DJ_ENABLE_SEARCH_PHASE_TIMING': {timing},\n"
        f"        'binary_name': '{path.name}'\n"
        "    }))\n"
        "else:\n"
        "    sys.exit(3)\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o755)


with tempfile.TemporaryDirectory(prefix="approx_alpha_build_info.") as tmp:
    tmp_dir = Path(tmp)
    prod = tmp_dir / "prod_probe"
    breakdown = tmp_dir / "breakdown_probe"
    write_probe(prod, 0, 0)
    write_probe(breakdown, 1, 1)
    verified = alpha_sweep.verified_build_defines(prod, breakdown)
    if verified["production_binary"]["DJ_ENABLE_PRUNE_BREAKDOWN_STATS"] != 0:
        raise SystemExit("production build-info was not read from the binary")
    if verified["breakdown_binary"]["DJ_ENABLE_SEARCH_PHASE_TIMING"] != 1:
        raise SystemExit("breakdown build-info was not read from the binary")

    wrong_prod = tmp_dir / "wrong_prod_probe"
    write_probe(wrong_prod, 1, 0)
    try:
        alpha_sweep.verified_build_defines(wrong_prod, breakdown)
    except alpha_sweep.HarnessError as exc:
        if "DJ_ENABLE_PRUNE_BREAKDOWN_STATS" not in str(exc):
            raise
    else:
        raise SystemExit("wrong production build-info was accepted")

if alpha_sweep.alpha_is_baseline("1") is not True:
    raise SystemExit("alpha_is_baseline did not accept '1'")
if alpha_sweep.alpha_is_baseline("1.00") is not True:
    raise SystemExit("alpha_is_baseline did not accept '1.00'")
if alpha_sweep.alpha_is_baseline("1.5") is not False:
    raise SystemExit("alpha_is_baseline accepted '1.5'")

flags = alpha_sweep.compiler_flags()
for marker in ("-mavx2", "-DNDEBUG", "-O3"):
    if marker not in flags:
        raise SystemExit(f"compiler_flags missing {marker}: {flags}")

deep1m_artifacts = alpha_sweep.dataset_artifact_hashes(
    "deep1m", alpha_sweep.DATASETS["deep1m"]
)
for key in (
    "config",
    "gt",
    "base",
    "query",
    "cluster",
    "metadata",
    "sqg",
):
    if key not in deep1m_artifacts:
        raise SystemExit(f"dataset artifact hashes missing {key}")
if deep1m_artifacts["base"]["sha256"] == deep1m_artifacts["gt"]["sha256"]:
    raise SystemExit("base and gt hashes should be distinct artifacts")

gt_data = json.loads(alpha_sweep.DATASETS["deep1m"]["gt"].read_text(encoding="utf-8"))
index_prefix = gt_data["index"]["prefix"]
for key, suffix in (
    ("cluster", "_diskrange/_cluster_file.bin"),
    ("metadata", "_diskrange/_metadata_file.bin"),
    ("sqg", "_diskrange/_sqg_file.bin"),
):
    if not deep1m_artifacts[key]["path"].endswith(index_prefix + suffix):
        raise SystemExit(f"{key} path is not derived from GT index prefix")

git_state = alpha_sweep.git_state()
for key in (
    "git_dirty",
    "git_status_porcelain",
    "git_tracked_diff_sha256",
    "git_untracked_files",
    "git_untracked_files_sha256",
):
    if key not in git_state:
        raise SystemExit(f"git_state missing {key}")
PY

python3 experiments/approx_alpha/alpha_sweep.py --self-test \
  --datasets deep1m \
  --prod-alphas 1.0,1.5 \
  --breakdown-alphas 1.0,1.5
