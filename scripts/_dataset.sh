# Shared dataset selection for all a_run_*.sh scripts.
#
# This file is sourced by a_run_search.sh / a_run_search_stats.sh /
# a_run_indexing.sh / a_run_indexing_stats.sh. Edit DEFAULT_CONFIG and
# DEFAULT_GT below to switch the dataset for all four scripts in one place.
#
# Per-call positional override still works:
#   bash scripts/a_run_search.sh <config_path> <gt_json_path>
# In that case, the positional args take precedence over the defaults below.
#
# configs/ ships parameter templates for several datasets (deep1m/10m/100m,
# sift1m, spacev1m, ssnpp, ...). Only one example ground-truth file ships with
# the repository (deep1m); compute ground truth for any other config/radius
# with experiments/range_gt/compute_range_gt_fixed_radius.py.
DEFAULT_CONFIG="configs/deep1m_disk_fast_path.config"
DEFAULT_GT="experiments/range_gt/output/deep1m_0.2632_gt.json"
