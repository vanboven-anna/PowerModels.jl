# solve-time quartiles, dc_ac_pf vs baseline_acpf, as a LaTeX table: python timing_comparison.py [pert_name]

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TESTCASE_PATH
from violation_distributions import dataset_times


def _case_order(name):
    """Sort key: numeric size in the case name, then the name."""
    m = re.search(r"\d+", name)
    return (int(m.group()) if m else 0, name)


def _quartiles(vals):
    """(25th, 75th) percentile."""
    return tuple(float(q) for q in np.percentile(list(vals), [25, 75]))


def case_timing(case_name, pert_name="extreme_pert"):
    """(25th, 75th) percentiles of dc_ac_pf time, baseline_acpf time, and per-datapoint dc_ac_pf/baseline_acpf."""
    data_dir = TESTCASE_PATH / "data" / case_name / pert_name
    dc = dataset_times(data_dir / "dc_ac_pf" / "dataset.h5")
    base = dataset_times(data_dir / "baseline_acpf" / "dataset.h5")
    ratios = [dc[d] / base[d] for d in dc if d in base and base[d] > 0]
    return _quartiles(dc.values()), _quartiles(base.values()), _quartiles(ratios)


def write_tex(rows, save_path):
    """LaTeX tabular from {case_name: ((dc_25, dc_75), (base_25, base_75), (ratio_25, ratio_75))}."""
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Test case & \multicolumn{2}{c}{dc\_ac\_pf (s)} & \multicolumn{2}{c}{baseline\_acpf (s)} "
        r"& \multicolumn{2}{c}{dc\_ac\_pf / baseline\_acpf} \\",
        r"\cline{2-3} \cline{4-5} \cline{6-7}",
        r" & 25th & 75th & 25th & 75th & 25th & 75th \\",
        r"\hline",
    ]
    for name, (dc, base, ratio) in rows.items():
        cells = [f"{v:.4f}" for v in dc + base] + [f"{v:.2f}" for v in ratio]
        lines.append(f"{name.replace('_', chr(92) + '_')} & " + " & ".join(cells) + " \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines) + "\n")


def main(pert_name="extreme_pert"):
    fig_dir = TESTCASE_PATH / "figures"
    cases = sorted((p.name for p in fig_dir.iterdir() if p.is_dir() and (fig_dir / p.name / pert_name).is_dir()),
                   key=_case_order)
    rows = {c: case_timing(c, pert_name) for c in cases}
    out = fig_dir / pert_name / "timing_comparison" / "tex_script.txt"
    write_tex(rows, out)
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:2]))
