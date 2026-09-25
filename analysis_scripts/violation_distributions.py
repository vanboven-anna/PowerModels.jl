# vm/qg/va/sm violation distributions, dc_ac_pf vs baseline_acpf: python violation_distributions.py <case_name> [pert_name]

import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TESTCASE_PATH

THRESHOLD = 1e-4
KEYWORDS = ("vm", "qg", "va", "sm")
DAT_TYPES = ("dc_ac_pf", "baseline_acpf")
MAX_VIOLATION = {"vm": 0.5, "qg": 20.0}


def _s(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def _native(v):
    return _s(v) if isinstance(v, bytes) else v


def iter_datapoints(h5_path):
    """Yield PowerModels case dicts from dataset.h5 (port of unpack_data.jl)."""
    with h5py.File(h5_path, "r") as f:
        top = json.loads(_s(f.attrs["toplevel_json"]))
        comps = [_s(c) for c in f.attrs["components"]]
        statics = {}
        for c in comps:
            if c in f:
                ids = [_s(i) for i in f[c].attrs["ids"]]
                statics[c] = (ids, json.loads(_s(f[c].attrs["static_json"])))
        for row, dp in enumerate(f["datapoint"][:]):
            case = dict(top)
            case["datapoint"] = int(dp)
            for c in comps:
                if c not in f:
                    case[c] = {}
                    continue
                ids, static = statics[c]
                comp = {i: {fld: byid[i] for fld, byid in static.items() if i in byid} for i in ids}
                vg = f[c]["varying"]
                for fld in vg:
                    if fld.endswith("__present"):
                        continue
                    d = vg[fld]
                    kind = _s(d.attrs["kind"])
                    fids = [_s(i) for i in d.attrs["ids"]]
                    if d.attrs["masked"] == 1:
                        present = vg[fld + "__present"][:, row] != 0
                    else:
                        present = np.ones(len(fids), dtype=bool)
                    vals = d[:, :, row].T if kind == "vec" else d[:, row]
                    for j, i in enumerate(fids):
                        if not present[j]:
                            continue
                        if kind == "json":
                            comp[i][fld] = json.loads(_s(vals[j]))
                        elif kind == "vec":
                            comp[i][fld] = vals[j].tolist()
                        else:
                            comp[i][fld] = _native(vals[j].item() if hasattr(vals[j], "item") else vals[j])
                case[c] = comp
            yield case


def vm_violations(case):
    """Per-bus vm bound violation; 0 at non-PQ buses (gen/slack/out of service)."""
    out = []
    for b in case["bus"].values():
        if b["bus_type"] != 1:
            out.append(0.0)
        else:
            out.append(max(b["vm"] - b["vmax"], b["vmin"] - b["vm"], 0.0))
    return out


def qg_violations(case):
    """Per-gen qg bound violation; 0 for out-of-service and slack gens."""
    out = []
    for g in case["gen"].values():
        if g["gen_status"] == 0 or case["bus"][str(g["gen_bus"])]["bus_type"] == 3:
            out.append(0.0)
        else:
            out.append(max(g["qg"] - g["qmax"], g["qmin"] - g["qg"], 0.0))
    return out


def va_violations(case):
    """Per-branch angle-difference bound violation; 0 if out of service."""
    out = []
    for br in case["branch"].values():
        if br["br_status"] == 0:
            out.append(0.0)
            continue
        vad = case["bus"][str(br["f_bus"])]["va"] - case["bus"][str(br["t_bus"])]["va"]
        out.append(max(vad - br["angmax"], br["angmin"] - vad, 0.0))
    return out


def _branch_sm(case, br):
    """max(|S_from|, |S_to|) of a branch, from bus vm/va."""
    y = 1 / complex(br["br_r"], br["br_x"])
    t = br["tap"] * np.exp(1j * br["shift"])
    yff = (y + complex(br["g_fr"], br["b_fr"])) / br["tap"] ** 2
    yft = -y / np.conj(t)
    ytf = -y / t
    ytt = y + complex(br["g_to"], br["b_to"])
    fb, tb = case["bus"][str(br["f_bus"])], case["bus"][str(br["t_bus"])]
    vf = fb["vm"] * np.exp(1j * fb["va"])
    vt = tb["vm"] * np.exp(1j * tb["va"])
    sf = vf * np.conj(yff * vf + yft * vt)
    st = vt * np.conj(ytf * vf + ytt * vt)
    return max(abs(sf), abs(st))


def sm_violations(case):
    """Per-branch thermal (rate_a) violation; 0 if out of service or unrated."""
    out = []
    for br in case["branch"].values():
        if br["br_status"] == 0 or "rate_a" not in br:
            out.append(0.0)
        else:
            out.append(max(_branch_sm(case, br) - br["rate_a"], 0.0))
    return out


VIOLATION_FUNCS = {"vm": vm_violations, "qg": qg_violations, "va": va_violations, "sm": sm_violations}


def case_violations(case, keyword):
    """Violation amount per component of `case` for keyword vm/qg/va/sm."""
    return VIOLATION_FUNCS[keyword](case)


def dataset_violations_by_datapoint(h5_path, keyword, threshold=THRESHOLD):
    """{datapoint: violations >= threshold} for datapoints with at least one."""
    out = {}
    for case in iter_datapoints(h5_path):
        vals = [v for v in case_violations(case, keyword) if v >= threshold]
        if vals:
            out[case["datapoint"]] = vals
    return out


def dataset_violations(h5_path, keyword, threshold=THRESHOLD):
    """Concatenated violations over all datapoints in a dataset.h5, dropping those below threshold."""
    by_dp = dataset_violations_by_datapoint(h5_path, keyword, threshold)
    return [v for vals in by_dp.values() for v in vals]


def bad_datapoints(by_dp):
    """Datapoints, in any dataset, with a vm/qg violation above MAX_VIOLATION."""
    bad = set()
    for kw, cap in MAX_VIOLATION.items():
        for dps in by_dp[kw].values():
            bad |= {d for d, vs in dps.items() if max(vs) > cap}
    return bad


def dataset_times(h5_path):
    """{datapoint: total solve time} from a dataset.h5."""
    with h5py.File(h5_path, "r") as f:
        total = f["acpf_time"][:] + (f["dcopf_time"][:] if "dcopf_time" in f else 0.0)
        return dict(zip(f["datapoint"][:].tolist(), total.tolist()))


def plot_distributions(data, keyword, save_path=None):
    """Overlaid kdeplot of {label: list}; keyword names the quantity."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    skipped = []
    for label, vals in data.items():
        vals = np.asarray(vals, dtype=float)
        if len(vals) < 2 or np.ptp(vals) == 0:
            skipped.append(f"{label} (n={len(vals)})")
            continue
        sns.kdeplot(x=vals, ax=ax, label=f"{label} (n={len(vals)})", fill=True, alpha=0.3, cut=0, common_norm=False)
    if skipped:
        ax.text(0.5, 0.5, "too few violations for KDE:\n" + "\n".join(skipped), transform=ax.transAxes,
                ha="center", va="center")
    ax.set_xlabel(f"{keyword} violation")
    ax.set_ylabel("density")
    ax.set_title(f"{keyword} violation distribution")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_scatter(xs, ys, xlabel="x", ylabel="y", title=None, save_path=None):
    """Scatter of xs[i] vs ys[i], with a y = x reference line."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(xs, ys, s=10, alpha=0.5)
    lo, hi = min(min(xs), min(ys)), max(max(xs), max(ys))
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def write_comparison_json(by_dp, times, save_path, n_excluded=0):
    """Mean violation (samples in the distributions) and mean solve time, over datapoints in both h5s.

    by_dp: {keyword: {dat_type: {datapoint: [violations]}}}; times: {dat_type: {datapoint: time}}.
    """
    shared = set.intersection(*(set(t) for t in times.values()))
    out = {
        "threshold": THRESHOLD,
        "n_excluded_datapoints": n_excluded,
        "n_shared_datapoints": len(shared),
        "mean_time": {t: float(np.mean([tm[d] for d in shared])) if shared else None for t, tm in times.items()},
    }
    for kw, per_type in by_dp.items():
        out[kw] = {}
        for t, dps in per_type.items():
            vals = [v for d, vs in dps.items() if d in shared for v in vs]
            out[kw][t] = {
                "n_violations": len(vals),
                "mean_violation": float(np.mean(vals)) if vals else None,
            }
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)


def plot_case(case_name, pert_name="extreme_pert"):
    """Distribution plots for vm/qg/va/sm and a solve-time scatter, dc_ac_pf vs baseline_acpf; drops bad datapoints."""
    data_dir = TESTCASE_PATH / "data" / case_name / pert_name
    out_dir = TESTCASE_PATH / "figures" / case_name / pert_name
    out_dir.mkdir(parents=True, exist_ok=True)
    h5s = {t: data_dir / t / "dataset.h5" for t in DAT_TYPES}

    by_dp = {kw: {t: dataset_violations_by_datapoint(p, kw) for t, p in h5s.items()} for kw in KEYWORDS}
    bad = bad_datapoints(by_dp)
    by_dp = {kw: {t: {d: vs for d, vs in dps.items() if d not in bad} for t, dps in per_type.items()}
             for kw, per_type in by_dp.items()}
    for kw in KEYWORDS:
        data = {t: [v for vs in dps.values() for v in vs] for t, dps in by_dp[kw].items()}
        plot_distributions(data, kw, out_dir / f"{kw}_distribution.png")

    times = {t: {d: tm for d, tm in dataset_times(p).items() if d not in bad} for t, p in h5s.items()}
    write_comparison_json(by_dp, times, out_dir / "violation_comparison.json", len(bad))
    shared = sorted(set(times["dc_ac_pf"]) & set(times["baseline_acpf"]))
    plot_scatter([times["dc_ac_pf"][d] for d in shared], [times["baseline_acpf"][d] for d in shared],
                 xlabel="dc_ac_pf time (s)", ylabel="baseline_acpf time (s)",
                 title="solve time", save_path=out_dir / "solve_time_scatter.png")
    return out_dir


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python violation_distributions.py <case_name> [pert_name]")
    print(plot_case(*sys.argv[1:3]))
