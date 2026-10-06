"""Parameter sweeps for the corrected nanonet transport model.

Conventions
-----------
* Activation voltage Va controls gating only.
* Junction resistance is fixed independently at node_resistance_ohm.
* V_T is the first sampled source-drain percolation voltage (V_T == V_perc)
  and is held fixed while fitting A and zeta in I = A (V - V_T)^zeta.
* N/density sweeps keep the domain and connection radius fixed.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SweepConfig:
    """Configuration shared by all standard parameter sweeps."""

    L: float = 1.0
    N: int = 500
    fv: float = 0.0
    mu_a: float = 6.0
    std_a: float = 3.0
    va_min: float = 0.0
    va_max: float = 20.0
    edge_k: float = 2.0e10
    node_resistance_ohm: float = 3.5e9
    connection_radius: float = 0.15
    source_frac: float = 0.15
    drain_frac: float = 0.15
    strict_N: bool = False
    void_radius: float = 0.08

    V_start: float = 0.0
    V_max: float = 16.0
    V_step: float = 0.5
    fit_window: float = 10.0

    seeds: list = field(default_factory=lambda: [41, 51, 61, 71, 81])
    max_workers: int = 2
    results_root: str = "IV_results"

    sigma_values: list = field(default_factory=lambda: [1.0, 3.0, 5.0, 7.0])
    fixed_mean: float = 8.0

    mean_values: list = field(default_factory=lambda: [4.0, 6.0, 8.0, 10.0])
    sigma_values_mean: list = field(default_factory=lambda: [1.0, 3.0, 5.0, 7.0])

    N_values_cross: list = field(default_factory=lambda: [200, 400, 600, 800])
    mean_values_cross: list = field(default_factory=lambda: [4.0, 6.0, 8.0, 10.0])
    fixed_std_cross: float = 3.0

    N_values: list = field(default_factory=lambda: [200, 400, 600, 800])
    fixed_mean_N: float = 6.0
    fixed_std_N: float = 3.0

    void_fractions: list = field(
        default_factory=lambda: [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    )
    N_voids: int = 500
    mu_a_voids: float = 6.0
    std_a_voids: float = 3.0

    def seed_for(self, i: int) -> int:
        return self.seeds[i % len(self.seeds)]


def connection_radius_for_N(n: int, base_radius: float = 0.15) -> float:
    """Density convention: r_c is fixed as N changes."""
    del n
    return float(base_radius)


scaled_radius_for_N = connection_radius_for_N


def _fit_power_law(
    v_arr,
    i_arr,
    *,
    V_T=None,
    v_transition=None,
    fit_window: float = 10.0,
    v_step: float | None = None,
) -> dict:
    """Fit I = A (V - V_T)^zeta with V_T fixed at percolation."""
    del v_step
    nan = float("nan")
    v = np.asarray(v_arr, float)
    current = np.asarray(i_arr, float)

    if V_T is None or not np.isfinite(V_T):
        positive = np.flatnonzero(current > 0)
        if len(positive) == 0:
            return dict(
                success=False, V_T=nan, zeta=nan, A=nan, R2=nan,
                reason="no_percolation",
            )
        V_T = float(v[positive[0]])
    else:
        V_T = float(V_T)

    if (
        v_transition is not None
        and np.isfinite(v_transition)
        and float(v_transition) > V_T
    ):
        V_stop = float(v_transition)
    else:
        V_stop = min(float(v[-1]), V_T + float(fit_window))

    mask = (v > V_T) & (v <= V_stop) & (current > 0)
    V_fit = v[mask]
    I_fit = current[mask]

    if len(V_fit) < 4:
        return dict(
            success=False, V_T=V_T, zeta=nan, A=nan, R2=nan,
            V_stop=V_stop, n_points=int(len(V_fit)),
            reason=f"too_few_points({len(V_fit)})",
        )

    x = np.log(V_fit - V_T)
    y = np.log(I_fit)
    design = np.vstack([x, np.ones_like(x)]).T
    (zeta, logA), *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = zeta * x + logA
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else nan
    A = float(np.exp(logA))

    if not np.isfinite(zeta) or not np.isfinite(R2) or not np.isfinite(A):
        return dict(
            success=False, V_T=V_T, zeta=float(zeta), A=A, R2=R2,
            V_stop=V_stop, n_points=int(len(V_fit)), reason="nonfinite",
        )
    if R2 < 0.80:
        return dict(
            success=False, V_T=V_T, zeta=float(zeta), A=A, R2=float(R2),
            V_stop=V_stop, n_points=int(len(V_fit)),
            reason=f"poor_fit_R2={R2:.3f}",
        )
    if not (0.1 < zeta < 10.0):
        return dict(
            success=False, V_T=V_T, zeta=float(zeta), A=A, R2=float(R2),
            V_stop=V_stop, n_points=int(len(V_fit)),
            reason="zeta_out_of_range",
        )
    return dict(
        success=True, V_T=V_T, zeta=float(zeta), A=A, R2=float(R2),
        V_stop=V_stop, n_points=int(len(V_fit)), reason="ok",
    )


def _transition_voltage_from_evo(evo: dict) -> float | None:
    """First voltage with >=90% of network nodes active."""
    rows = evo.get("rows") or []
    if not rows:
        return None
    n_total = evo.get("n_total_nodes") or max(
        (row.get("activated_nodes", 0) or 0 for row in rows),
        default=0,
    )
    if not n_total:
        return None
    target = 0.90 * float(n_total)
    ordered = sorted(rows, key=lambda row: float(row["V"]))
    for row in ordered:
        if float(row.get("activated_nodes", 0)) >= target:
            return float(row["V"])
    return None


def _build_and_run(
    N: int,
    mu_a: float,
    std_a: float,
    seed: int,
    cfg: SweepConfig,
    fv: float = 0.0,
) -> dict:
    from nanonet.analysis.sweep import sweep as _sweep
    from nanonet.core.network import NanoparticleNetwork

    connection_radius = connection_radius_for_N(N, cfg.connection_radius)

    net = NanoparticleNetwork(
        L=cfg.L,
        N=N,
        fv=fv,
        mu_a=mu_a,
        std_a=std_a,
        connection_radius=connection_radius,
        edge_k=cfg.edge_k,
        node_resistance_ohm=cfg.node_resistance_ohm,
        source_frac=cfg.source_frac,
        drain_frac=cfg.drain_frac,
        va_min=cfg.va_min,
        va_max=cfg.va_max,
        void_radius=cfg.void_radius,
        strict_N=cfg.strict_N,
    ).build(seed=seed)

    iv = net.iv_curve(
        V_start=cfg.V_start, V_max=cfg.V_max, V_step=cfg.V_step
    )
    evo = _sweep(
        net,
        cfg.V_start,
        cfg.V_max,
        cfg.V_step,
        effective_resistance=False,
        algebraic_connectivity=False,
    )

    node_va = np.asarray(
        [net.G.nodes[node].get("Va", net.G.nodes[node]["Vth"])
         for node in net.G.nodes()],
        float,
    )
    peak_idx = int(np.argmax(iv["currents"])) if len(iv["currents"]) else 0
    perc_V = evo.get("percolation_V")
    trans_V = _transition_voltage_from_evo(evo)
    fit = _fit_power_law(
        iv["voltages"],
        iv["currents"],
        V_T=perc_V,
        v_transition=trans_V,
        fit_window=cfg.fit_window,
    )

    return {
        "N": int(net.n_junctions),
        "N_requested": int(N),
        "mean_va_target": float(mu_a),
        "sigma_va_target": float(std_a),
        "seed": int(seed),
        "void_fraction": float(fv),
        "void_fraction_requested": float(net.requested_void_fraction),
        "void_fraction_achieved": float(net.achieved_void_fraction),
        "connection_radius_used": float(connection_radius),
        "node_resistance_ohm": float(cfg.node_resistance_ohm),
        "voltages": iv["voltages"],
        "currents": iv["currents"],
        "conductances": iv["conductances"],
        "node_va": node_va,
        "n_nodes_actual": int(net.n_junctions),
        "n_edges": int(net.G.number_of_edges()),
        "n_sources": int(len(net.source_nodes)),
        "n_drains": int(len(net.drain_nodes)),
        "sampled_mean_va": float(np.mean(node_va)) if len(node_va) else float("nan"),
        "sampled_std_va": float(np.std(node_va)) if len(node_va) else float("nan"),
        "peak_current_A": float(iv["currents"][peak_idx]) if len(iv["currents"]) else 0.0,
        "peak_voltage_V": float(iv["voltages"][peak_idx]) if len(iv["voltages"]) else float("nan"),
        "percolation_voltage_V": float(perc_V) if perc_V is not None else float("nan"),
        "edge_disjoint_pathways_at_Vperc": int(evo.get("percolation_pathways", 0)),
        "active_nodes_at_Vperc": int(evo.get("percolation_active_nodes", 0)),
        "transition_voltage_V": (
            float(trans_V) if trans_V is not None else float("nan")
        ),
        "max_conductance_S": (
            float(np.max(iv["conductances"])) if len(iv["conductances"]) else 0.0
        ),
        "fit_V_T_V": float(fit["V_T"]) if np.isfinite(fit["V_T"]) else float("nan"),
        "fit_zeta": float(fit["zeta"]) if np.isfinite(fit["zeta"]) else float("nan"),
        "fit_A": float(fit["A"]) if np.isfinite(fit["A"]) else float("nan"),
        "fit_R2": float(fit["R2"]) if np.isfinite(fit["R2"]) else float("nan"),
        "fit_success": bool(fit["success"]),
        "fit_reason": fit.get("reason", ""),
        "evolution_rows": evo["rows"],
        "evolution_fields": evo["fieldnames"],
    }


def _worker(args):
    N, mu_a, std_a, seed, cfg, fv = args
    return _build_and_run(N, mu_a, std_a, seed, cfg, fv)


def _pool_workers(cfg: SweepConfig, n_tasks: int) -> int:
    return max(1, min(int(cfg.max_workers), int(n_tasks)))


def _run_tasks(tasks, cfg: SweepConfig) -> list[dict]:
    if not tasks:
        return []
    if _pool_workers(cfg, len(tasks)) == 1:
        return [_worker(task) for task in tasks]
    with Pool(
        processes=_pool_workers(cfg, len(tasks)),
        maxtasksperchild=4,
    ) as pool:
        return list(pool.imap_unordered(_worker, tasks, chunksize=1))


_ARRAY_KEYS = {
    "voltages", "currents", "conductances", "node_va",
    "evolution_rows", "evolution_fields",
}


def _scalar_row(result: dict) -> dict:
    return {key: value for key, value in result.items() if key not in _ARRAY_KEYS}


def make_fit_table(results: list[dict], case_name: str) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append({
            "case": case_name,
            "N": result["N"],
            "mean_Va_target_V": result["mean_va_target"],
            "sigma_Va_target_V": result["sigma_va_target"],
            "seed": result["seed"],
            "connection_radius": result["connection_radius_used"],
            "node_resistance_ohm": result["node_resistance_ohm"],
            "void_fraction_requested": result["void_fraction_requested"],
            "void_fraction_achieved": result["void_fraction_achieved"],
            "sampled_mean_Va_V": result["sampled_mean_va"],
            "sampled_sigma_Va_V": result["sampled_std_va"],
            "percolation_voltage_V": result["percolation_voltage_V"],
            "edge_disjoint_pathways_at_Vperc": result[
                "edge_disjoint_pathways_at_Vperc"
            ],
            "transition_voltage_V": result["transition_voltage_V"],
            "fit_V_T_V": result["fit_V_T_V"],
            "fit_zeta": result["fit_zeta"],
            "fit_A": result["fit_A"],
            "fit_R2": result["fit_R2"],
            "fit_success": result["fit_success"],
            "peak_current_A": result["peak_current_A"],
            "max_conductance_S": result["max_conductance_S"],
        })
    return pd.DataFrame(rows)


def aggregate_fit_table(
    results: list[dict], group_key: str, group_col: str
) -> pd.DataFrame:
    rows = []
    for value in sorted({result[group_key] for result in results}):
        group = [result for result in results if result[group_key] == value]
        vt = np.asarray(
            [result["fit_V_T_V"] for result in group
             if np.isfinite(result["fit_V_T_V"])],
            float,
        )
        zeta = np.asarray(
            [result["fit_zeta"] for result in group
             if result.get("fit_success") and np.isfinite(result["fit_zeta"])],
            float,
        )
        rows.append({
            group_col: value,
            "n_runs": len(group),
            "VT_mean": float(np.mean(vt)) if len(vt) else float("nan"),
            "VT_std": float(np.std(vt, ddof=1)) if len(vt) > 1 else 0.0,
            "zeta_mean": float(np.mean(zeta)) if len(zeta) else float("nan"),
            "zeta_std": float(np.std(zeta, ddof=1)) if len(zeta) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def run_sweep_vary_std(cfg: SweepConfig | None = None) -> list[dict]:
    cfg = cfg or SweepConfig()
    tasks = [
        (cfg.N, cfg.fixed_mean, sigma, seed, cfg, 0.0)
        for sigma in cfg.sigma_values
        for seed in cfg.seeds
    ]
    results = _run_tasks(tasks, cfg)
    for result in results:
        result["case"] = "vary_std"
    return sorted(results, key=lambda result: (result["sigma_va_target"], result["seed"]))


def run_sweep_vary_mean(
    cfg: SweepConfig | None = None,
) -> dict[float, list[dict]]:
    cfg = cfg or SweepConfig()
    results_by_sigma = {}
    for sigma in cfg.sigma_values_mean:
        tasks = [
            (cfg.N, mean, sigma, seed, cfg, 0.0)
            for mean in cfg.mean_values
            for seed in cfg.seeds
        ]
        results = _run_tasks(tasks, cfg)
        for result in results:
            result["case"] = f"vary_mean_sigma{sigma:g}"
        results_by_sigma[float(sigma)] = sorted(
            results, key=lambda result: (result["mean_va_target"], result["seed"])
        )
    return results_by_sigma


def run_sweep_vary_N_mean(cfg: SweepConfig | None = None) -> list[dict]:
    """Case 3: crossed N x <Va> sweep at fixed sigma and fixed r_c."""
    cfg = cfg or SweepConfig()
    tasks = [
        (N, mean, cfg.fixed_std_cross, seed, cfg, 0.0)
        for N in cfg.N_values_cross
        for mean in cfg.mean_values_cross
        for seed in cfg.seeds
    ]
    results = _run_tasks(tasks, cfg)
    for result in results:
        result["case"] = "vary_N_mean"
    return sorted(
        results,
        key=lambda result: (
            result["N_requested"], result["mean_va_target"], result["seed"]
        ),
    )


def run_sweep_vary_N(cfg: SweepConfig | None = None) -> list[dict]:
    """Case 4: vary N while keeping the domain and r_c fixed."""
    cfg = cfg or SweepConfig()
    tasks = [
        (N, cfg.fixed_mean_N, cfg.fixed_std_N, seed, cfg, 0.0)
        for N in cfg.N_values
        for seed in cfg.seeds
    ]
    results = _run_tasks(tasks, cfg)
    for result in results:
        result["case"] = "vary_N"
    return sorted(results, key=lambda result: (result["N_requested"], result["seed"]))


def run_sweep_vary_voids(cfg: SweepConfig | None = None) -> list[dict]:
    cfg = cfg or SweepConfig()
    tasks = [
        (cfg.N_voids, cfg.mu_a_voids, cfg.std_a_voids, seed, cfg, fv)
        for fv in cfg.void_fractions
        for seed in cfg.seeds
    ]
    results = _run_tasks(tasks, cfg)
    for result in results:
        result["case"] = "vary_voids"
    return sorted(
        results,
        key=lambda result: (result["void_fraction_requested"], result["seed"]),
    )


def run_all_cases(cfg: SweepConfig | None = None) -> dict:
    cfg = cfg or SweepConfig()
    return {
        "vary_std": run_sweep_vary_std(cfg),
        "vary_mean": run_sweep_vary_mean(cfg),
        "vary_N_mean": run_sweep_vary_N_mean(cfg),
        "vary_N": run_sweep_vary_N(cfg),
        "vary_voids": run_sweep_vary_voids(cfg),
    }


if __name__ == "__main__":
    from datetime import datetime
    from multiprocessing import freeze_support

    freeze_support()
    config = SweepConfig()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(config.results_root) / f"sweep_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    bundle = run_all_cases(config)
    for sweep_name, results in bundle.items():
        if isinstance(results, dict):
            for sigma, result_list in results.items():
                make_fit_table(
                    result_list, f"{sweep_name}_sigma{sigma:g}"
                ).to_csv(
                    outdir / f"{sweep_name}_sigma{sigma:g}_fit_table.csv",
                    index=False,
                )
        else:
            make_fit_table(results, sweep_name).to_csv(
                outdir / f"{sweep_name}_fit_table.csv", index=False
            )

    print(f"Results saved to {outdir}")
