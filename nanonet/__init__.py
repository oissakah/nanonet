"""
nanonet — graph-based nanoparticle necklace network simulator
=============================================================

Quick-start
-----------
Create and analyse a network:

    from nanonet import NanoparticleNetwork, sweep, plot_iv_curve

    # Build the network
    # L=1.0  : 1 × 1 normalised domain
    # N=500  : 500 junction nodes
    # fv=0.0 : no voids
    # mu_a=6 : mean activation voltage 6 V
    # std_a=3: standard deviation 3 V
    net = NanoparticleNetwork(L=1.0, N=500, fv=0.0, mu_a=6.0, std_a=3.0).build(seed=42)

    # I-V curve
    iv = net.iv_curve(V_start=0.0, V_max=16.0, V_step=0.5)
    print(iv["voltages"], iv["currents"])
    print("Threshold voltage:", iv["threshold_voltage"], "V")

    # Microscopic analysis at one voltage
    mic = net.edge_currents(V_applied=8.0)
    print("Total current:", mic["total_current"], "A")
    print("Conducting edges:", mic["conducting_edges"])

    # Full sweep (connectivity, distribution, spectral metrics)
    result = sweep(net, V_start=0.0, V_max=16.0, V_step=0.5)

    # Export
    from nanonet.analysis import write_iv_csv, write_csv
    write_iv_csv(result, "iv_curve.csv")
    write_csv(result, "full_sweep.csv")

    # Parameter sweeps
    from nanonet.sweeps import SweepConfig, run_sweep_vary_std, run_all_cases
    cfg = SweepConfig(N=500, mu_a=6.0, std_a=3.0,
                      sigma_values=[1.0, 3.0, 5.0, 7.0])
    results = run_sweep_vary_std(cfg)
"""

from nanonet.core.network import NanoparticleNetwork
from nanonet.analysis.sweep import (
    sweep,
    count_edge_disjoint_pathways,
    write_csv,
    write_iv_csv,
    write_edge_currents_csv,
)
from nanonet.analysis.spectral import effective_resistance, spectral_metrics
from nanonet.sweeps.parameter_sweep import (
    SweepConfig,
    run_sweep_vary_std,
    run_sweep_vary_mean,
    run_sweep_vary_N_mean,
    run_sweep_vary_N,
    run_sweep_vary_voids,
    run_all_cases,
)
from nanonet.visualization.plots import (
    plot_iv_curve,
    plot_iv_overlay,
    plot_evolution,
    plot_snapshots,
    plot_spectral_evolution,
)

__version__ = "0.2.0"

__all__ = [
    # Core
    "NanoparticleNetwork",
    # Analysis
    "sweep",
    "count_edge_disjoint_pathways",
    "write_csv",
    "write_iv_csv",
    "write_edge_currents_csv",
    "effective_resistance",
    "spectral_metrics",
    # Sweeps
    "SweepConfig",
    "run_sweep_vary_std",
    "run_sweep_vary_mean",
    "run_sweep_vary_N_mean",
    "run_sweep_vary_N",
    "run_sweep_vary_voids",
    "run_all_cases",
    # Visualization
    "plot_iv_curve",
    "plot_iv_overlay",
    "plot_evolution",
    "plot_snapshots",
    "plot_spectral_evolution",
]
