# nanonet

[![CI](https://github.com/oissakah/nanonet/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/oissakah/nanonet/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`nanonet` is a Python package and Dash application for graph-based simulation of electron transport in self-assembled nanoparticle necklace networks.

This release synchronizes the reusable package with the corrected V14 model developed in [graph-based-nanoparticle-necklace-network](https://github.com/oissakah/graph-based-nanoparticle-necklace-network).

## Corrected transport model

The network is a spatial random graph. Junction (i) has a phenomenological activation voltage (V_{a,i}). It becomes electrically available when

[
V_{a,i} le V.
]

Activation and resistance are **independent**. A junction has a fixed resistance

[
R_{mathrm{node},i}=R_{mathrm{node}},
]

and an active connection (i-j) uses

[
R_{ij}=R_{mathrm{edge},ij}
+rac{1}{2}R_{mathrm{node},i}
+rac{1}{2}R_{mathrm{node},j},
]

with electrode-node junction resistance omitted. The resulting passive circuit is represented by one symmetric conductance-weighted Laplacian and solved using Kirchhoff nodal analysis.

The same solution is used for device current, source current, drain current, edge currents, effective resistance, and spectral diagnostics.

### Production defaults

| Parameter | Default |
|---|---:|
| Junction count (N) | 500 |
| Connection radius (r_c) | 0.15 |
| Domain | (1	imes1) |
| Edge resistance constant | (2.0	imes10^{10}) |
| Fixed junction resistance | (3.5	imes10^9 Omega) |
| Source / drain strips | 0.15 / 0.15 |
| Voltage sweep | 0–16 V in 0.5 V steps |
| Activation bounds | 0–20 V |
| Seeds | 41, 51, 61, 71, 81 |

See `optimized_config.yaml`.

## Threshold and nonlinear fit convention

The transport threshold is not a free fit parameter. It is defined as the first sampled source-drain percolation voltage:

[
V_T equiv V_{mathrm{perc}}.
]

The nonlinear region is fitted to

[
I=A(V-V_T)^zeta
]

using only positive-current points strictly above (V_T). (V_T) remains fixed while (A) and (zeta) are fitted. The upper fit boundary is the first voltage at which 90% of nodes are active; if that point is not reached, the configured fit-window limit is used.

## Current participation ratio

The package reports the current participation ratio

[
N_{mathrm{eff}}
=
rac{left(sum_e |I_e|ight)^2}
     {sum_e I_e^2}.
]

It estimates the effective number of conducting edges sharing the current. A small (N_{mathrm{eff}}) means strongly localized transport; a larger value means current is distributed across more of the conducting network.

The sweep table exposes the same quantity as `participation_ratio`, `current_participation_ratio`, and `N_eff` for compatibility.

## Independent pathways

At (V_{mathrm{perc}}), `nanonet` reports the maximum number of **edge-disjoint** source-to-drain transport channels. It is computed using a unit-capacity max-flow/min-cut calculation and is not a count of all simple paths.

## Spectral diagnostics

Spectral quantities come from the exact same active-circuit Laplacian used for transport. Raw (lambda_2) therefore has conductance units:

[
[lambda_2]=mathrm{S}.
]

The package also reports (lambda_2/lambda_{max}) as a dimensionless spectral-gap ratio.

## Density studies

When (N) is varied, the domain and connection radius remain fixed. The standard density studies use

[
r_c=0.15
]

for every (N). No (N^{-1/2}) radius rescaling is applied.

Two density runners are available:

- `run_sweep_vary_N_mean`: crossed (N	imeslangle V_aangle) study at fixed (sigma_a)
- `run_sweep_vary_N`: (N)-only study at fixed (langle V_aangle) and (sigma_a)

## Void fraction

For random-void simulations the package stores both:

- `void_fraction_requested`
- `void_fraction_achieved`

The achieved value is estimated from the union of the void areas, so overlaps are counted only once.

## Installation

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e ".[dev]"
pytest -q
```

Python 3.9 or newer is required.

## Quick start

```python
from nanonet import NanoparticleNetwork, sweep

net = NanoparticleNetwork(
    L=1.0,
    N=500,
    fv=0.0,
    mu_a=7.0,
    std_a=2.0,
    connection_radius=0.15,
    edge_k=2.0e10,
    node_resistance_ohm=3.5e9,
).build(seed=41)

iv = net.iv_curve(V_start=0.0, V_max=16.0, V_step=0.5)
print("V_perc =", iv["percolation_voltage"])

result = sweep(
    net,
    V_start=0.0,
    V_max=16.0,
    V_step=0.5,
)

for row in result["rows"]:
    if row["source_drain_connected"]:
        print("V =", row["V"])
        print("I =", row["total_current_A"])
        print("N_eff =", row["N_eff"])
        print("edge-disjoint pathways =", row["edge_disjoint_pathways_at_Vperc"])
        break
```

## Parameter sweeps

```python
from nanonet import (
    SweepConfig,
    run_sweep_vary_std,
    run_sweep_vary_mean,
    run_sweep_vary_N_mean,
    run_sweep_vary_N,
    run_sweep_vary_voids,
)

cfg = SweepConfig(
    node_resistance_ohm=3.5e9,
    connection_radius=0.15,
    seeds=[41, 51, 61, 71, 81],
)

density_map = run_sweep_vary_N_mean(cfg)
```

## Interactive web app

Run:

```bash
python app.py
```

and open `http://127.0.0.1:8050`.

The resistance control is **Junction resistance [Ω]** rather than an activation-dependent resistance scale. The web app uses the same fixed-(V_T) fit convention and the same conductance-weighted circuit Laplacian as the package solver.

## Package structure

```text
nanonet/
├── core/
│   └── network.py
├── analysis/
│   ├── sweep.py
│   └── spectral.py
├── sweeps/
│   └── parameter_sweep.py
├── visualization/
│   └── plots.py
└── io.py

docs/
└── MODEL_CORRECTIONS.md

tests/
├── test_model_consistency.py
├── test_threshold_convention.py
└── test_N_density_definition.py
```

## Model corrections

The complete consistency checklist is in [docs/MODEL_CORRECTIONS.md](docs/MODEL_CORRECTIONS.md).

## Research reference

The synchronized research implementation is associated with:

> Obed Issakah, Srivathsan Badrinarayanan, Ravi F. Saraf, and Janghoon Ock, “Graph-Based Kirchhoff Modeling of Non-Ohmic Electron Transport in Self-Assembled Nanonecklace Networks,” arXiv:2607.03698 (2026), DOI: 10.48550/arXiv.2607.03698.

```bibtex
@article{issakah2026nanonecklace,
  title={Graph-Based Kirchhoff Modeling of Non-Ohmic Electron Transport in Self-Assembled Nanonecklace Networks},
  author={Issakah, Obed and Badrinarayanan, Srivathsan and Saraf, Ravi F. and Ock, Janghoon},
  journal={arXiv preprint arXiv:2607.03698},
  year={2026},
  doi={10.48550/arXiv.2607.03698}
}
```

## License

MIT. See `LICENSE`.
