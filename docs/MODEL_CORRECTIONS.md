# Model consistency corrections

This package now follows the corrected V14 nanonecklace transport conventions.

1. **One transport engine.** Voltage gating is followed by Kirchhoff nodal analysis. Device current and every edge current come from the same potential solution.
2. **Symmetric junction resistance.** Half of each endpoint junction resistance is assigned to an active connection. This replaces the orientation-dependent split-node construction with a symmetric passive circuit Laplacian.
3. **Activation/resistance decoupling.** `Va_i` controls activation only. Junction resistance is fixed independently at `node_resistance_ohm`.
4. **No parallel transport law.** No separate tunneling/path-current approximation is used by the active framework.
5. **Bounded activation sampling.** Bounded normal activation distributions use rejection sampling rather than clipping, avoiding artificial probability mass at the bounds.
6. **Canonical current reconstruction.** Source current, drain current, device current, and edge currents are calculated from the same Kirchhoff solution. Current imbalance is retained as a numerical diagnostic.
7. **Fixed threshold convention.** `V_T` is the first sampled source-drain percolation voltage, so `V_T = V_perc`. It is held fixed when fitting `A` and `zeta`.
8. **Nonlinear fit window.** The power-law fit uses positive-current points strictly above `V_T` and ends at 90% node activation, or at the available fit-window limit if saturation is not reached.
9. **N-density definition.** Density studies keep the domain and connection radius fixed (`r_c = 0.15`) as `N` varies. The former `N^(-1/2)` radius scaling is not used.
10. **Void fraction.** Requested and achieved void fractions are stored. Achieved void area is computed from the geometric union, so overlaps count once.
11. **Spectral consistency.** Spectral diagnostics use the same conductance-weighted active-circuit Laplacian as the electrical solver. Raw `lambda_2` is reported in siemens [S].
12. **Independent transport channels.** The pathway diagnostic is the edge-disjoint source-drain path count at `V_perc`, obtained by unit-capacity max flow rather than simple-path enumeration.
13. **Current participation.** `N_eff = (sum |I_e|)^2 / sum I_e^2` measures how broadly current is distributed across conducting edges.
