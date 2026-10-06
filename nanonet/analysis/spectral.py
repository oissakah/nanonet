"""Spectral diagnostics for the canonical nanonet circuit."""

from __future__ import annotations

import numpy as np

EIG_REL_TOL = 1e-9


def build_full_system(net, activated_nodes):
    """Return the exact active circuit system used by the Kirchhoff solver."""
    return net.build_active_laplacian(activated_nodes)


_build_full_system = build_full_system


def effective_resistance(net, activated_nodes: set, V_test: float = 1.0) -> float:
    """Source-to-drain effective resistance, computed as V/I."""
    solution = net.solve_active_network(activated_nodes, float(V_test))
    current = solution["total_current_A"]
    return float(V_test) / current if current > 0 else np.nan


def spectral_metrics(net, activated_nodes: set) -> tuple[float, float, int]:
    """Return raw lambda_2 [S], lambda_2/lambda_max, and zero-mode count."""
    system = build_full_system(net, activated_nodes)
    if system is None:
        return np.nan, np.nan, 0

    matrix = system["laplacian"].toarray()
    eigenvalues = np.sort(np.linalg.eigvalsh(0.5 * (matrix + matrix.T)))
    maximum = float(eigenvalues[-1]) if len(eigenvalues) else 0.0
    if maximum <= 0:
        return np.nan, np.nan, 0

    tolerance = EIG_REL_TOL * maximum
    n_zero = int(np.sum(eigenvalues < tolerance))
    nonzero = eigenvalues[eigenvalues >= tolerance]
    lambda_2 = float(nonzero[0]) if len(nonzero) else np.nan
    ratio = float(lambda_2 / maximum) if np.isfinite(lambda_2) else np.nan
    return lambda_2, ratio, n_zero
