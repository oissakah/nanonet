import numpy as np

from nanonet.sweeps.parameter_sweep import _fit_power_law


def test_fit_holds_vt_fixed_at_percolation():
    voltage = np.arange(0.0, 7.0, 1.0)
    V_T = 2.0
    zeta_true = 2.5
    A_true = 3.0e-9
    current = np.zeros_like(voltage)
    mask = voltage > V_T
    current[mask] = A_true * (voltage[mask] - V_T) ** zeta_true

    fit = _fit_power_law(
        voltage,
        current,
        V_T=V_T,
        v_transition=6.0,
        fit_window=10.0,
    )
    assert fit["success"]
    assert fit["V_T"] == V_T
    assert np.isclose(fit["zeta"], zeta_true, rtol=1e-10)
    assert np.isclose(fit["A"], A_true, rtol=1e-10)


def test_failed_fit_still_retains_threshold():
    voltage = np.array([0.0, 1.0, 2.0, 3.0])
    current = np.array([0.0, 0.0, 1e-9, 2e-9])
    fit = _fit_power_law(voltage, current, V_T=1.5, v_transition=3.0)
    assert not fit["success"]
    assert fit["V_T"] == 1.5
