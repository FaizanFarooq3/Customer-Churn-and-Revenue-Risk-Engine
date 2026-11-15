import numpy as np
import pandas as pd
from src.monitoring.drift_report import calculate_psi

def test_calculate_psi_identical_distributions_is_near_zero():
    """Identical reference and current distributions should have ~0.0 PSI"""
    rng = np.random.default_rng(42)
    ref = pd.Series(rng.normal(loc=100.0, scale=15.0, size=1000))
    cur = ref.copy()

    psi = calculate_psi(ref, cur)
    assert psi == 0.0

def test_calculate_psi_detects_shifted_distribution():
    """Heavily shifted distribution should produce PSI > 0.25 (significant drift)"""
    rng = np.random.default_rng(42)
    ref = pd.Series(rng.normal(loc=100.0, scale=15.0, size=1000))
    cur = pd.Series(rng.normal(loc=140.0, scale=15.0, size=1000))

    psi = calculate_psi(ref, cur)
    assert psi > 0.25
    