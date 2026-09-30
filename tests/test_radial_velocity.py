import numpy as np
import pytest

import nicomedia


def test_retr_rvel_crosses_zero_at_mid_transit_with_the_physical_semi_amplitude():
    time = np.linspace(100.0, 110.0, 4001)  # [day]
    semi_amplitude = nicomedia.retr_rvelsema(5.0, 1.0, 1e-3, 90.0, 0.0)  # [m/s]
    velocity = nicomedia.retr_rvel(time, 102.5, 5.0, 1e-3, 1.0, 90.0, 0.0, 90.0)
    assert nicomedia.retr_rvel(np.array([102.5]), 102.5, 5.0, 1e-3, 1.0, 90.0, 0.0, 90.0) == pytest.approx(0.0, abs=1e-9)
    assert np.amax(velocity) == pytest.approx(semi_amplitude, rel=1e-5)
