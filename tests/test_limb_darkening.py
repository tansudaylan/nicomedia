import numpy as np
import pytest

import nicomedia
from tdpy.exoplanet import quadratic_limb_darkening


def test_legacy_limb_darkening_helpers_delegate_without_behavior_changes():
    coefficients = np.array((0.4, 0.25))
    cosine_emission_angle = np.linspace(0.0, 1.0, 5)

    kipping_parameters = nicomedia.retr_coeflmdkkipp(*coefficients)
    assert nicomedia.retr_coeflmdkfromkipp(*kipping_parameters) == pytest.approx(coefficients)
    assert nicomedia.retr_brgtlmdk(cosine_emission_angle, coefficients) == pytest.approx(
        quadratic_limb_darkening(cosine_emission_angle, coefficients)
    )
