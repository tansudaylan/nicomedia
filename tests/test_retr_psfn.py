import numpy as np
import pytest

from nicomedia.main import retr_psfn


@pytest.mark.parametrize(
    ("model", "parameters"),
    (
        ("singgaus", None),
        ("singking", [0.3, 2.5]),
        ("doubking", [0.3, 2.5, 0.8, 3.0, 0.7]),
    ),
)
def test_psf_model_branches_accept_explicit_parameters_and_normalize(model, parameters):
    angles = np.linspace(0.0, 1.0, 501)
    psfp = None if parameters is None else np.asarray(parameters, dtype=float)[:, None, None]

    profile = retr_psfn(
        {"sigc": 0.3},
        np.array([0]),
        angles,
        model,
        typenormangl="ferm",
        psfp=psfp,
        indxpsfpinit=0 if psfp is not None else None,
        fermscalfact=np.ones((1, 1)),
    )

    assert profile.shape == (1, angles.size, 1)
    assert np.isfinite(profile).all()
    integral = 2.0 * np.pi * np.trapezoid(profile[0, :, 0] * np.sin(angles), angles)
    assert integral == pytest.approx(1.0)


def test_king_profiles_require_parameter_array_and_start_index():
    angles = np.linspace(0.0, 1.0, 51)

    with pytest.raises(ValueError, match="psfp and indxpsfpinit"):
        retr_psfn({}, np.array([0]), angles, "singking")