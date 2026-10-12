from types import SimpleNamespace

import numpy as np
import pytest
from tdpy.spectral import evaluate_continuum_spectrum, evaluate_line_profile

from nicomedia.main import retr_spec


@pytest.fixture
def spectral_state():
    energies = np.linspace(0.1, 1.0, 101)
    return SimpleNamespace(
        numbener=energies.size,
        numbenerplot=energies.size,
        bctrpara=SimpleNamespace(ener=energies, enerplot=energies),
        enerpivt=0.5,
        indxener=np.arange(energies.size),
        indxenerpivt=50,
        factlogtenerpivt=np.log(energies / 0.5),
    )


@pytest.mark.parametrize(
    ("spectype", "parameters"),
    (
        ("gaus", {"sigm": [0.08, 0.1]}),
        ("voig", {"sigm": [0.08, 0.1], "gamm": [0.03, 0.04]}),
        ("edis", {"edisintp": lambda centers: np.full_like(centers, 0.08)}),
        ("pvoi", {"sigm": [0.08, 0.1], "gamm": [0.03, 0.04], "frac": [0.3, 0.6]}),
        ("lore", {"gamm": [0.03, 0.04]}),
    ),
)
def test_line_spectra_use_explicit_spectype_and_parameters(spectral_state, spectype, parameters):
    spectrum = retr_spec(
        spectral_state,
        [2.0, 3.0],
        elin=[0.4, 0.7],
        spectype=spectype,
        **parameters,
    )

    assert spectrum.shape == (101, 2)
    assert np.isfinite(spectrum).all()
    assert np.any(spectrum > 0.0)


@pytest.mark.parametrize(
    ("spectype", "parameters"),
    (
        ("powr", {"sind": [1.0, 2.0]}),
        ("colr", {"sindcolr": np.full(100, 1.0)}),
        ("curv", {"sind": [1.0, 2.0], "curv": [0.1, 0.2]}),
        ("expc", {"sind": [1.0, 2.0], "expc": [0.5, 0.8]}),
    ),
)
def test_continuum_spectra_use_explicit_spectype(spectral_state, spectype, parameters):
    spectrum = retr_spec(spectral_state, [2.0, 3.0], spectype=spectype, **parameters)

    assert spectrum.shape == (101, 2)
    assert np.isfinite(spectrum).all()


def test_unknown_spectral_type_is_rejected(spectral_state):
    with pytest.raises(ValueError, match="Unsupported spectype"):
        retr_spec(spectral_state, [1.0], spectype="missing")


def test_line_profiles_match_tdpy_and_keep_scalar_width_broadcasting(spectral_state):
    axis = spectral_state.bctrpara.ener
    flux = [2.0, 3.0]
    centers = [0.4, 0.7]
    profile_inputs = (
        ("gaus", {"sigm": 0.08}, {"gaussian_width": [0.08, 0.08]}),
        ("lore", {"gamm": 0.03}, {"lorentz_width": [0.03, 0.03]}),
        (
            "voig",
            {"sigm": 0.08, "gamm": 0.03},
            {"gaussian_width": [0.08, 0.08], "lorentz_width": [0.03, 0.03]},
        ),
        (
            "pvoi",
            {"sigm": 0.08, "gamm": 0.03, "frac": 0.4},
            {
                "gaussian_width": [0.08, 0.08],
                "lorentz_width": [0.03, 0.03],
                "mixing_fraction": [0.4, 0.4],
            },
        ),
        (
            "edis",
            {"edisintp": lambda line_centers: np.full_like(line_centers, 0.08)},
            {"gaussian_width": [0.08, 0.08]},
        ),
    )
    for spectype, nicomedia_arguments, tdpy_arguments in profile_inputs:
        actual = retr_spec(
            spectral_state,
            flux,
            elin=centers,
            spectype=spectype,
            **nicomedia_arguments,
        )
        expected = evaluate_line_profile(
            axis,
            spectype,
            flux,
            centers,
            **tdpy_arguments,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0.0)


def test_continuum_models_match_tdpy(spectral_state):
    flux = np.array([2.0, 3.0])
    spectral_index = np.array([1.0, 2.0])
    models = (
        ("powr", {}),
        ("curv", {"curv": np.array([0.1, 0.2])}),
        ("expc", {"expc": np.array([0.8, 1.2])}),
    )
    for spectype, parameters in models:
        actual = retr_spec(
            spectral_state,
            flux,
            spectype=spectype,
            sind=spectral_index,
            **parameters,
        )
        expected = evaluate_continuum_spectrum(
            spectral_state.bctrpara.ener,
            flux,
            spectype,
            spectral_state.enerpivt,
            spectral_index,
            curvature=parameters.get("curv"),
            cutoff_energy=parameters.get("expc"),
            log_energy_ratio=spectral_state.factlogtenerpivt,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0.0)