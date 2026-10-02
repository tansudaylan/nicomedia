from types import SimpleNamespace

import numpy as np
import pytest

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