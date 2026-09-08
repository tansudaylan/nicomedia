import numpy as np

import nicomedia
import tdpy


def test_retr_doubking_scalar_matches_tdpy():
    scaldevi = np.array([0.1, 0.5, 1.0])
    frac = 0.35
    sigc = 0.8
    gamc = 2.2
    sigt = 1.7
    gamt = 3.4

    psfn_nico = nicomedia.retr_doubking(scaldevi, frac, sigc, gamc, sigt, gamt)
    psfn_tdpy = tdpy.retr_doubking(scaldevi, frac, sigc, gamc, sigt, gamt)

    assert np.allclose(psfn_nico, psfn_tdpy)


def test_retr_doubking_broadcast_matches_tdpy():
    scaldevi = np.linspace(0.0, 2.0, 5)[None, :, None]
    frac = np.array([0.25, 0.75])[:, None, None]
    sigc = np.array([0.7, 1.0])[:, None, None]
    gamc = np.array([2.0, 2.5])[:, None, None]
    sigt = np.array([1.2, 1.8])[:, None, None]
    gamt = np.array([3.0, 4.0])[:, None, None]

    psfn_nico = nicomedia.retr_doubking(scaldevi, frac, sigc, gamc, sigt, gamt)
    psfn_tdpy = tdpy.retr_doubking(scaldevi, frac, sigc, gamc, sigt, gamt)

    assert psfn_nico.shape == (2, 5, 1)
    assert np.allclose(psfn_nico, psfn_tdpy)