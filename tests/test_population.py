import numpy as np
import pytest

import nicomedia


def test_missing_planet_metrics_remain_nan_without_runtime_warning():
    values = {
        'masscomp': [1.0, 1.0],
        'radicomp': [2.0, 2.0],
        'tmptcomp': [800.0, np.nan],
        'radistar': [1.0, 1.0],
        'magtsystJbnd': [10.0, 10.0],
        'magtsystKbnd': [10.0, 10.0],
        'tmptstar': [5700.0, 5700.0],
    }
    population = {name: [np.asarray(data)] for name, data in values.items()}

    with np.errstate(all='raise'):
        nicomedia.calc_tsmmesmm(population)

    for name in ('tsmm', 'stdvtsmm', 'esmm', 'stdvesmm'):
        assert np.isfinite(population[name][0][0])
        assert np.isnan(population[name][0][1])


def test_retr_subp_preserves_population_metadata():
    populations = {"all": {"mass": [np.array([1.0, 2.0, 3.0]), "kg"]}}
    sample_counts = {}
    sample_indices = {"all": {}}

    nicomedia.retr_subp(
        populations,
        sample_counts,
        sample_indices,
        "all",
        "chosen",
        np.array([0, 2]),
    )

    np.testing.assert_array_equal(populations["chosen"]["mass"][0], np.array([1.0, 3.0]))
    assert populations["chosen"]["mass"][1] == "kg"
    assert sample_counts["chosen"] == 2
    np.testing.assert_array_equal(sample_indices["all"]["chosen"], np.array([0, 2]))
    assert sample_indices["chosen"] == {}


def test_retr_subp_preserves_list_rejection():
    with pytest.raises(Exception):
        nicomedia.retr_subp({"all": {}}, {}, {"all": {}}, "all", "chosen", [0])