import numpy as np
import pytest

import nicomedia


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