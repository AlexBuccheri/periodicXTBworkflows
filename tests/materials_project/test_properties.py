"""
Running from project root gets the paths correct. Make more robust later

 pytest -s tests/materials_project/test_properties.py
"""
from typing import Dict
import pickle

import numpy as np
import pytest

from pymatgen.core.structure import Structure

from materials_project.properties import get_nearest_neighbours, get_pair_interactions, \
    Pairs, diff_electronegativities


@pytest.fixture(scope="module")
def setup_structures() -> Dict[str, Structure]:
    with open("materials_project/structures/structures.pkl", "rb") as file:
        structures: Dict[str, Structure] = pickle.load(file)

    assert len(structures.keys()), "one system"
    assert "mp-149" in structures.keys()

    # Check structure of "mp-149"
    positions = structures["mp-149"].cart_coords
    # species = structures["mp-149"].

    assert np.allclose(positions, [[3.8891685, 2.7500585, 6.7362365], [0.5555955, 0.3928655, 0.9623195]])
    # assert species == ['Element Si', 'Element Si']

    return structures


def test_unique_bonds(setup_structures: Dict[str, Structure]):

    structure = setup_structures["mp-149"]
    nn_indices = get_nearest_neighbours(structure)
    assert np.array_equal(nn_indices, [1, 0])

    #TODO(Alex) Test on a less-trivial system


def test_diff_electronegativities(setup_structures):
    structure = setup_structures["mp-149"]
    species = [x.symbol for x in structure.species]

    assert species == ['Si', 'Si'], 'Primitive silicon cell'

    pair_indices = get_pair_interactions(structure, Pairs.ALL)
    average_en, missed_interactions = diff_electronegativities(structure, pair_indices)

    assert np.isclose(average_en, 0.0), "Cell with one species should have net EN of zero"
    assert missed_interactions == [], "Silicon has tabulated electronegativities"


    #TODO(Alex) Test diff_electronegativities on a less-trivial system

