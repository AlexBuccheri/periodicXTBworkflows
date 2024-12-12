"""Compute properties for a given unit cell
"""
from __future__ import annotations

import enum
from typing import List, Tuple

import numpy as np
from enum import Enum

from pymatgen.core.structure import Structure

from materials_project.electronegativities import pauling


class Pairs(Enum):
    ALL = enum.auto
    NN = enum.auto


class BondType(Enum):
    COVALENT = enum.auto
    POLAR_COVALENT = enum.auto
    IONIC = enum.auto


def get_nearest_neighbours(structure: Structure) -> np.ndarray:
    """ Get the nearest neighbours of each atom in a periodic cell.

    Other functions of interest:
        get_bond_length()
        get_neighbor_list() Annoyingly needs a radial cut-off, whereas I'd
        like to ask for 1, 2, .. N neighbours.
        Could get a list of N shortest distances from the distance matrix and use
        that. i.e set(distances) . For N species, take Nth-neighbour * N(N-1)/2
    See https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure.Structure

    For single atom cells, this will return any empty array

    :param structure: Pymatgen structure object.
    :return nn_indices: Nearest neighbour indices, where nn_indices[i]
    gives the index of the NN of atom i.
    """
    # For periodic structures, this should return the nearest image distance.
    distance_m = structure.distance_matrix
    # Pad diagonals such that self-interactions are skipped
    np.fill_diagonal(distance_m, np.inf)
    # Indices of nearest neighbours in the cell
    nn_indices = np.argmin(distance_m, axis=1)
    return nn_indices


def get_all_pair_interactions(structure: Structure) -> list:
    """ Get all bonded and non-bonded, unique atomic pairs in a unit cell.

    Get all unique combinatorial pairs of atomic indices, which corresponds
    to stirling(N, 2) = N(N-1)/2, which is the upper triangle of matrix(N, N),
    where N is the number of atoms.

    For single-atom cells, this will return any empty list
    """
    # Can probably replace this line
    n_atoms = len(structure.cart_coords)
    pair_indices = []
    for i in range(0, n_atoms):
        for j in range(i + 1, n_atoms):
            pair_indices.append((i, j))
    return pair_indices


def get_pair_interactions(structure: Structure, pairs: Pairs):
    """ Wrapper for obtaining pair interactions
    """
    if pairs == Pairs.ALL:
        return get_all_pair_interactions(structure)
    elif pairs == Pairs.NN:
        nn_indices = get_nearest_neighbours(structure)
        # Repackage to be consistent
        pair_indices = [(i, nn_indices[i]) for i in range(len(nn_indices))]
        return pair_indices
    else:
        raise ValueError('Enum not valid')


def diff_electronegativities(structure: Structure, pair_indices: List[tuple]) -> Tuple[float | np.ndarray | None, list]:
    """ Calculate the average electronegativity difference
    for N pair interactions.

    <delta_EN> = 1/N sum_(ij)^N |E_j - E_i|
    where N = number of pair interactions.

    In the instance of a single-atom cell, return 0.0
    """
    n_atoms = len(structure.cart_coords)

    # Single-atom cells
    if not pair_indices:
        assert n_atoms == 1
        return 0.0, []

    delta_E = np.empty(shape=(len(pair_indices)))
    species = [x.symbol for x in structure.species]

    missed_interactions = []
    cnt = 0
    for i, j in pair_indices:
        try:
            delta_E[cnt] = np.abs(pauling[species[i]] - pauling[species[j]])
            cnt += 1
        except TypeError:
            missed_interactions.append((i, j))

    # Exceptions - not all elements have tabulated Pauling electronegativities
    if n_atoms == 2 and cnt == 0:
        # Same species, Pauling EN not defined (i.e cnt ==0)
        if species[0] == species[1]:
            return 0.0, []
        # Different species, one or both do not have Pauling EN defined
        else:
            return None, [(0, 1)]

    average_en = np.mean(delta_E[0:cnt])

    return average_en, missed_interactions


def label_interaction(electronegativity: float, as_string: bool) -> BondType:
    """
    """
    convert = {True: lambda x: str(x), False: lambda x: x}

    assert electronegativity >= 0, 'electronegativity should be positive'
    if electronegativity < 0.4:
        return convert[as_string](BondType.COVALENT)
    elif (electronegativity >= 0.4) and (electronegativity <= 1.7):
        return convert[as_string](BondType.POLAR_COVALENT)
    else:
        return convert[as_string](BondType.IONIC)
