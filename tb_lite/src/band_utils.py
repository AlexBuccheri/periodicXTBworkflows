"""Generate consistent band paths and k-grids.
"""
import numpy as np
from typing import Union, List

from ase.atoms import Atoms
from ase.dft.kpoints import BandPath

from tb_lite.src.utils import reciprocal_lattice_vectors


def preprocess_ase_path(path: str) -> list:
    """Preprocess for symbols that contain numbers.

    For example, ensure we get 'G1', not 'G' '1'
    from a path like 'G1XWKGLUWLK,UX'.

    :param path: ASE bandpath.path.
    :return: new_path. ['G1', 'X', 'W', 'K', 'G', 'L', 'U', 'W', 'L', 'K', ',', 'U', 'X']
    """
    new_path = []
    for symbol in path:
        if symbol.isnumeric():
            last_element = new_path.pop()
            new_path.append(last_element + symbol)
            continue
        new_path.append(symbol)
    return new_path


def labels_from_ase_bandpath(path) -> list:
    """ Convert ASE path into labels.

    bandpath.path = 'G1XWKGLUWLK,UX'
    returned as ['G1', 'X', 'W', 'K', 'G', 'L', 'U', 'W', 'L', 'K,U', 'X']

    :param bandpath:
    :return:
    """
    new_path = preprocess_ase_path(path)

    labels = [new_path[0]]
    for symbol in new_path[1:]:
        # Concatenate ['K', ',', 'U'] -> ['K,U']
        if labels[-1] == ',':
            labels.pop()
            labels[-1] += ',' + symbol
            continue
        labels.append(symbol)

    return labels


def high_symmetry_point_indices(k_points, special_points: dict) -> np.ndarray:
    """ Given an array of k-points, return the indices of k-points
    corresponding to high-symmetry points.

    :param k_points: Array of k-points, with shape(?, ?)
    :param special_points: Dict of high symmetry {labels: points} for example
    {'Gamma': [0., 0., 0.]}.
    :return: index array of high-symmetry points, with ordering consistent with special_points
    """
    indices = []
    for point in special_points.values():
        found_indices = np.where((k_points == point).all(axis=1))[0]
        indices.extend(found_indices)

    indices = np.array(indices)
    indices.sort()

    return indices


def flatten_whole_kpath(k_points: np.ndarray) -> np.ndarray:
    """ Given a set of k-points, return a vector of accumulated
    displacement norms.

    Much simpler than `flattened_kpath` because this does not
    distinguish each separate k-path between high symmetry points
    like:  |G ...| X ... |K ... W|

    It does |... ...  ... | where high-symmetry points are not relevant.

    For use with band structure plotting.

    :param k_points: An array of k-points.
    :return: Flat path.
    """
    assert k_points.shape[1] == 3, "Expect k_points.shape = (n_k_points, 3)"

    n_k_points = k_points.shape[0]
    flat_path = np.empty(shape=n_k_points)

    flat_path[0] = 0.
    for ik in range(1, n_k_points):
        dk = k_points[ik, :] - k_points[ik - 1, :]
        flat_path[ik] = flat_path[ik - 1] + np.linalg.norm(dk)

    return flat_path


def flattened_kpath(k_points: np.ndarray, special_points: dict) -> tuple:
    """ Flatten a k-path for use in Band Structure Plotting.
    Also return high-symmetry points in this flattened vector, as xticks.

    Algorithm Behaviour
    -------------------------
    Each k-path will stop one short of the end point. For example

    k-path 0: G-X [k_Gamma, k1, k2, ..., k_N-1]
    k-path 1: X-W [k_X, k1, k2, ..., k_N-1]
    ...
    k-path M: U-G [k_U, k1, k2, ..., k_N-1, k_Gamma]

    where M = number of band paths, equal to the number high-symmetry points - 1, and the final
    path includes the end point.

    For discontinuous points, the end point from the prior path is
    included. For example at K,U:

    W-K [k_W, k1, k2, ..., k_N-1]
    K,U [k_K]
    U-X [k_U, k1, k2, ..., k_N-1]

    This results in two adjacent points in the band structure with the same energies,
    but it should not be a visible problem with plotting when using a dense k-grid,
    and it is consistent with the k-grid from ASE.

    Note on Implementation
    -------------------------
     Discontiguous band paths show up as two high symmetry points next to each
     other in k_points array. When translated into k indices this will look like
     [..., 33, 34, ...], so (indices[i] - indices[i - 1]) == 1 indicates a
     discontinuity between two k-points.
     NOTE, this will fail if the band path sampling is not sufficiently dense.


    :param k_points: Array of k-points =[n_k x 3]
    :param special_points: For example special_points[G] = array[0., 0., 0.]
    """
    indices = high_symmetry_point_indices(k_points, special_points)
    n_high_sym_points = len(indices)

    k_vectors = []
    xticks = []
    k_start = 0.

    # Iterate over each k-path 'G-X', 'X-W', etc
    for i in range(1, n_high_sym_points):

        ik_current, ik_prior = indices[i], indices[i - 1]

        # Skip vectors between high-symmetry points like K,U
        # which should have no separation on a band structure plot.
        if (ik_current - ik_prior) == 1:
            # Conserve number of k sampling points
            k_vectors.append(k_start)
            # Don't add to xticks - only need to label the point once
            k_start = k_vectors[-1]
            continue

        dk = np.linalg.norm(k_points[ik_current, :] - k_points[ik_prior, :])
        n_k = ik_current - ik_prior + 1

        k_vector = np.linspace(k_start, k_start + dk, num=n_k, endpoint=True)
        # First point in each path = high symmetry point
        xticks.append(k_vector[0])
        # Add all but the end point to the total path
        k_vectors.extend(k_vector[:-1])
        # Start of next path = end point of prior
        k_start = k_vector[-1]

    # Last path, include the end point
    xticks.append(k_start)
    k_vectors.append(k_start)

    assert len(k_vectors) == k_points.shape[0], "Number of grid points not conserved when flattening"

    # I appear to have written it such that K,U get the same flattened point
    k_vectors = np.array(k_vectors)
    unique_high_symmetry_points = np.unique(k_vectors[indices])
    if not np.allclose(unique_high_symmetry_points, xticks):
        print('unique_high_symmetry_points', unique_high_symmetry_points)
        print('xticks', xticks)
        raise ValueError("xticks is not consistent")

    return k_vectors, xticks


def monkhorst_pack_k_grid_sampling(n_grids: int, atoms_or_lattice: Union[Atoms, np.ndarray], even_grids=True) -> List[np.ndarray]:
    """ Generate k-sampling grid/s with the correct sampling per dimension,
    as required by the system's reciprocal unit cell vectors.

    :param n_grids: Number of k-grids to return
    :param atoms_or_lattice: Expect lattice row-wise:.
      a0 = lattice[0, :], a1 = lattice[1, :], a2 = lattice[2, :].
    :param even_grids: Only return even grids, if possible
    :return:
    """
    assert n_grids > 0, "Must request a finite number of k-grids"

    lattice = atoms_or_lattice
    if isinstance(atoms_or_lattice, Atoms):
        lattice = atoms_or_lattice.get_cell()

    recip_lattice = reciprocal_lattice_vectors(lattice)

    # v will iterate rows
    recip_magnitude = [np.linalg.norm(v) for v in recip_lattice]

    # Get a base grid that share some common factor, and the sampling
    # of the smallest reciprocal vector starts at 1.
    #
    # Note, the various rounding functions do not behave as
    # one might expect (for rint(2.5) -> 2.0)
    # Should not be a problem here, as expect rounding to be small
    # https://numpy.org/doc/stable/reference/generated/numpy.rint.html
    base_grid = np.array(np.rint(recip_magnitude / np.amin(recip_magnitude)), dtype=int)

    if even_grids:
        start = 2
        step = 2
    else:
        start = 1
        step = 2

    k_grids = [i * base_grid for i in range(start, n_grids + 1, step)]

    return k_grids


def bands_at_high_symmetry_points(bands: np.ndarray, band_path: BandPath):
    """ Given an  array of bands along a k-path, return
    an array of bands only at high symmetry points.

    :param bands: shape(n_k, n_bands)
    :param band_path: ASE band path
    :return bands_at_hs: Bands at high-symmetry k-points.
    """
    indices = high_symmetry_point_indices(band_path.kpts, band_path.special_points)
    bands_at_hs = bands[indices, :]
    return bands_at_hs
