""" Band processing

Wrote our own routines to process band gap data, as
from ase.dft.bandgap import bandgap
didn't work with the standard calculator.
"""
import numpy as np
import re
from typing import Optional


class BandGap:
    def __init__(self, energy_vbt: float, energy_cbb: float, kv: int, kc: int):
        """
        :param energy_vbt: Energy of valence band top
        :param energy_cbb: Energy of conduction band bottom
        :param kv: K-point associated with the valence band top
        :param kc: K-point associated with the conduction band bottom
        """
        self.e_vbt = energy_vbt
        self.e_cbb = energy_cbb
        self.kv = kv
        self.kc = kc

    def gap(self):
        return self.e_cbb - self.e_vbt

    def print(self, string: Optional[str] = 'Bandgap', conversion=1.):
        print(string + f" {self.gap() * conversion} from k = {self.kv} to k = {self.kc}")


def direct_bandgap(eigenvalues: np.ndarray, k_points: np.ndarray, vbt: float) -> BandGap:
    """ Get the direct band gap of a spin-unpolarised material.

    Testing shows bs.reference (argument passed to vbt) may have a floating point
    disagreement with the actual VBT. As such, loosen the search criterion.

    One could alternatively grep for
    "number of electrons       =         8.00" in the QE output and
    divide by 2 to get the number of occupied bands.

    :param eigenvalues: Band energies, with shape (nkpoints, nbands)
    :param k_points: k-points of the band path, with shape (nkpoints, 3)
    :param vbt: Energy of the valence band top
    :return Instance of BandGap
    """
    if eigenvalues.ndim != 2:
        raise ValueError('Expect eigenvalues to have 2 dimensions. '
                         'Spin-polarised case is not handled')

    assert eigenvalues.shape[0] == k_points.shape[0], \
        'Number of k-points in `eigenvalues` inconsistent with `k_points`'

    # Find Gamma index (k-index)
    gamma = np.array([0., 0., 0.])
    bool_mask = (k_points == gamma).all(-1)

    if bool_mask.size == 0:
        raise ValueError(f'Gamma point not present in k_points path: {k_points}')

    indices = np.arange(0, k_points.shape[0])[bool_mask]
    # Take first instance of Gamma
    ik = indices[0]

    # Find band edges
    i_vbt = np.argmax(np.where(eigenvalues[ik, :] <= 1.1 * vbt)[0])
    i_cbb = i_vbt + 1

    return BandGap(eigenvalues[ik, i_vbt], eigenvalues[ik, i_cbb], k_points[ik, :], k_points[ik, :])


def direct_bandgap_alt(eigenvalues: np.ndarray, k_points: np.ndarray, n_occupied: int) -> BandGap:
    """ Get the direct band gap of a spin-unpolarised material.
    """
    if eigenvalues.ndim != 2:
        raise ValueError('Expect eigenvalues to have 2 dimensions. '
                         'Spin-polarised case is not handled')

    assert eigenvalues.shape[0] == k_points.shape[0], \
        'Number of k-points in `eigenvalues` inconsistent with `k_points`'

    # Find Gamma index (k-index)
    gamma = np.array([0., 0., 0.])
    bool_mask = (k_points == gamma).all(-1)

    if bool_mask.size == 0:
        raise ValueError(f'Gamma point not present in k_points path: {k_points}')

    indices = np.arange(0, k_points.shape[0])[bool_mask]
    # Take first instance of Gamma
    ik = indices[0]

    # Find band edges
    i_vbt = n_occupied - 1
    i_cbb = i_vbt + 1

    return BandGap(eigenvalues[ik, i_vbt], eigenvalues[ik, i_cbb], k_points[ik, :], k_points[ik, :])


def get_spin_unpolarised_occupation(input_str: str) -> int:
    """ Number of occupied bands.

    regex this line:
    number of electrons       =         <FLOAT>

    :param input_str: String contents of espresso.pwo
    :return n_occupied: Integer number of occupied bands
    """
    match = re.findall(r'number of electrons\s*=.*$', input_str, flags=re.MULTILINE)[0]
    if not match:
        raise ValueError('"number of electrons" not found in `input_str`')
    n_electrons = match.split('=')[-1].strip()
    n_occupied = 0.5 * float(n_electrons)
    assert int(n_occupied) == n_occupied, "Expect integer number of electrons, hence occupations"
    return int(n_occupied)


def fundamental_bandgap_alt(eigenvalues: np.ndarray, k_points: np.ndarray, n_occupied: int) -> BandGap:
    """ Get the fundamental band gap of a spin-unpolarised material.

    Rather than find the VBT using a loose tolerance and risk getting the wrong point,
    regex the number of electrons from espresso output, infer the occupation and get the
    VB maximum index from that.

    :param eigenvalues: Band energies, with shape (nkpoints, nbands)
    :param k_points: k-points of the band path, with shape (nkpoints, 3)
    :param n_occupied: Number of occupied states
    :return Instance of BandGap
    """
    if eigenvalues.ndim != 2:
        raise ValueError('Expect eigenvalues to have 2 dimensions. '
                         'Spin-polarised case is not handled')

    assert eigenvalues.shape[0] == k_points.shape[0], \
        'Number of k-points in `eigenvalues` inconsistent with `k_points`'

    # Python indexing
    ie_vbm = n_occupied - 1
    ik_vbm = np.argmax(eigenvalues[:, ie_vbm])
    ie_cbm = ie_vbm + 1
    ik_cbm = np.argmin(eigenvalues[:, ie_cbm])

    return BandGap(eigenvalues[ik_vbm, ie_vbm], eigenvalues[ik_cbm, ie_cbm], k_points[ik_vbm, :], k_points[ik_cbm, :])


def fundamental_bandgap(eigenvalues: np.ndarray, k_points: np.ndarray, vbt: float) -> BandGap:
    """ Get the fundamental band gap of a spin-unpolarised material.

    :param eigenvalues: Band energies, with shape (nkpoints, nbands)
    :param k_points: k-points of the band path, with shape (nkpoints, 3)
    :param vbt: Energy of the valence band top
    :return Instance of BandGap
    """
    if eigenvalues.ndim != 2:
        raise ValueError('Expect eigenvalues to have 2 dimensions. '
                         'Spin-polarised case is not handled')

    assert eigenvalues.shape[0] == k_points.shape[0], \
        'Number of k-points in `eigenvalues` inconsistent with `k_points`'

    # Large tol because vbt reference from ASE has some error w.r.t. actual VBT
    tol = 1.e-2
    indices = np.where(np.abs(eigenvalues - vbt) <= tol)
    # To select eigenvalues using indices: eigenvalues[indices])
    if indices[0].size == 0:
        raise ValueError(f'No eigenvalues/bands found <= the reference {vbt}')

    # Take the largest band index,
    ik_vbm, ie_vbm = indices[0][-1], indices[1][-1]

    # Conduction band minimum
    ie_cbm = ie_vbm + 1
    ik_cbm = np.argmin(eigenvalues[:, ie_cbm])

    return BandGap(eigenvalues[ik_vbm, ie_vbm], eigenvalues[ik_cbm, ie_cbm], k_points[ik_vbm, :], k_points[ik_cbm, :])
