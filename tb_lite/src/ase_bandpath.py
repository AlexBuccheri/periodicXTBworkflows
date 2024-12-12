""" Wrappers for ASE band path.
"""
import ase
from ase.dft.kpoints import BandPath

from tb_lite.src.band_utils import labels_from_ase_bandpath, flattened_kpath
from tb_lite.src.utils import Units


def get_standardised_band_path_object(lattice_vectors, npoints: int) -> BandPath:
    """ ASE standardised band path and a fixed k-grid sampling the path.

    Object contains:
    path, cell=[3x3], special_points={ }, kpts=[n_pointsx3]

    :param lattice_vectors: Lattice vectors stored row wise np array or as [a, b, c]
    :param n_points: Total number of k-points used in the band structure path.
    :return: BandPath
    """
    cell = ase.atoms.Cell(lattice_vectors)
    return cell.bandpath(npoints=npoints)


class ASEBandPath:
    """
    Use when one wishes to use ASE path AND ASE sampling
    along that path. ie `bandpath.kpts`
    """

    def __init__(self, bandpath: BandPath, unit: Units):
        """

        :param bandpath: BandPath(path='GXWKGLUWLK,UX', cell=[3x3],
                                  special_points={GKLUWX}, kpts=[37x3])
        """
        self.bandpath = bandpath
        self.unit = unit
        self.labels = self.labels_from_bandpath()
        self.flattened_k, self.xticks = self.flattened_kpath()

        if len(self.labels) != len(self.xticks):
            print('Labels: ', self.labels)
            print('xticks: ', self.xticks)
            raise ValueError("Number of labels differs from the number of xticks. \n"
                             "This suggests that the ASE k-grid is not dense enough "
                             "See comment in `flattened_kpath`.")

    def get_k_points(self):
        return self.bandpath.kpts

    def get_path(self):
        return self.bandpath.path

    def get_special_points(self) -> dict:
        return self.bandpath.special_points

    def labels_from_bandpath(self, prettify=True) -> list:
        """ Convert ASE path into labels.
        """
        labels = labels_from_ase_bandpath(self.bandpath.path)

        if prettify:
            unicode_gamma = '\u0393'
            return list(map(lambda x: x.replace('Gamma', unicode_gamma), labels))

        return labels

    def flattened_kpath(self) -> tuple:
        """ Flatten a k-path for use in Band Structure Plotting
        """
        return flattened_kpath(self.bandpath.kpts, self.bandpath.special_points)
