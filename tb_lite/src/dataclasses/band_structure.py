""" Class for holding results of a  band structure calculation.
Should be independent of the code and parser.
"""
from typing import Optional, Union
import numpy as np
from pathlib import Path

from tb_lite.src.ase_bandpath import ASEBandPath
from tb_lite.src.band_utils import high_symmetry_point_indices


class BandData:
    """
    BandData should be able to hold data on:
     1. k-points in the band path
     2. Bands (eigenvalues)
     3. A flattened k-grid or abscissa
     4. The band path
     5. Where the high symmetry points are situated on the abscissa.

     Data 3 - 5 are themselves packaged in band_path.
     Better to inject an object rather than initialise in the class.
    """
    def __init__(self,
                 k_points: np.ndarray,
                 bands: np.ndarray,
                 band_path: ASEBandPath,
                 n_occupied_bands: Optional[Union[int, float]] = None,
                 fermi_level: Optional[float] = None
                 ):
        self.k_points = k_points
        self.bands = bands
        self.band_path = band_path
        self.n_k_points, self.n_bands = self.bands.shape
        self.fermi_level = fermi_level
        self.n_occupied_bands = int(np.rint(n_occupied_bands))
        self.partial_occupancy = False

        # If using smearing, one might not expect integer values
        # If the integer value is within 99% of the float value, assume it's fine
        # to define the band gap (arbitrary choice)
        if 1 - (n_occupied_bands / float(self.n_occupied_bands)) > 0.01:
            self.partial_occupancy = True
  
        # Python indexing at 0
        self.i_vbm = self.n_occupied_bands - 1
        self.i_cbm = self.i_vbm + 1

    def get_k_point_indices(self, k_point_v, k_point_c) -> tuple:
        """
        Note, this will return multiple indices if a given k-point is
        present more than once.
        """
        ik_v = np.where((self.k_points == k_point_v).all(axis=1))[0]
        ik_c = np.where((self.k_points == k_point_c).all(axis=1))[0]
        return ik_v, ik_c

    def valence_band_maximum(self) -> float:
        return np.amax(self.bands[:, self.i_vbm])

    def band_gap(self, k_point_v, k_point_c) -> float:
        ik_v, ik_c = self.get_k_point_indices(k_point_v, k_point_c)
        # If the requested k-point is degenerate, take the first instance
        if len(ik_v) > 1:
            ik_v = ik_v[0]
        if len(ik_c) > 1:
            ik_c = ik_c[0]
        return self.bands[ik_c, self.i_cbm] - self.bands[ik_v, self.i_vbm]

    def fundamental_band_gap(self) -> float:
        ik_v = np.argmax(self.bands[:, self.i_vbm])
        ik_c = np.argmin(self.bands[:, self.i_cbm])
        return self.bands[ik_c, self.i_cbm] - self.bands[ik_v, self.i_vbm]

    def set_bands_zeropoint(self, zero_point: float):
        """ Shift the bands to a new zero point.

        :param bands: Bands with shape(n_k_points, n_bands).
        :param zero_point: New zero point of the energy.
        :return: shifted_bands
        """
        if not type(zero_point) in [float, np.float64]:
            raise ValueError(f'Expect zero_point to be a float, not type {type(zero_point)}')
        self.bands -= zero_point

    def print(self):
        k_v = np.amax(self.bands[:, self.i_vbm])
        k_c = np.amin(self.bands[:, self.i_cbm])

        print(f'Valence Band Maximum at k = {k_v} (eV): ')
        print(f'Conduction Band Minimum at k = {k_c} (eV): ')
        
        if self.partial_occupancy:
            print('System has partial occupancy. band-gap is ambiguous')
        else:
            print(f'Fundamental Band Gap (eV): {self.fundamental_band_gap()}')

    def dump_gnuplot(self, path: Path):
        """ Dump to file that's easily plot by GNUPlot

        # k-point (1D)   band 1    band 2    ... band N

        plot 'path' u 1:2 w l
        replot 'path' u 1:3 w l
        ...
        """
        k_and_bands = np.empty(shape=(self.n_k_points, self.n_bands + 1))
        k_and_bands[:, 0] = self.band_path.flattened_k
        k_and_bands[:, 1:] = self.bands[:, :]
        np.savetxt(path, k_and_bands)

    def bands_at_high_symmetry_points(self) -> np.ndarray:
        """ Given an  array of bands along a k-path, return
        an array of bands only at high symmetry points.
        """
        bandpath = self.band_path.bandpath
        indices = high_symmetry_point_indices(bandpath.kpts, bandpath.special_points)
        return self.bands[indices, :]
