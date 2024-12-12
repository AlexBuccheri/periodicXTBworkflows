"""Generate Bulk Band Structures using DFTB+ and TB Lite (xTB1, xTB2).
"""
from qcore import default_settings
from typing import Union, Optional, List
from pathlib import Path

from ase.dft.kpoints import BandPath
from ase.atoms import Atoms

from qcoreutils.src.utils import Set
from qcoreutils.src.run_qcore import run_qcore
from qcoreutils.src import qcore_input_strings

# Path type
from tb_lite.src.dataclasses.band_structure import BandData
from tb_lite.src.ase_bandpath import get_standardised_band_path_object, ASEBandPath
from tb_lite.src.calculators.iocalculator import IOCalculator


path_type = Union[Path, str]

import collections
import numpy as np

def parse_qcore_bands(scf_result: dict):
    named_result = "band"

    band_result = scf_result[named_result]
    n_band_paths = band_result["n_band_paths"]

    band_eigenvalues = []
    band_occupations = []
    for i in range(n_band_paths):
        band_path_key = "band_path_" + str(i)
        sample_result = band_result[band_path_key]["sample_0"]['k_channel_0']
        eigenvalues = sample_result["eigenvalues"]
        band_eigenvalues.append(eigenvalues)
        occupations = sample_result["occupations"]
        band_occupations.append(occupations)

    return np.array(band_eigenvalues), band_occupations


def qcore_path(path_kpts: np.ndarray):
    assert path_kpts.shape[1] == 3

    result = []

    for i in range(path_kpts.shape[0]):
        qcore_path.append([list(path_kpts[i]), list(path_kpts[i])])

    return result


def qcore_input(atoms: Atoms,
                bz_path: BandPath,
                kpts: List[int],
                qcore_setting=default_settings):
    qcore_setting['monkhorst_pack'] = Set(kpts)

    positions = atoms.positions

    position_key = 'xyz'
    species = atoms.symbols

    lattice_parameters = {'lattice_vectors': np.array(atoms.cell),
                          'volume': atoms.get_volume()}

    crystal = {position_key: positions,
               'species': species,
               'lattice_parameters': lattice_parameters,
               'space_group': None,
               'bravais': None,
               'n_atoms': len(species)}

    named_result = "band"

    path_checkpoints = bz_path.kpts
    assert path_checkpoints.shape[0] > 1

    path = qcore_path(path_checkpoints)

    default_band_structure_settings = collections.OrderedDict([
        ('n_samples', Set(1)),
        ('symmetry_points', Set(path))
    ])

    input_string = qcore_input_strings.xtb_input_string(crystal, default_settings,
                                                        named_result=named_result,
                                                        external_command='band_structure',
                                                        external_settings=default_band_structure_settings)

    return input_string


class QcoreBandStructure(IOCalculator):
    """ Band structure calculator, which expects to have reference SCC charges
    in a provided directory.
    """
    binary_name = 'qcore'

    def __init__(self, system: Atoms, directory: path_type, kpts: List[int],
                 npoints: Optional[int] = None, points_per_path: Optional[int] = None):
        """ Initialise files for a band structure calculation

        :param systems: systems in ASE Atoms form
        :param directory: Working directory for band structure calculation
        :param kpts: K-points for SCC calculation
        :param npoints: Number of k-points in the band path
        :param run_settings: Run settings for BinaryRunner.
        Passed at initialisation as one will likely want to define the run
        settings and execute at some later point.
        """

        self.directory = directory
        self.npoints = npoints
        self.kpts = kpts
        self.points_per_path = points_per_path
        self.atoms: Atoms = system
        self.scf_result = None

        self.ase_k_grid = True

        self.band_path = get_standardised_band_path_object(self.atoms.get_cell())

        self.qcore_in = None

        self.run_settings = None
        super(QcoreBandStructure, self).__init__(directory)

    def write_input(self):
        self.qcore_in = qcore_input(self.atoms, self.band_path, kpts=self.kpts)

    def run(self) -> dict:
        self.scf_result = run_qcore(self.qcore_in)
        return self.scf_result

    def parse_result(self) -> BandData:

        if self.ase_k_grid:
            k_points = self.band_path.kpts
            band_plot_info = ASEBandPath(self.band_path, unit=Units.eV)
        else:
            # To be consistent with DFTB+, one has to ensure line segments do not include their
            # starting points but their end points, which is the opposite of what I have done in
            # flattened_kpath. Therefore, it's easier to explicitly supply the k-points to the code
            # using those given by ASE.

            # k_points = generate_kpoints(self.band_path.path, self.band_path.special_points)
            raise NotImplementedError("generate_kpoints (n_kpts, 3) given an ASE path, requires implementing")

        if self.scf_result is None:
            raise NotImplementedError("SCC has not been carried out for this system")
        band_energies, occupations = parse_qcore_bands(self.scf_result)

        n_occ = len(np.nonzero(occupations[0])[0])

        fermi_level = float(np.max(band_energies[:, n_occ - 1]))

        return BandData(k_points, band_energies, band_plot_info, n_occupied_bands=n_occ, fermi_level=fermi_level)
