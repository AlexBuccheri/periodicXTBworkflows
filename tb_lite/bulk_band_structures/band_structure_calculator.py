"""Generate Bulk Band Structures using DFTB+ and TB Lite (xTB1, xTB2).
"""
import os.path
from typing import Union, Optional
from pathlib import Path
import shutil

from ase.io.gen import read_gen
from ase.atoms import Atoms
from ase.io.dftb import write_dftb

from tb_lite.src.dataclasses.band_structure import BandData
from tb_lite.src.ase_bandpath import get_standardised_band_path_object, ASEBandPath
from tb_lite.src.dftb_input import KLines, ExplicitKPoints, BandStructureHamiltonian, DftbInput
from tb_lite.src.calculators.iocalculator import IOCalculator
from tb_lite.src.parsers.dftb_parsers import parse_dftb_output, parse_dftb_bands, parse_number_of_occupied_bands
from tb_lite.src.classes.runner import BinaryRunner, SubprocessRunResults
from tb_lite.src.utils import Units

# Path type
path_type = Union[Path, str]


class TBLiteBandStructure(IOCalculator):
    """ Band structure calculator, which expects to have reference SCC charges
    in a provided directory.
    """
    binary_name = 'dftb+'

    def __init__(self, xtb_method: str, scc_dir: path_type, directory: path_type,
                 npoints: Optional[int] = None, points_per_path: Optional[int] = None,
                 ParameterFile: Optional[path_type] = None, run_settings=None):
        """ Initialise files for a band structure calculation, given a prior SCC calculation.

        TODO(Alex) Extend to work with run_settings

        :param xtb_method: XTB Method
        :param scc_dir: SCC calculation directory
        :param directory: Working directory for band structure calculation
        :param npoints: Number of k-points in the band path
        :param ParameterFile: xTB parameter file. If this is passed, xtb_method is ignored.
        xtb_method was not changed to optional as it will break the API everywhere.
        :param run_settings: Run settings for BinaryRunner.
        Passed at initialisation as one will likely want to define the run
        settings and execute at some later point.
        """
        assert xtb_method in ['GFN1-xTB', 'GFN2-xTB']
        self.xtb_method = xtb_method
        self.scc_dir = scc_dir
        self.directory = directory
        self.npoints = npoints
        self.points_per_path = points_per_path
        self.ParameterFile = ParameterFile
        self.atoms: Atoms = read_gen(os.path.join(scc_dir, 'geometry.gen'))

        self.ase_k_grid = True
        if self.ase_k_grid:
            if npoints is None:
                raise ValueError('Missing `npoints`. Must specific total number of k-points to use in'
                                 'the k-grid, when using ASE''s k-sampling')
            self.band_path = get_standardised_band_path_object(self.atoms.get_cell(), self.npoints)
            k_data = ExplicitKPoints(self.band_path.kpts)
        else:
            if points_per_path is None:
                raise ValueError('Missing `points_per_path`. Must specific number of k-points '
                                 'to use in each k-path when NOT using ASE''s k-sampling')
            self.band_path = get_standardised_band_path_object(self.atoms.get_cell())
            k_data = KLines(self.band_path, self.points_per_path)

        # Define xTB model from parameters
        if self.ParameterFile is not None:
            h_bands = BandStructureHamiltonian(k_data, ParameterFile=self.ParameterFile)
        # Use internal xTB parameters, hence specify model
        else:
            h_bands = BandStructureHamiltonian(k_data, method=self.xtb_method)

        self.dftb_in: str = DftbInput(hamiltonian=h_bands).generate_dftb_hsd()

        self.run_settings = run_settings
        super(TBLiteBandStructure, self).__init__(directory)

    def _copy_charges(self):
        if not os.path.isdir(self.scc_dir):
            raise NotADirectoryError(f"Directory contain SCC "
                                     f"calculation does not exist: {self.scc_dir}")

        shutil.copyfile(os.path.join(self.scc_dir, 'charges.bin'),
                        os.path.join(self.directory, 'charges.bin'))

    def write_input(self):
        if not os.path.isdir(self.directory):
            raise NotADirectoryError(f"Directory for band structure "
                                     f"calculation does not exist: {self.directory}")
        # Atoms
        write_dftb(os.path.join(self.directory, "geometry.gen"), self.atoms)

        # Input hsd
        with open(os.path.join(self.directory, 'dftb_in.hsd'), 'w') as fid:
            fid.write(self.dftb_in)

        # Charges
        self._copy_charges()

    def run(self) -> SubprocessRunResults:
        runner = BinaryRunner(binary=self.binary_name,
                              run_cmd=['./'],
                              omp_num_threads=1,
                              directory=self.directory,
                              time_out=600)
        process_result = runner.run()
        return process_result

    def parse_result(self) -> BandData:
        """
        Note, DFTB+ band_tot.dat just gives an index for k-points, not a useful vector.
        :return:
        """
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

        with open(os.path.join(self.directory, 'detailed.out'), "r") as fid:
            detailed_str = fid.read()

        band_energies = parse_dftb_bands(self.directory)
        n_occ = parse_number_of_occupied_bands(detailed_str)
        results = parse_dftb_output(detailed_str)

        return BandData(k_points, band_energies, band_plot_info, n_occupied_bands=n_occ, fermi_level=results['fermi_level'])
