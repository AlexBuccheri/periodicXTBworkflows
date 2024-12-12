""" DFTB+ Calculators
"""
from typing import Optional
import os

from ase.atoms import Atoms
from ase.io.dftb import write_dftb

from tb_lite.src.dftb_input import DftbInput
from tb_lite.src.calculators.iocalculator import IOCalculator
from tb_lite.src.parsers.dftb_parsers import parse_dftb_output
from tb_lite.src.classes.runner import BinaryRunner, SubprocessRunResults


class DFTBIOCalculator(IOCalculator):

    binary_name = 'dftb+'

    def __init__(self, directory, input: DftbInput, atoms: Optional[Atoms] = None):
        self.input = input
        self.atoms = atoms
        super(DFTBIOCalculator, self).__init__(directory)

    def write_input(self, atoms: Optional[Atoms] = None):
        """
        If atoms is passed as an argument, it takes precedent over internal value.
        """
        _atoms = atoms

        # Fall back to internal value
        if _atoms is None:
            _atoms = self.atoms

        if _atoms is None:
            raise ValueError('Must either initialise DFTBIOCalculator with atoms, or '
                             'pass as an argument to `write_input`')

        write_dftb(os.path.join(self.directory, "geometry.gen"), _atoms)

        with open(os.path.join(self.directory, "dftb_in.hsd"), "w") as fid:
            fid.write(self.input.generate_dftb_hsd())

    def run(self, run_cmd=None, omp=1, time_out=600) -> SubprocessRunResults:
        if run_cmd is None:
            run_cmd = ['./']
        runner = BinaryRunner(binary=self.binary_name,
                              run_cmd=run_cmd,
                              omp_num_threads=omp,
                              directory=self.directory,
                              time_out=600)
        process_result = runner.run()
        return process_result

    def parse_result(self) -> dict:
        with open(os.path.join(self.directory, "detailed.out"), "r") as fid:
            output_str = fid.read()
        return parse_dftb_output(output_str)
