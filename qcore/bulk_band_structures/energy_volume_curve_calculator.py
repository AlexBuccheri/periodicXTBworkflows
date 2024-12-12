import copy

from qcore import default_settings
from typing import List
from ase.atoms import Atoms

from qcoreutils.src.utils import Set
from qcoreutils.src.run_qcore import run_qcore
from qcoreutils.src import qcore_input_strings
from tb_lite.src.calculators.iocalculator import IOCalculator

import numpy as np

from delta_factor import EVCurveData


def qcore_input(atoms: Atoms,
                scaling: np.array,
                kpts: List[int],
                qcore_setting=default_settings):
    qcore_setting['monkhorst_pack'] = Set(kpts)

    input_files = []
    for scale in scaling:
        new_atoms = copy.deepcopy(atoms)
        new_atoms.cell = scale * np.array(atoms.cell)
        positions = scale * np.array(new_atoms.positions)
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

        named_result = "EVcurve"

        input_string = qcore_input_strings.xtb_input_string(crystal, default_settings,
                                                            named_result=named_result)

        input_files.append(input_string)

    return input_files


class QcoreEnergyVolumeCurve(IOCalculator):
    """ Energy vs Volume curve calculator, which expects to have reference SCC charges
    in a provided directory.
    """
    binary_name = 'qcore'

    def __init__(self, system: Atoms, scaling: np.array, kpts: List[int]):
        """ Initialise files for an energy vs volume curve calculation

        :param systems: systems in ASE Atoms form
        :param scaling: a list of scaling parameters for the cell volume
        :param kpts: K-points for SCC calculation
        """

        self.kpts = kpts
        self.atoms: Atoms = system
        self.scaling: np.array = scaling
        self.scf_result = None
        self.qcore_in = None

        self.run_settings = None

    def write_input(self):
        self.qcore_in = qcore_input(self.atoms, self.scaling, self.kpts)

    def run(self) -> [dict]:
        result = []
        for i in self.qcore_in:
            result.append(run_qcore(self.qcore_in))

        self.scf_result = result
        return self.scf_result

    def parse_result(self) -> EVCurveData:

        if self.scf_result is None:
            raise NotImplementedError("SCC has not been carried out for this system")

        volume = []
        energy = []

        ref_volume = self.atoms.get_volume()

        for index, result in enumerate(self.scf_result):
            scf_name = "EVcurve"
            xtb_result = result[scf_name]
            energy.append(xtb_result["energy"])
            volume.append(self.scaling[index] * ref_volume)

        return EVCurveData(np.array(volume), np.array(energy))
