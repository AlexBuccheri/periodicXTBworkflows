"""
Run from root:
python3 tb_lite/molecular_crystals/run_molecular_evsv.py
"""
from __future__ import annotations

import copy
import glob
import os
import re
from pathlib import Path

from ase.atoms import Atoms
import numpy as np
from typing import List

from tb_lite.src.dftb_input import DftbInput, Hamiltonian, Options
from tb_lite.src.calculators.dftb_calculator import DFTBIOCalculator
from tb_lite.src.utils import get_project_root
from tb_lite.src.parsers.parsers import parse_x23_xyz_to_ase

from delta_factor.energy_volume import atoms_range, pooled_runner, repack_results_as_np


def make_run_dir(root: str | Path, N_children: int):
    """ Makes top level run directory for a molecule,
    and all subdirectories required for each total energy calculation.

    Note, one could split the naming and directory generation, but that's
    overkill for what we want.

    :param root: Root directory for molecule calculation to run in.
    """
    # Root directory for molecule
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)

    # Subdirectories for each energy calculation
    for i in range(0, N_children):
        run_dir = Path(directory, str(i))
        run_dir.mkdir(parents=True, exist_ok=True)

    return


def calculators_for_material(molecule_root: Path,
                             equilibrium_atoms: Atoms,
                             input_settings: dict,
                             lattice_multipliers: np.ndarray) -> List[DFTBIOCalculator]:
    """ Define a list of calculators with the same input settings
    but differing in lattice constants.
    """
    # TB Lite 1 calculator with fixed settings, no atoms attached
    dftb_input = DftbInput(**input_settings)
    base_calculator = DFTBIOCalculator(directory=molecule_root, input=dftb_input)

    # Atoms instances for each lattice constant
    atoms_list = atoms_range(equilibrium_atoms, lattice_multipliers)

    # Define a list of calculators (our custom calc, not ASE's)
    calculators = []
    for i, modified_atoms in enumerate(atoms_list):
        calculator = copy.deepcopy(base_calculator)
        calculator.directory = Path(molecule_root, str(i))
        calculator.atoms = modified_atoms
        calculators.append(calculator)

    return calculators


def run_dftb_custom_calculator(calculator: DFTBIOCalculator) -> dict:
    """ Run custom DFTB+ calculator (for TB lite) and return E vs V result.
    """
    calculator.write_input()
    # Only use 1 thread per calculation, such that task farming does not oversubscribe
    calculator.run(omp=1)
    results: dict = calculator.parse_result()
    volume = calculator.atoms.get_cell().volume
    return {'volume': volume, 'total_energy': results['total_energy'],
            'volume_unit': 'Ang^3', 'total_energy_unit': 'eV'}


if __name__ == '__main__':

    # Paths w.r.t. project root
    project_root = get_project_root()
    input_root = Path(project_root, 'data/x23')

    # Try [1,1,1] [4,4,4] and [8,8,8]
    k_grid = [4, 4, 4]
    k_str = "".join(str(k) + '_' for k in k_grid)[:-1]
    run_dir_root = Path(project_root, 'outputs/tblite1/x23_e_vs_v', k_str)
    run_dir_root.mkdir(parents=True, exist_ok=True)

    # Find solids
    all_xyz_files = glob.glob(input_root.as_posix() + "/*.xyz")
    molecule_files = []
    for file in all_xyz_files:
        if re.search(r'^(?!.*_g\.xyz$).*\.xyz$', file):
            molecule_files.append(file)

    n_processes = 4
    lattice_multipliers = np.linspace(0.8, 1.2, num=11, endpoint=True)
    input_settings = {'hamiltonian': Hamiltonian(method='GFN1-xTB',
                                                 temperature=0,
                                                 scc_tolerance=1.e-6,
                                                 k_grid=k_grid),
                      'options': Options()
                      }

    print(f'TB Lite 1 E vs V Calculation for X23 Dataset. Running in {run_dir_root}')
    print(f'using k-grid: {k_str}')

    # Batched per molecule (so lattice_multipliers.size)
    for file in molecule_files:
        molecule_name = os.path.basename(file).split('.')[0]
        molecule_root = Path(run_dir_root, molecule_name)
        make_run_dir(molecule_root, lattice_multipliers.size)
        print(f'Running E vs V for {molecule_name} in {molecule_root}')
        equilibrium_atoms: Atoms = parse_x23_xyz_to_ase(file)
        calculators = calculators_for_material(molecule_root, equilibrium_atoms, input_settings, lattice_multipliers)
        results = pooled_runner(calculators, run_dftb_custom_calculator, n_processes=n_processes)
        with open(molecule_root / "e_vs_v.dat", 'w') as fid:
            np.savetxt(fid, repack_results_as_np(results),
                       header=f'TBlite1 calc. on {molecule_name}\nVol(Ang^3)   Energy (eV)')
