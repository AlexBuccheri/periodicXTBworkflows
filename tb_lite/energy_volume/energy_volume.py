""" Energy vs volume curve for TB lite.

To execute from the project root:
```
python3 tb_lite/energy_volume/energy_volume.py
```
"""
import copy
import numpy as np
from pathlib import Path
import os
import json
from typing import List
import sys
from ase.atoms import Atoms


from delta_factor.energy_volume import atoms_range, pooled_runner, repack_results_as_np

from tb_lite.crystal_references.crystal_systems import bulk_materials
from tb_lite.src.utils import get_project_root
from tb_lite.src.dftb_input import DftbInput, Hamiltonian
from tb_lite.src.calculators.dftb_calculator import DFTBIOCalculator


def get_converged_k_grid(results: dict) -> List[float]:
    """ Get converged k-grid from TB lite results of a single material.

    Assumes converged k-grid in each case
    TODO(Alex) Should add converged to the TB lite results
    """
    major, minor, _, _, _ = sys.version_info
    assert major == 3 and minor >= 7, "Require python 3.7"
    last_result = results['energy_vs_k'][-1]
    return last_result['k_sampling']


def calculators_for_material(run_dir_root: Path,
                             scf_results: dict,
                             input_settings: dict,
                             material: str,
                             lattice_multipliers: np.ndarray) -> List[DFTBIOCalculator]:
    """
    """
    # TODO(Alex) Move this responsibility elsewhere
    # Would be nice to make all the run dirs separately
    # Material directory
    directory = Path(run_dir_root, material)
    directory.mkdir(parents=True, exist_ok=True)

    # Converged TB lite settings
    converged_k_grid = get_converged_k_grid(scf_results[material])
    # Update converged k-grid
    input_settings["k_grid"] = converged_k_grid

    # TB Lite 1 calculator with fixed settings (unless we decide to also vary xk-grid)
    dftb_input = DftbInput(hamiltonian=Hamiltonian(**input_settings))
    equilibrium_atoms: Atoms = bulk_materials[material]
    base_calculator = DFTBIOCalculator(directory=directory, input=dftb_input)

    # Range of lattice constants
    atoms_list = atoms_range(equilibrium_atoms, lattice_multipliers)

    # Define a list of calculators (our custom, not ASE)
    calculators = []
    for i, modified_atoms in enumerate(atoms_list):
        calculator = copy.deepcopy(base_calculator)
        # Ensure each run of the same material occurs in a subdirectory
        run_dir = Path(calculator.directory, str(i))
        run_dir.mkdir(parents=True, exist_ok=True)
        calculator.directory = run_dir
        # Update atoms
        calculator.atoms = modified_atoms
        calculators.append(calculator)

    return calculators


def run_dftb_custom_calculator(calculator: DFTBIOCalculator) -> dict:
    """ Run custom DFTB+ calculator (for TB lite) and return E vs V result.
    """
    calculator.write_input()
    # Only use 1 thread per calculation, such that task farming does not oversubscribe
    process_result = calculator.run(omp=1)
    # print("STDOUT\n", process_result.stdout)
    results: dict = calculator.parse_result()
    volume = calculator.atoms.get_cell().volume
    return {'volume': volume, 'total_energy': results['total_energy'],
            'volume_unit': 'Ang^3', 'total_energy_unit': 'eV'}


if __name__ == '__main__':

    # Paths w.r.t. project root
    project_root = get_project_root()
    scf_json_file = Path(project_root, 'outputs/tblite1/scc/results.json')
    run_dir_root = Path(project_root, 'outputs/tblite1/e_vs_v')
    run_dir_root.mkdir(parents=True, exist_ok=True)
    n_processes = 4

    print(f'TB Lite 1 E vs V Calculation. Running in {run_dir_root}')

    # Load all previously-converged SCF results
    if os.path.isfile(scf_json_file):
        with open(scf_json_file) as fid:
            scf_results = json.load(fid)
            input_settings = scf_results.pop("input_settings")
            total_energy_tol = scf_results.pop("total_energy_tol")
            # Disregard non-keyword
            input_settings.pop("units")
    else:
        FileNotFoundError(f'File does not exist: {scf_json_file}')

    materials = [name for name, result in scf_results.items()]
    print('Materials listed in TBlite SCC results file:', materials, '\n')

    lattice_multipliers = np.linspace(0.8, 1.2, num=11, endpoint=True)

    # Pool doesn't return results until ALL processes are finished, so batch up
    # according to material (in case a calculation fails). Not optimally efficient but will do.
    for material in materials:
        print(f'Running E vs V for {material}')
        calculators = calculators_for_material(run_dir_root, scf_results, input_settings, material, lattice_multipliers)
        results = pooled_runner(calculators, run_dftb_custom_calculator, n_processes=n_processes)
        with open(run_dir_root / material / "e_vs_v.dat", 'w') as fid:
            np.savetxt(fid, repack_results_as_np(results),  header=f'TBlite1 calc. on {material}\nVol(Ang^3)   Energy (eV)')






