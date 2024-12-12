""" Energy vs volume curve for QE

python espresso/energy_volume.py
"""
import copy

import numpy as np
from pathlib import Path
import os
import json
from typing import List

from ase.atoms import Atoms
from ase.calculators.espresso import Espresso

from tb_lite.crystal_references.crystal_systems import bulk_materials

from delta_factor.energy_volume import atoms_range, pooled_runner, run_ase_calculator, repack_results_as_np
from espresso.inputs import set_espresso_input
from espresso.espresso_settings import get_dune_qe_settings
from espresso.scf import get_scf_input


def calculators_for_material(run_dir_root: Path,
                             scf_results: dict,
                             material: str,
                             lattice_multipliers: np.ndarray) -> List[Atoms]:
    """
    """
    # Espresso equilibrium defaults
    env_settings = get_dune_qe_settings(run_dir_root, material)
    converged_settings = get_scf_input(scf_results[material])
    specific_settings = {'ecutwfc': converged_settings['ecut'],
                         'pseudo_dir': env_settings.pseudo_dir,
                         'conv_thr': 1.e-6}

    k_grid = converged_settings['k_grid']
    qe_input = set_espresso_input(material, **specific_settings)
    equilibrium_atoms: Atoms = bulk_materials[material]

    # Calculator with fixed settings (unless we decide to also vary k-grid)
    base_calculator = Espresso(directory=env_settings.run_dir,
                               input_data=qe_input,
                               kspacing=None, kpts=k_grid,
                               pseudopotentials=env_settings.pseudos)

    # Range of lattice constants
    atoms_list = atoms_range(equilibrium_atoms, lattice_multipliers)

    # Attach calculators to atoms
    calculators = []
    for i, a in enumerate(atoms_list):
        calculator = copy.deepcopy(base_calculator)
        # Ensure each run occurs in a subdirectory
        calculator.directory += f'/{i}'
        a.calc = calculator
        calculators.append(a)

    return calculators


if __name__ == '__main__':

    # Paths
    dune_root = Path('/users/sol/abuccheri/packages/tb_benchmarking')
    scf_json_file = dune_root / 'outputs/espresso_scf/results.json'
    run_dir_root = dune_root / 'outputs/espresso_e_vs_v'
    n_processes = 8

    print(f'Espresso E vs V calculation. Running in {run_dir_root}')

    # Load all previously-converged espresso SCF results
    if os.path.isfile(scf_json_file):
        with open(scf_json_file) as fid:
            scf_results = json.load(fid)

    materials = [name for name, result in scf_results.items() if result['converged'] is True]
    print('Materials listed as converged:', materials, '\n')

    lattice_multipliers = np.linspace(0.8, 1.2, num=11, endpoint=True)

    # Pool doesn't return results until ALL processes are finished, so batch up
    # according to material. Not optimally efficient but will do.
    for material in materials:
        print(f'Running E vs V for {material}')
        calculators = calculators_for_material(run_dir_root, scf_results, material, lattice_multipliers)
        results = pooled_runner(calculators, run_ase_calculator, n_processes=n_processes)
        with open(run_dir_root / material / "e_vs_v.dat", 'w') as fid:
            np.savetxt(fid, repack_results_as_np(results),  header=f'QE calc. on {material}\nVol(Ang^3)   Energy (eV)')
