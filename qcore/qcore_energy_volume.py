""" Energy vs volume curve for QE
"""
import numpy as np
from pathlib import Path
import os
import json

from ase.atoms import Atoms
from ase.calculators.espresso import Espresso

from tb_lite.crystal_references.crystal_systems import bulk_materials
from qcore.bulk_band_structures.energy_volume_curve_calculator import QcoreEnergyVolumeCurve

import delta_factor
from delta_factor.energy_volume import atoms_range, pooled_runner, run_ase_calculator

from espresso.inputs import set_espresso_input
from espresso.espresso_settings import get_dune_qe_settings
from espresso.scf import get_scf_input


def run_ev_calculator(calculator: QcoreEnergyVolumeCurve) -> dict:
    """ Run Qcore EV curve calculator and return results
    """
    # Run an SCF calc
    calculator.scaling = np.array([1])
    calculator.write_input()
    calculator.run()
    result = calculator.scf_result[0]
    scf_name = "EVcurve"
    energy = result[scf_name]["energy"] * 27.2114
    volume = calculator.atoms.get_cell().volume
    return {'volume (Ang^3)': volume, 'total_energy (eV)': energy}


def to_ev_curve_data(result):
    volume = np.array([i['volume (Ang^3)'] for i in result])
    energy = np.array([i['total_energy (eV)'] for i in result])
    return delta_factor.EVCurveData(volume, energy)


if __name__ == '__main__':
    # TODO Rui. Adapt this for Qcore

    # Paths
    dune_root = Path('/scratch/Documents/git/tb_benchmarking')
    scf_json_file = dune_root / 'outputs/qcore_scf/results.json'
    run_dir_root = dune_root / 'outputs/qcore_e_vs_v'

    print(f'Qcore E vs V calculation. Running in {run_dir_root}')

    # Load all previously-converged espresso SCF results
    if os.path.isfile(scf_json_file):
        with open(scf_json_file) as fid:
            scf_results = json.load(fid)

    # Build one working instance ...
    material = 'silicon'

    # Espresso equilibrium defaults
    # converged_settings = get_scf_input(scf_results[material])
    # TODO(Alex/Rui) Change the k-sampling?
    # k_grid = converged_settings['k_grid']
    k_grid = [4, 4, 4]
    equilibrium_atoms: Atoms = bulk_materials[material]

    # Calculator with fixed settings (unless we decide to also vary k-grid)
    calculator = QcoreEnergyVolumeCurve(equilibrium_atoms, np.array([1]), kpts=k_grid)

    # Test single run
    # print(run_ev_calculator(calculator))

    # quit()

    # TODO Rui. Check each `atoms` contains the correct set of lattice vectors

    # Range of lattice constants
    lattice_multipliers = np.linspace(0.8, 1.2, num=11, endpoint=True)
    # lattice_multipliers = [1.]
    atoms_list = atoms_range(equilibrium_atoms, lattice_multipliers)

    # Attach calculators to atoms
    calculators = []
    for a in atoms_list:
        calculators.append(QcoreEnergyVolumeCurve(a, np.array([1]), kpts=k_grid))

    # Change n_processes to run more instances of QE at the same time
    results = pooled_runner(calculators, run_ev_calculator, n_processes=1)

    ev_curve_data = to_ev_curve_data(results)
    print(ev_curve_data)
    # TODO Rui Try fitting/computing the delta-factors
