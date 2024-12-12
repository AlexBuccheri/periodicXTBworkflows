""" Generic Energy-Volume Routines
"""
import warnings
from multiprocessing import Pool
from typing import Callable, List
from copy import deepcopy
import math
from ase.atoms import Atoms
import numpy as np


def atoms_range(equilibrium_atoms: Atoms, lattice_multipliers) -> List[Atoms]:
    """ Create a list of Atoms instances, with lattice vectors
    scaled by lattice multipliers.

    """
    equilibrium_lattice = equilibrium_atoms.get_cell()
    atoms_list = []

    for multiplier in lattice_multipliers:
        atoms = deepcopy(equilibrium_atoms)
        # Proportionally expand the lattice (vectors are stored row-wise)
        lattice = math.pow(multiplier, 1/3.0) * equilibrium_lattice
        atoms.set_cell(lattice, scale_atoms=True)

        # Consistency check
        diff = np.asarray(equilibrium_atoms.get_scaled_positions()) - np.asarray(atoms.get_scaled_positions())
        if np.any(diff > 1.e-10):
            msg = "Fractional positions are not consistent, `atoms` positions are erroneous\n"
            warnings.warn(msg)
            print('Diff:')
            print(diff)

        atoms_list.append(atoms)

    return atoms_list


def pooled_runner(calculations: list, run_func: Callable, n_processes: int) -> list:
    """ Pooled function execution, with dynamic load balancing.

    NOTE. MUST be called from within `if __name__ == '__main__':`

    :param calculations: A list of calculation instances.
    :param run_func: A function that can evaluate (run) a calculation. Cannot be a lambda
    :param n_processes: Number of processes (check wod choice) to run concurrently.
    :return: List of result instances, where a result must be serialisable.
    """
    with Pool(n_processes) as p:
        results = p.map(run_func, calculations)
    return results


def run_ase_calculator(atoms: Atoms) -> dict:
    """ Run ASE calculator and return results
    """
    # Run an SCF calc
    atoms.get_potential_energy()
    # Read results (assume this is true beyond QE calculator)
    atoms.calc.read_results()
    total_energy = atoms.calc.results['energy']
    volume = atoms.get_cell().volume
    return {'volume': volume, 'total_energy': total_energy,
            'volume_unit': 'Ang^3', 'total_energy_unit': 'eV'}


def repack_results_as_np(results: List[dict]):
    """ Repackage results from `run_ase_calculator` in a np array.
    """
    n_results = len(results)
    np_data = np.empty(shape=(n_results, 2))
    for i in range(n_results):
        np_data[i, :] = [results[i]['volume'], results[i]['total_energy']]
    return np_data
