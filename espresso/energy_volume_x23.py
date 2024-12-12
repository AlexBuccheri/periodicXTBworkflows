""" Energy vs Volume for Quantum espresso, but targeting the X23 molecular set

Run from project root:
python3 espresso/energy_volume_x23.py
or
nohup python3 espresso/energy_volume_x23.py > evsv_444_x23.out &
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import List
import numpy as np
import os

from ase.atoms import Atoms
from ase.calculators.espresso import Espresso

from delta_factor.energy_volume import pooled_runner, atoms_range, run_ase_calculator, repack_results_as_np
from tb_lite.src.parsers.parsers import parse_x23_xyz_to_ase

# ---------------------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------------------
rugters_pseudos = {'C': 'c_pbesol_v1.2.uspp.F.UPF',
                   'O': 'o_pbesol_v1.2.uspp.F.UPF',
                   'H': 'h_pbesol_v1.4.uspp.F.UPF',
                   'N': 'n_pbesol_v1.2.uspp.F.UPF'
                   }


def get_pseudos(atoms: Atoms, species_pseudos=None) -> dict[str, str]:
    """ Given an ASE Atoms object, generate a list of pseudo-potentials.

    Expect a returned dict of the form:  {'Si': 'si_pbesol_v1.uspp.F.UPF'}
    """
    if species_pseudos is None:
        species_pseudos = rugters_pseudos
    symbols = list(set(atoms.get_chemical_symbols()))
    symbols = sorted(symbols)
    pseudos = {x.capitalize(): species_pseudos[x] for x in symbols}
    return pseudos


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


def define_ase_espresso_env(machine_settings: dict) -> dict:
    """ Set run command for Espresso when called with ASE.
    Note, pseudo can be specified in the input, hence why not set here.
    """
    env = {'ASE_ESPRESSO_COMMAND': "{binarypath}/pw.x -in PREFIX.pwi > PREFIX.pwo",
           'ESPRESSO_PSEUDO': ''}
    value = env['ASE_ESPRESSO_COMMAND']
    env['ASE_ESPRESSO_COMMAND'] = value.format(binarypath=machine_settings['binary_path'])
    return env


# ---------------------------------------------------------------------------------------
# Machine-specific
# ---------------------------------------------------------------------------------------

dune3 = {'binary_path': '/users/sol/abuccheri/packages/qe-7.1/build/bin',
         'pseudo_dir': '/users/sol/abuccheri/rutgers_pseudos/pbesol'
         }


# ---------------------------------------------------------------------------------------
# Espresso and X23-specific (not reusable)
# ---------------------------------------------------------------------------------------

def calculators_for_material(molecule_root: Path,
                             machine_settings: dict,
                             equilibrium_atoms: Atoms,
                             espresso_settings: dict,
                             lattice_multipliers: np.ndarray,
                             k_grid) -> List[Atoms]:
    """ Define a list of calculators with the same input settings but differing in lattice constants.
    """
    pseudos = get_pseudos(equilibrium_atoms)
    env = define_ase_espresso_env(machine_settings)
    os.environ.update(env)

    # Update relevant espresso inputs with molecule-specific info
    # (note: could ignore the prefix and set pseudo_dir in the env, instead)
    espresso_settings['control']['prefix'] = espresso_settings['control']['prefix'].format(molecule=molecule_root.name)
    espresso_settings['control']['pseudo_dir'] = machine_settings['pseudo_dir']

    # Calculator with fixed settings
    base_calculator = Espresso(directory=molecule_root,
                               input_data=espresso_settings,
                               kspacing=None, kpts=k_grid,
                               pseudopotentials=pseudos)

    # atoms with a range of lattice constants
    atoms_list: List[Atoms] = atoms_range(equilibrium_atoms, lattice_multipliers)

    # Attach calculators to atoms
    calculators = []
    for i, modified_atoms in enumerate(atoms_list):
        calculator = copy.deepcopy(base_calculator)
        # Ensure each run occurs in a subdirectory
        calculator.directory += f'/{i}'
        # Attach and store calculator (technically atoms, with calculator attached)
        modified_atoms.calc = calculator
        calculators.append(modified_atoms)

    return calculators


# Set contains 23 molecules (plus 23 gas phase variants we ignore)
x23_molecules = ['CO2.xyz',
                 'acetic.xyz',
                 'adaman.xyz',
                 'ammonia.xyz',
                 'anthracene.xyz',
                 'benzene.xyz',
                 'cyanamide.xyz',
                 'cytosine.xyz',
                 'ethcar.xyz',
                 'formamide.xyz',
                 'hexamine.xyz',
                 'hexdio.xyz',
                 'imdazole.xyz',
                 'naph.xyz',
                 'oxaca.xyz',
                 'oxacb.xyz',
                 'pyrazine.xyz',
                 'pyrazole.xyz',
                 'succinic.xyz',
                 'triazine.xyz',
                 'trioxane.xyz',
                 'uracil.xyz',
                 'urea.xyz']

# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------

if __name__ == '__main__':

    # User settings
    # DUNE ROOT
    dune_root = Path('/users/sol/abuccheri/packages/tb_benchmarking')
    n_processes = 4
    # These tests show that [4,4,4] is required for several systems (> 1 meV / atom error)
    k_grid = [4, 4, 4]

    # Inputs
    input_root = Path(dune_root, 'data/x23')
    molecule_files = x23_molecules

    # Run directory
    k_str = "".join(str(k) + '_' for k in k_grid)[:-1]
    run_dir_root = Path(f'outputs/espresso_e_vs_v_x23/{k_str}')
    run_dir_root.mkdir(parents=True, exist_ok=True)

    # Base espresso settings
    lattice_multipliers = np.linspace(0.8, 1.2, num=11, endpoint=True)
    base_espresso_settings = {'control':
                                  {'prefix': '{molecule}_PBESOL',
                                   'calculation': 'scf',
                                   'pseudo_dir': 'REPLACE-ME',
                                   'verbosity': 'high'},
                              'electrons':
                                  {'conv_thr': 1.e-6},
                              'system':
                                  {'ecutwfc': 100,            # Ryd ==~ 1360.57 eV.  Above the 1000 eV used in the X23 paper.
                                   'vdw_corr': 'grimme-d3'},  # (zero damping) - Default
                              }

    # Batched per molecule
    for file_name in molecule_files:
        molecule_name = file_name.split('.')[0]
        molecule_root = Path(run_dir_root, molecule_name)
        make_run_dir(molecule_root, lattice_multipliers.size)

        print(f'Running E vs V for {molecule_name} in {molecule_root}')

        equilibrium_atoms: Atoms = parse_x23_xyz_to_ase(Path(input_root, file_name))
        calculators = calculators_for_material(molecule_root, dune3, equilibrium_atoms, base_espresso_settings,
                                               lattice_multipliers, k_grid)

        results = pooled_runner(calculators, run_ase_calculator, n_processes=n_processes)

        with open(molecule_root / "e_vs_v.dat", 'w') as fid:
            np.savetxt(fid, repack_results_as_np(results),
                       header=f'QE calc. on {molecule_name}\nVol(Ang^3)   Energy (eV)')


# Example calculation with a single calculator, and no pooling
# if __name__ == '__main__':
#
#     # User settings
#     # DUNE ROOT
#     dune_root = Path('/users/sol/abuccheri/packages/tb_benchmarking')
#     n_processes = 1
#     # Move to [4, 4, 4] once tested
#     k_grid = [1, 1, 1]
#
#     # Inputs
#     input_root = Path(dune_root, 'data/x23')
#
#     # Run directory
#     k_str = "".join(str(k) + '_' for k in k_grid)[:-1]
#     run_dir_root = Path(f'outputs/espresso_e_vs_v_x23/{k_str}')
#     run_dir_root.mkdir(parents=True, exist_ok=True)
#
#     # Base espresso settings
#     lattice_multipliers = np.linspace(0.8, 1.2, num=11, endpoint=True)
#     base_espresso_settings = {'control':
#                                   {'prefix': '{molecule}_PBESOL',
#                                    'calculation': 'scf',
#                                    'pseudo_dir': 'REPLACE-ME',
#                                    'verbosity': 'high'},
#                               'electrons':
#                                   {'conv_thr': 1.e-6},
#                               'system':
#                                   {'ecutwfc': 100},  # Ryd ==~ 1360.57 eV.  Above the 1000 eV used in the X23 paper.
#                               # This is not valid QE key, and should get popped
#                               # before passing to the settings dict
#                               'k_grid': k_grid}
#
#     molecule_files = ['CO2.xyz']
#
#     molecule_name = 'CO2'
#     molecule_root = Path(run_dir_root, molecule_name)
#     molecule_root.mkdir(parents=True, exist_ok=True)
#     print(f'Running E vs V for {molecule_name} in {molecule_root}')
#
#     equilibrium_atoms: Atoms = parse_x23_xyz_to_ase(Path(input_root, molecule_files[0]))
#     pseudos = get_pseudos(equilibrium_atoms)
#     env = define_ase_espresso_env(dune3)
#     os.environ.update(env)
#
#     # Update relevant espresso inputs with molecule-specific info
#     base_espresso_settings['control']['prefix'] = base_espresso_settings['control']['prefix'].format(molecule=molecule_root.name)
#     base_espresso_settings['control']['pseudo_dir'] = dune3['pseudo_dir']
#     k_grid = base_espresso_settings.pop('k_grid')
#
#     # Calculator with fixed settings
#     base_calculator = Espresso(directory=molecule_root,
#                                input_data=base_espresso_settings,
#                                kspacing=None, kpts=k_grid,
#                                pseudopotentials=pseudos)
#
#     equilibrium_atoms.calc = base_calculator
#     equilibrium_atoms.get_potential_energy()
#     equilibrium_atoms.calc.read_results()
#     print(equilibrium_atoms.calc.results)
