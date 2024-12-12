""" Converge GS of bulk materials with QE.
"""
import os.path
import json
import numpy as np
from typing import List

from espresso.espresso_settings import EspressoSettings, rutgers_pseudos
from espresso.inputs import set_espresso_input
from espresso.scf import run_scf_calculation

from tb_lite.crystal_references.crystal_systems import bulk_materials
from tb_lite.src.classes.convergence import Convergence
from tb_lite.unused_scc_prototypes.parse_convergence import parse_converged_kgrid
from tb_lite.src.parsers.espresso_parsers import parse_espresso_total_energy


def k_grid_range(k_grid, n: int, stride=2):
    """
    Given some k-grid, give some range before and after it, for convergence
    :return:
    """
    k_grids = [k_grid]
    for i in range(1, n + 1):
        k_grids.append([k + i * stride for k in k_grid])
    return k_grids


def collate_variables(variable_a, variable_b):
    """ Concatenate all permutations of the two input variables
    :return:
    """
    xx, yy = np.meshgrid(np.arange(0, len(variable_a)),
                         np.arange(0, len(variable_b)))
    indices = np.vstack([xx.ravel(), yy.ravel()]).T
    return indices


def store_result(results_file, qe_settings: EspressoSettings, label: str, convergence: Convergence):
    """ Stores result to JSON, amending the existing output file.

    :param results_file:
    :param qe_settings:
    :param label:
    :param convergence:
    :return:
    """
    # Current calculation result
    result[label] = {'directory': qe_settings.run_dir,
                     'input': convergence.input[-1],
                     'total_energy': {'value': convergence.output[-1], 'unit': 'eV'}
                     }

    # Parse existing results
    with open(results_file) as fid:
        results = json.load(fid)

    # Update
    results[material].update(result)
    results[material].update({'converged': convergence.has_converged()})

    # Dump back to file
    with open(results_file, 'w', encoding='utf-8') as fid:
        json.dump(results, fid, ensure_ascii=False, indent=4)


def initialise_json_results(json_file) -> bool:
    """ Initialise JSON file

    :param json_file: JSON file name, prepended by full path
    :return file_exists: Whether the file already existed
    """
    file_exists = os.path.isfile(json_file)
    if file_exists:
        return file_exists

    # Initialise file
    results = {material: {'converged': 'Not Run'} for material in list(bulk_materials)}
    with open(json_file, 'w', encoding='utf-8') as fid:
        json.dump(results, fid, ensure_ascii=False, indent=4)

    return file_exists


def materials_to_run(file_existed: bool, json_file) -> List[str]:
    """ Get a list of materials to run.

    :param file_existed: Has the JSON output file been initialised/exists?
    :param json_file: JSON file name, prepended by full path
    :return materials: List of materials to run
    """
    if file_existed:
        # Restart, running those not converged, or not run yet
        if os.path.isfile(json_file):
            with open(json_file) as fid:
                results = json.load(fid)
        materials = [name for name, result in results.items()
                     if result['converged'] is False or result['converged'] == 'Not Run']
    else:
        # Run all
        materials = [name for name in bulk_materials.keys()]

    # Remove troublesome materials that crash QE
    materials.remove('pbte')
    materials.remove('copper')

    return materials


if __name__ == "__main__":

    # Settings from TB Lite
    json_file = "outputs/tb_lite_xtb1/scc_results/data.json"
    # NOTE Target delta used in TB lite calculations is 1.e-4.
    target_delta = 0.001
    converged_k_grids, unconverged_k_grids = parse_converged_kgrid(json_file, target_delta=target_delta)

    # Espresso env Settings
    dune3 = {'binary_path': '/users/sol/abuccheri/packages/qe-7.1/build/bin',
             'pseudo_dir': '/users/sol/abuccheri/rutgers_pseudos/pbesol',
             'pseudos': {},
             'job_root': '/users/sol/abuccheri/packages/tb_benchmarking/outputs/espresso_scf'
             }

    # Set results file and materials to run
    # NOTE, if the JSON file exists, only materials from the current run
    # will overwrite existing entries in the file
    json_file = dune3['job_root'] + '/results.json'
    file_existed = initialise_json_results(json_file)
    materials = materials_to_run(file_existed, json_file)

    print("Running materials: ", materials)
    for material in materials:
        # Settings
        dune3['pseudos'] = rutgers_pseudos[material]
        qe_settings = EspressoSettings(**dune3)
        atoms = bulk_materials[material]
        tb_converged_k_grid = converged_k_grids[material]
        # Sensible approach looks like start at the grid provided by xTB and go up another 6 settings
        k_grids = k_grid_range(tb_converged_k_grid, 6)
        # Simplify life. Fix at one reasonably-high value
        ecuts = [120.]
        indices = collate_variables(k_grids, ecuts)
        convergence = Convergence(target_delta=target_delta)
        print(material)

        # Variables loop
        result = {}
        for i, j in indices:
            # Variable inputs
            k_grid = k_grids[i]
            ecut = ecuts[j]
            dir = "".join(str(k) + "_" for k in k_grid) + str(ecut)
            qe_settings.run_dir = os.path.join(dune3['job_root'], material, dir)

            # Makes the output messy
            # print(material, f'k_grid {k_grid}, ecut {ecut}', qe_settings.run_dir_root)

            # Calculation
            qe_input = set_espresso_input(material, ecutwfc=ecut, pseudo_dir=qe_settings.pseudo_dir)
            atoms, calculator = run_scf_calculation(qe_settings.run_dir, qe_input, atoms, k_grid,
                                                    qe_settings.pseudos)
            total_energy_in_ev = parse_espresso_total_energy(qe_settings.run_dir)

            # Convergence
            convergence.update(input={'k_grid': k_grid, 'ecut': ecut}, output=total_energy_in_ev)
            convergence.print_iteration()

            # Store result
            store_result(json_file, qe_settings, dir, convergence)
            if convergence.has_converged():
                break

        # White space between systems
        print()
        #   convergence.print_summary()
