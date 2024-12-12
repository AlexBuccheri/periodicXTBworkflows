""" Converge TB Lite charges via total energy w.r.t. k-grids
"""
import json
from pathlib import Path
from typing import Dict
from ase.atoms import Atoms

from tb_lite.crystal_references.crystal_systems import bulk_materials

from tb_lite.src.band_utils import monkhorst_pack_k_grid_sampling
from tb_lite.src.classes.convergence import Convergence
from tb_lite.src.calculators.dftb_calculator import DFTBIOCalculator
from tb_lite.src.dftb_input import DftbInput, Hamiltonian
from tb_lite.src.parsers.parsers import clear_directory
from tb_lite.src.utils import get_project_root, Value, Units


def converge_scc(calculator: DFTBIOCalculator,
                 atoms: Atoms,
                 converger: Convergence,
                 k_grids) -> Convergence:
    """ Converge the density/charges for a material w.r.t. k-points.

    Note, no mechanism has been included for handling:
        a) Job crashes
        b) Job restarts from some arbitrary k input

    This is because the calculations are extremely fast, and so do not warrant
    additional code development.

    :return: Instance of Convergence per calculation.
    """
    for i, k_grid in enumerate(k_grids):

        clear_directory(calculator.directory)

        # Set input variable. Should inject behaviour
        calculator.input.hamiltonian.set_k_grid(k_grid)

        # Write input and structure to file
        calculator.write_input(atoms)

        process_result = calculator.run()

        if process_result.success:
            result = calculator.parse_result()
            converger.update(k_grid.tolist(), result['total_energy'])
            converger.print_iteration()
        else:
            # No means of recovering a calculation - cheap, so kill
            print('Calculation failed')
            print(process_result.stdout)
            print(process_result.stderr)
            quit()

        if converger.has_converged():
            # Re-run the prior calculation so charges are available
            clear_directory(calculator.directory)
            calculator.input.hamiltonian.set_k_grid(k_grids[i - 1])
            calculator.write_input(atoms)
            process_result = calculator.run()
            return converger

    return converger


def set_directories(root, materials: dict) -> Dict[str, Path]:
    directories = {}
    for name in materials.keys():
        directories[name] = (root / Path(name))
    return directories


if __name__ == "__main__":

    # 1. Env Settings
    # -------------------------
    # When running in docker mounted in pycharm
    root = Path("/tb_benchmarking/tblite1_scc_results")
    save_results = True

    # 2. Calculation Settings
    # -------------------------
    # Tolerance for convergence in total energy w.r.t. change in the input variable i.e. k-sampling
    # Note, this is in eV, as energies are parsed in eV
    total_energy_tol = Value(1.e-4, Units.eV)
    # TB Lite 1 calculator
    dummy_grid = [1, 1, 1]
    # Note, input units are atomic
    dftb_settings = {'method': 'GFN1-xTB', 'temperature': 300.0, 'scc_tolerance': 1.e-6, 'k_grid': dummy_grid}
    input = DftbInput(hamiltonian=Hamiltonian(**dftb_settings))
    # Max integer multiple number of grids to use in convergence
    n_grids = 20

    # 3. Calculation Execution
    # -------------------------
    materials = bulk_materials
    directories = set_directories(root, materials)
    results = {name:{} for name in materials}

    for material, atoms in materials.items():
        print(material)
        directory = directories[material]
        Path.mkdir(directory, parents=True, exist_ok=True)
        # Input settings
        # NOTE, would need to pass run settings too
        calculator = DFTBIOCalculator(directory, input)
        converger = Convergence(target_delta=total_energy_tol.value)
        # Input variable, noting that convergence is observed faster
        # if one only compares grids with even-only (or odd-only) sampling
        k_grids = monkhorst_pack_k_grid_sampling(n_grids, atoms, even_grids=True)
        # Converge SCC
        converger = converge_scc(calculator, atoms, converger, k_grids)
        # Store results
        results[material]['directory'] = directory.as_posix()
        results[material]['energy_vs_k'] = converger.serialise('k_sampling', 'total_energy')

    # 4. Results
    # -------------------------
    print(results)

    # Append input settings to results
    dftb_settings['units'] = 'a.u.'
    results.update({'input_settings': dftb_settings,
                    'total_energy_tol': total_energy_tol.to_dict()
                    })

    if save_results:
        project_root = get_project_root()
        output_path = Path(project_root, 'outputs/tblite1/scc')
        output_path.mkdir(parents=True, exist_ok=True)
        file_name = Path(output_path, 'results.json')

        print(f"Dump all results to {file_name}\n")
        with open(file_name, 'w', encoding='utf-8') as fid:
            json.dump(results, fid, ensure_ascii=False, indent=4)
