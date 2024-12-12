""" TB lite parameter fitting.

Run from the project root:
```
python3 tb_lite/bulk_band_structures/parameter_fit.py
```

To generate the default parameter file in the corresponding Docker container:

```bash
cd /dftbplus/build/external/tblite/origin/app
tblite param --method gfn1 --output gfn1-xtb.toml
```
NOTE: In principle, one should converge the SCF again, however we will use the converged SCC settings:
"""
import copy
import numpy as np
from pathlib import Path
import json
from typing import List, Dict
from scipy.optimize import minimize
from ase.atoms import Atoms

from band_structure_comparisons.parse import parse_band_structures, serialise_band_structure

from tb_lite.src.classes.runner import SubprocessRunResults
from tb_lite.crystal_references import cubic
from tb_lite.crystal_references.crystal_systems import bulk_materials
from tb_lite.src.calculators.dftb_calculator import DFTBIOCalculator
from tb_lite.src.dataclasses.band_structure import BandData
from tb_lite.src.dftb_input import DftbInput, Hamiltonian

from tb_lite.bulk_band_structures.band_structure_calculator import TBLiteBandStructure
from tb_lite.bulk_band_structures.loss_functions import valence_loss_function
from tb_lite.bulk_band_structures.tblite_parameter_parse import get_elemental_x, put_elemental_x, \
    write_gfn1_parameters, parse_gfn1_parameters


def get_converged_k_sampling(material_data: dict) -> List[float]:
    """

    """
    second_last_result = material_data['energy_vs_k'][-2]
    last_result = material_data['energy_vs_k'][-1]

    # TB lite converge runs one denser grid to confirm convergence
    if np.isclose(second_last_result['total_energy'], last_result['total_energy']):
        return second_last_result['k_sampling']

    # If they differ, take the last grid as "most-converged"
    return last_result['k_sampling']


def report_process_error(process: SubprocessRunResults, msg):
    print('stdout:')
    print(process.stdout.decode("utf-8"))
    print('stderr:')
    print(process.stderr.decode("utf-8"))
    raise RuntimeError(msg)


def extended_minimise(material_root: Path,
                      original_gfn1_params: dict,
                      parameters_to_optimise: Dict[str, List[str]],
                      dftb_settings: dict,
                      atoms: Atoms,
                      max_itr: int):
    """ Wrapper about minimise.

    For provenance, one should consider adding a callback function that
    sets up new SCC & bands directories per minimisation iteration.

    https://docs.scipy.org/doc/scipy/tutorial/optimize.html#custom-minimizers
    """
    x = get_elemental_x(parameters_to_optimise, original_gfn1_params)
    print('Parameter before optimisation', x)

    # Directory generation
    scc_run_dir = material_root / f'scc'
    bands_run_dir = material_root / f'bands'
    scc_run_dir.mkdir(parents=True, exist_ok=True)
    bands_run_dir.mkdir(parents=True, exist_ok=True)
    directories = {'material_root': material_root, 'scc': scc_run_dir, 'bands': bands_run_dir}

    result = minimize(evaluate_tblite_bandstructure,
                      x,
                      method='nelder-mead',
                      args=(directories, original_gfn1_params, parameters_to_optimise, dftb_settings, atoms),
                      options={'maxiter': max_itr, 'disp': True})
    print('result.x', result.x)

    print('Parameter after optimisation', result.x)
    return result


# NOTE: If this API changes, one must update the args list of `minimize` call
def evaluate_tblite_bandstructure(x: np.ndarray,
                                  directories: dict,
                                  original_gfn1_params: dict,
                                  parameters_to_optimise: Dict[str, List[str]],
                                  dftb_settings: dict,
                                  atoms: Atoms) -> float:
    """ Evaluate the agreement of TB lite 1 band structure with Quantum Espresso

    Involves:
     * Running an SCC calculation
     * Running a band structure calculation with TB Lite
     * Evaluate a loss function, taking the dif between TB lite and the reference QE band structure

    :param x: GFN xTB1 parameter/s being optimised
    :param directories: Work directories
    :param original_gfn1_params: Original TB Lite parameters (all)
    :param parameters_to_optimise: GFN parameters to optimise. Of the form {'Si': ['gam', 'gam3', ...]}
    :param dftb_settings Input settings for DFTB+-TB Lite. Should not change for a given material.
    :param atoms: ASE atoms. Should not change for a given material
    :return valence_loss_function: Evaluated fit function.
    """
    # Directories
    material_root = directories['material_root']
    scc_run_dir = directories['scc']
    bands_run_dir = directories['bands']

    # Update TB parameters based in input x, and write to file
    gfn1_params = copy.deepcopy(original_gfn1_params)
    gfn1_params = put_elemental_x(x, parameters_to_optimise, gfn1_params)
    write_gfn1_parameters(gfn1_params, scc_run_dir)

    # SCC calculation
    dftb_settings['ParameterFile'] = scc_run_dir / 'gfn1-xtb.toml'
    input = DftbInput(hamiltonian=Hamiltonian(**dftb_settings))
    scc_calculator = DFTBIOCalculator(scc_run_dir, input, atoms)
    scc_calculator.write_input()

    scc_process = scc_calculator.run(omp=4)
    if not scc_process.success:
        report_process_error(scc_process, f'DFTB SCC calculation failed. See {scc_run_dir}')

    # Band structure calculation
    bs_calculator = TBLiteBandStructure('GFN1-xTB',
                                        scc_run_dir, bands_run_dir,
                                        ParameterFile=dftb_settings['ParameterFile'],
                                        npoints=100)
    bs_calculator.write_input()
    bs_process = bs_calculator.run()

    if not bs_process.success:
        report_process_error(bs_process, f'DFTB band structure calculation failed. See {bands_run_dir}')

    band_data: BandData = bs_calculator.parse_result()

    # Ensure TB band structure is zeroed at Ef
    band_data.set_bands_zeropoint(band_data.fermi_level)

    # Dump to file to confirm raw data is valid
    band_data.dump_gnuplot(material_root / "tb_bands.dat")

    # Fit according to the valence bands only
    # One could try several schemes here
    i_vbt_tb = band_data.n_occupied_bands
    tb_valence_bands = band_data.bands[:, 0:i_vbt_tb]

    return valence_loss_function(tb_valence_bands, qe_valence_bands)


if __name__== "__main__":

    # Define directories
    # This works as long as `parameter_fit.py` is executed from the project root.
    # Note: No nice way to do this with python
    PROJECT_ROOT = Path.cwd().absolute()
    FIT_ROOT = PROJECT_ROOT / 'outputs/tblite1/parameter_fitting'
    RESULTS_ROOT = Path.cwd().absolute() / 'tb_results'

    # Default xTB parameters
    original_gfn1_params = parse_gfn1_parameters()

    # Load reference QE band structure data
    # QE bands are already zeroed at E_fermi
    qe_results = parse_band_structures(RESULTS_ROOT / 'espresso/band_structure/espresso_bandstructures.pickle')
    spin_unpolarised = 0

    # Reference TB lite converged SCC data
    with open(RESULTS_ROOT / 'tblite1/scc/results.json', 'r', encoding='utf-8') as fid:
        scc_data = json.load(fid)

    # Test with silicon
    # materials = bulk_materials
    materials = {'silicon': cubic.silicon()}
    parameters_to_optimise = {'silicon': {'Si': ['gam', 'lgam', 'gam3', 'en']},
                              'cdse': {'Cd': ['gam', 'lgam', 'gam3', 'en'], 'Se': ['gam', 'lgam', 'gam3', 'en']}}

    for material, atoms in materials.items():

        print('Parameters to optimise', parameters_to_optimise[material])

        # QE reference
        qe_bands = qe_results[material]['instance'].energies[spin_unpolarised, :, :]
        i_vbt_qe = qe_results[material]['n_occupied']
        qe_valence_bands = qe_bands[:, 0:i_vbt_qe]

        # Define & make run dir
        material_root = FIT_ROOT / material
        material_root.mkdir(parents=True, exist_ok=True)
        serialise_band_structure(qe_results[material]['instance'], material_root /"qe_bands.dat")

        # Converged SCC TB k-grid
        converged_k_grid = get_converged_k_sampling(scc_data[material])

        # Define TB lite SCC inputs
        dftb_settings = {'ParameterFile': 'SET IN EVALUATOR',
                         'temperature': 300.0,
                         'scc_tolerance': 1.e-6,
                         'k_grid': converged_k_grid
                         }

        # Minimisation routine
        result = extended_minimise(material_root,
                                   original_gfn1_params, parameters_to_optimise[material],
                                   dftb_settings, atoms,
                                   max_itr=10)

        # Evaluate TB band structure with optimal parameters
        scc_run_dir = material_root / f'scc_eval'
        bands_run_dir = material_root / f'bands_eval'

        scc_run_dir.mkdir(parents=True, exist_ok=True)
        bands_run_dir.mkdir(parents=True, exist_ok=True)

        directories = {'material_root': material_root, 'scc': scc_run_dir, 'bands': bands_run_dir}
        evaluate_tblite_bandstructure(result.x,
                                      directories,
                                      original_gfn1_params, parameters_to_optimise[material],
                                      dftb_settings, atoms)
