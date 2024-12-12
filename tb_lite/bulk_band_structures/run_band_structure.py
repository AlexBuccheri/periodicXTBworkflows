""" Execution and Post-processing of TB-Lite Band Structures
"""
import json
import os.path
from pathlib import Path
import shutil
import pickle
import bz2
from typing import Dict

from tb_lite.bulk_band_structures.band_structure_calculator import TBLiteBandStructure


def define_directories(root):
    """ Read SCC directories and define new working directories.

    :param root:
    :return:
    """
    # Parse SCC directories, noting converged result is present
    with open('outputs/tb_lite_xtb1/scc_results/data.json', 'r', encoding='utf-8') as fid:
        scc_data: dict = json.load(fid)

    # Define working directories
    scc_dirs = []
    directories = []
    for name, result in scc_data.items():
        scc_dirs.append(result['directory'])
        directories.append(os.path.join(root, name))
    return scc_dirs, directories


def setup_calculators(scc_dirs, directories, npoints: int) -> Dict[str, TBLiteBandStructure]:
    """ Set up Band Structure Calculators

    :param scc_dirs:
    :param directories:
    :return:
    """
    calculators = {}
    for scc_dir, directory in zip(scc_dirs, directories):
        name = os.path.basename(directory)
        print(name)
        calculators[name] = TBLiteBandStructure('GFN1-xTB', scc_dir, directory, npoints=npoints)
    print('Calculators set up')
    return calculators


def initialise_inputs(calculators: Dict[str, TBLiteBandStructure]):
    """ Set up input files for band structure jobs.

    :param calculators:
    :return:
    """
    for name, calculator in calculators.items():
        path = Path(calculator.directory)
        if path.exists():
            shutil.rmtree(path.as_posix(), ignore_errors=False, onerror=None)
        path.mkdir(parents=True, exist_ok=True)
        calculator.write_input()
    print('Job directories and inputs set up')


def run(calculators: Dict[str, TBLiteBandStructure]) -> dict:
    """ Run calculators.

    :param calculators: Band structure calculators
    :return: Dict of band structure results
    """
    # Run
    band_structure_results = {}
    for name, calculator in calculators.items():
        print(f'Running band structure for {name}')
        process_result = calculator.run()
        band_structure_results[name] = {'process': process_result, 'calculator': calculator.parse_result()}
    return band_structure_results


if __name__ == "__main__":

    root = 'outputs/tb_lite_xtb1/band_structures'
    scc_dirs, directories = define_directories(root)

    calculators = setup_calculators(scc_dirs, directories, npoints=100)
    initialise_inputs(calculators)

    print('GaN removed due to an issue with DFTB+ converting band file')
    print('wo3_monoclinic due to inconsistent xticks vs labels')
    calculators.pop('gan')
    calculators.pop('wo3_monoclinic')

    results = run(calculators)

    # Serialise result
    with bz2.BZ2File('tblite_bandstructures.pickle', 'wb') as fid:
        pickle.dump(results, fid, protocol=pickle.HIGHEST_PROTOCOL)
