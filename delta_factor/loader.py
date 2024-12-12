"""Data loaders (parsers) for energy vs volume data
"""
import bz2
from pathlib import Path
import pickle
import numpy as np
from typing import List, Dict, Callable
from delta_factor.fit_eos import EVCurveData


def numpy_loader(root: Path, materials: List[str], file_name: Callable) -> Dict[str, np.ndarray]:
    """ Load numpy E vs V Data

    Usage:
    ```
     results = <>_loader(RESULTS_ROOT)
     volume = results[material][:, 0]
     energy = results[material][:, 1]

    :return Dict of EvsV np array
    """
    results = {}
    for material in materials:
        f_name = root / file_name(material)
        results[material] = np.loadtxt(f_name, skiprows=2)
    return results


def tblite_loader(root: Path, materials: List[str]) -> Dict[str, np.ndarray]:
    """ Load TBLite E vs V Data

    :return Dict of EvsV np array
    """
    return numpy_loader(root, materials, lambda m: Path('tblite1/e_vs_v/collated_results', f'e_vs_v_{m}.dat'))


def qe_loader(root: Path, materials: List[str]) -> Dict[str, np.ndarray]:
    """ Load QE E vs V Data

    :return Dict of EvsV np array
    """
    return numpy_loader(root, materials, lambda m: Path('espresso/e_vs_v/collated_results', f'e_vs_v_{m}.dat'))


def qcore_loader(root: Path) -> Dict[str, EVCurveData]:
    with bz2.open(root / 'qcore/qcore_e_vs_v.pickle', "rb") as fid:
        qcore_results = pickle.load(fid)

    # Modify keys to be consistent
    # At some point, should just redump the data and avoid this
    keys_to_rename = {'silicon': 'si', 'zinc_oxide': 'zno', 'germanium': 'ge'}
    for old, new in keys_to_rename.items():
        qcore_results[new] = qcore_results[old]
        del qcore_results[old]

    return qcore_results
