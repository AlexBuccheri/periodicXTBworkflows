""" JSON loader for QCore results
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from delta_factor.fit_eos import EVCurveData


# To be consistent with Rui's repackaging when using JSON
def qcore_repack(data: dict, output: str) -> dict:

    # Output Format
    package_funcs = {'np': lambda v, e: np.column_stack((v, e)),
                     'EVCurveData': lambda v, e: EVCurveData(v, e)
                     }
    try:
        package_func = package_funcs[output]
    except KeyError:
        raise ValueError(f'output arg was {output}. Can only be (EVCurveData, np)')

    repackaged_data = {}
    for material in data.keys():
        v = data[material]['volume']
        e = data[material]['energy']
        repackaged_data[material] = package_func(v, e)

    # Map inconsistent key names
    keys_to_rename = {'silicon': 'si', 'zinc_oxide': 'zno', 'germanium': 'ge'}
    for old, new in keys_to_rename.items():
        repackaged_data[new] = copy.deepcopy(repackaged_data[old])
        del repackaged_data[old]

    return repackaged_data


def load_json_qcore(RESULTS_ROOT,
                    file,
                    cutoffs,
                    output='EVCurveData'
                    ) -> dict | EVCurveData:
    """ Load QCore JSON data, for specified cutoff, and repackage in
    a sensible format.
    """
    # QCore results for all cutoffs
    with open(Path(RESULTS_ROOT, file), 'r') as fid:
        qcore_raw_data = json.load(fid)

    # Select data for a given cut-off, and repack
    qcore_results = {}
    for m, cutoff in cutoffs.items():
        qcore_results[m] = copy.deepcopy(qcore_raw_data[cutoff][m])

    qcore_results = qcore_repack(qcore_results, output)
    return qcore_results