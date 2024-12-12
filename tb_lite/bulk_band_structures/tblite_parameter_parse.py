""" Parsing and manipulating the TB Lite TOML parameter file.
"""
import copy
import numpy as np
from pathlib import Path
import toml
from typing import List


def parse_gfn1_parameters() -> dict:
    """ Parse GFN1 parameters from TOML to dict.

    NOTE: Run from tb_benchmarking repo root for directory to work.
    """
    # Default GFN1 parameters
    with open('data/gfn1-xtb.toml', 'r') as fid:
        toml_string = fid.read()
    gfn1_params = toml.loads(toml_string)
    return gfn1_params


def write_gfn1_parameters(gfn1_params: dict, path: Path):
    """ Write GFN1 parameters to TOML.

    :param gfn1_params:
    :param path:
    """
    with open(path / 'gfn1-xtb.toml', 'w') as fid:
        toml.dump(gfn1_params, fid, encoder=None)
    return


# Number of values/entries per elemental parameter
parameter_entry_lens = {'levels': 3,
                        'slater': 3,
                        'ngauss': 3,
                        'refocc': 3,
                        'shpoly': 3,
                        'kcn': 3,
                        'gam': 1,
                        'lgam': 3,
                        'gam3': 1,
                        'zeff': 1,
                        'arep': 1,
                        'xbond': 1,
                        'en': 1,
                        'dkernel': 1,
                        'qkernel': 1,
                        'mprad': 1,
                        'mpvcn': 1}


class ParamID:
    """ Container for parameter information.
    """
    def __init__(self, symbol: str, key: str, i: int, j: int):
        self.symbol = symbol
        self.key = key
        self.i = i
        self.j = j


def flatten_parameters(parameters: dict) -> List[ParamID]:
    """ Define parameter order in a continuous numpy vector, and provide
    the (symbol, key, start, end indices) of each parameter key in this vector.

    For example, given a dict of element-specific parameters, such as:

      parameters = {'Si': ['levels', 'slater', 'ngauss', ...]}

    create a continuous index of the entries:

      [param_entry1, param_entry2, param_entry3, ...]

    where
      param_entry1: ParamID contains 'Si', 'levels', (i,j) = (0, 3)
      param_entry2: ParamID contains 'Si', 'slater', (i,j) = (3, 6)
      param_entry3: ParamID contains 'Si', 'ngauss', (i,j) = (6, 9)

    :param parameters: Element-specific parameters to be verified in the optimisation.
    :return param_ids: Symbol, key, start/end index info for each parameter, in a list.
    """
    i = 0
    j = 0
    param_ids = []
    for symbol, element_params in parameters.items():
        for key in element_params:
            j += parameter_entry_lens[key]
            param_ids.append(ParamID(symbol, key, i, j))
            i += parameter_entry_lens[key]
    return param_ids


def get_elemental_x(parameters: dict, gfn_params: dict) -> np.ndarray:
    """ Get a subset of element-specific parameters from a dict of all GFN parameters.

    For element-specific parameters:

      parameters = {'Si': ['levels', 'slater', 'ngauss', ...]}

     where 'levels', 'slater' and 'ngauss' all consist of three values, this routine will return:

      x = [ -1.8865587E+01, -9.386464E+00, -6.73989E-01, 1.993165E+00, 1.826973E+00, 1.293345E+00, 6, 6, 4, ...]

    :param parameters: Element parameters to obtain. This also defines the parameter order in x.
    :param gfn_params: All GFN parameters.
    :return x: Numpy vector of parameters extracted from the gfn_params dict
    """
    x = []
    for symbol, keys in parameters.items():
        for key in keys:
            parameter = gfn_params['element'][symbol][key]
            # Single values are annoyingly not in a list
            parameter = parameter if type(parameter) is list else [parameter]
            x += parameter
    return np.asarray(x)


def put_elemental_x(x: np.ndarray, parameters: dict, gfn_params: dict) -> dict:
    """ Update `parameters` in `gfn_params`, using optimised values from `x`.

    Given optimised parameters, x, in a numpy vector, insert them back into
    the GFN parameters dict (typically for writing to file).

    :param x: Optimised parameter values (only).
    :param parameters: Parameters labels and ordering in x.
    :param gfn_params: GFN parameter dict in which to insert optimised x values.
    """
    updated_gfn_params = copy.deepcopy(gfn_params)
    p_flattened = flatten_parameters(parameters)

    n_parameters = sum([param.j - param.i for param in p_flattened])
    if n_parameters != x.shape[0]:
        message = f"Number of parameters optimised by scipy (in x), {x.shape[0]} is inconsistent wth the total " \
                  f"number of parameters in `parameters` {n_parameters}"
        raise ValueError(message)

    for parameter in p_flattened:
        i, j = parameter.i, parameter.j
        # Float if one value, else list
        # TODO(Alex) Test if x[i:j].tolist() will ultimately end up as strings (SCIPY?)
        optimised_p = float(x[i]) if j - i == 1 else x[i:j].tolist()
        updated_gfn_params['element'][parameter.symbol][parameter.key] = optimised_p

    return updated_gfn_params
