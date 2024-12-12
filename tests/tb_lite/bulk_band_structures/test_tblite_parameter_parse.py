import pytest
import toml
from typing import List
import numpy as np

from tb_lite.bulk_band_structures.tblite_parameter_parse import flatten_parameters, ParamID, parameter_entry_lens, \
    put_elemental_x, get_elemental_x


def test_flatten_parameters_with_one_element():
    parameters = {'Si': ['levels',
                        'slater',
                        'ngauss',
                        'refocc',
                        'shpoly',
                        'kcn',
                        'gam',
                        'lgam',
                        'gam3',
                        'zeff',
                        'arep',
                        'xbond',
                        'en',
                        'dkernel',
                        'qkernel',
                        'mprad',
                        'mpvcn']
                  }

    param_ids: List[ParamID] = flatten_parameters(parameters)

    for pid in param_ids:
        assert pid.symbol == 'Si', 'All parameters should be defined for Si'

    n_elements = 1
    # Number of parameter entries per element (note, shell is considered fixed, hence not included here)
    n_elem_params = 17

    assert len(param_ids) == n_elements * n_elem_params

    assert param_ids[0].key == 'levels'
    assert param_ids[0].i == 0
    assert param_ids[0].j == 3

    assert param_ids[1].key == 'slater'
    assert param_ids[1].i == 3
    assert param_ids[1].j == 6

    assert param_ids[-1].key == 'mpvcn'
    assert param_ids[-1].i == 30
    assert param_ids[-1].j == 31, 'Total number of parameters per element'

    assert sum([i for i in parameter_entry_lens.values()]) == 31, \
        'Total number of parameters per element (excluding shell)'


def test_flatten_parameters_with_two_elements():
    parameters = {'Si': ['levels',
                         'slater',
                         'ngauss',
                         'refocc',
                         'shpoly',
                         'kcn',
                         'gam',
                         'lgam',
                         'gam3',
                         'zeff',
                         'arep',
                         'xbond',
                         'en',
                         'dkernel',
                         'qkernel',
                         'mprad',
                         'mpvcn'],
                  'P': ['levels',
                        'slater',
                        'ngauss',
                        'refocc',
                        'shpoly',
                        'kcn',
                        'gam',
                        'lgam',
                        'gam3',
                        'zeff',
                        'arep',
                        'xbond',
                        'en',
                        'dkernel',
                        'qkernel',
                        'mprad',
                        'mpvcn']
                  }

    param_ids: List[ParamID] = flatten_parameters(parameters)

    n_elements = len({x.symbol for x in param_ids})
    assert n_elements == 2

    total_entries = len(param_ids)
    assert total_entries == 34, 'Number of keys per element, summed over all elements'

    # First Si entry
    assert param_ids[0].symbol == 'Si'
    assert param_ids[0].key == 'levels'
    assert param_ids[0].i == 0
    assert param_ids[0].j == 3

    # Last Si entry
    assert param_ids[16].symbol == 'Si'
    assert param_ids[16].key == 'mpvcn'
    assert param_ids[16].i == 30
    assert param_ids[16].j == 31

    # First P entry
    assert param_ids[17].symbol == 'P'
    assert param_ids[17].key == 'levels'
    assert param_ids[17].i == 31
    assert param_ids[17].j == 34

    # Last P entry
    assert param_ids[-1].symbol == 'P'
    assert param_ids[-1].key == 'mpvcn'
    assert param_ids[-1].i == 61
    assert param_ids[-1].j == 62, 'Total number of parameters for SiP'


@pytest.fixture()
def gfn_xtb_elemental_params() -> dict:
    """ Elemental
    """
    toml_string = """[element.P]
shells = [ "3s", "3p", "3d" ]
levels = [ -1.8865587000000001E+01, -9.3864640000000001E+00, -6.7398899999999995E-01 ]
slater = [ 1.9931650000000001E+00, 1.8269730000000000E+00, 1.2933450000000000E+00 ]
ngauss = [ 6, 6, 4 ]
refocc = [ 2.0000000000000000E+00, 3.0000000000000000E+00, 0.0000000000000000E+00 ]
shpoly = [ -1.6118985000000000E-01, -2.2411890000000000E-02, 3.0984577000000002E-01 ]
kcn = [ 1.1319352200000002E-01, -2.8159391999999995E-02, -3.3699449999999996E-03 ]
gam = 7.9831900000000000E-01
lgam = [ 1.0000000000000000E+00, 7.4691249999999998E-01, 1.0000000000000000E+00 ]
gam3 = 1.5000000000000002E-01
zeff = 1.5249559000000000E+01
arep = 1.0671970000000000E+00
xbond = 0.0000000000000000E+00
en = 2.1899999999999999E+00
dkernel = 0.0000000000000000E+00
qkernel = 0.0000000000000000E+00
mprad = 0.0000000000000000E+00
mpvcn = 0.0000000000000000E+00
[element.S]
shells = [ "3s", "3p", "3d" ]
levels = [ -2.3819013000000002E+01, -1.2120136000000000E+01, -1.7112609999999999E+00 ]
slater = [ 2.5069340000000002E+00, 1.9927750000000000E+00, 1.9648669999999999E+00 ]
ngauss = [ 6, 6, 4 ]
refocc = [ 2.0000000000000000E+00, 4.0000000000000000E+00, 0.0000000000000000E+00 ]
shpoly = [ -1.6989921999999999E-01, -6.0677790000000002E-02, 1.6248394999999999E-01 ]
kcn = [ 1.4291407800000000E-01, -3.6360408000000004E-02, -8.5563050000000002E-03 ]
gam = 6.4395899999999995E-01
lgam = [ 1.0000000000000000E+00, 8.3218530000000002E-01, 1.0000000000000000E+00 ]
gam3 = 1.5000000000000002E-01
zeff = 1.5100322999999999E+01
arep = 1.2008030000000001E+00
xbond = 0.0000000000000000E+00
en = 2.5800000000000001E+00
dkernel = 0.0000000000000000E+00
qkernel = 0.0000000000000000E+00
mprad = 0.0000000000000000E+00
mpvcn = 0.0000000000000000E+00
    """
    return toml.loads(toml_string)


def test_get_elemental_x(gfn_xtb_elemental_params):
    parameters = {'S': ['levels',
                        'slater',
                        'mpvcn'],
                  'P': ['en']
                  }

    x = get_elemental_x(parameters, gfn_xtb_elemental_params)
    assert len(x) == 3 + 3 + 1 + 1

    # See values in `gfn_xtb_elemental_params` for reference
    x_ref = np.array([-2.3819013000000002E+01, -1.2120136000000000E+01, -1.7112609999999999E+00,  # Si levels
                       2.5069340000000002E+00, 1.9927750000000000E+00, 1.9648669999999999E+00,     # Si slater
                       0.0,  # Si mpvcn
                       2.1899999999999999E+00])  # P en
    assert np.allclose(x, x_ref)


def test_put_elemental_x(gfn_xtb_elemental_params):
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])

    parameters = {'S': ['levels',
                        'slater',
                        'mpvcn'],
                  'P': ['en']
                  }

    # Update parameters in gfn_xtb_elemental_params, using values from x
    new_gfn_xtb_elemental_params = put_elemental_x(x, parameters, gfn_xtb_elemental_params)

    assert new_gfn_xtb_elemental_params['element']['S']['levels'] == [1, 2, 3]
    assert new_gfn_xtb_elemental_params['element']['S']['slater'] == [4, 5, 6]
    assert new_gfn_xtb_elemental_params['element']['S']['mpvcn'] == 7
    assert new_gfn_xtb_elemental_params['element']['P']['en'] == 8
