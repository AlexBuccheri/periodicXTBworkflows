""" Compute delta factors from the set of MP systems
- Adapt Rui's notebook and write back to file
"""
import json


def repackage_qcore_data(dict_file) -> dict:
    """
    Input file has top-level structure:
    dict_keys(['n_points_for_systems', 'shell_charges', 'atomic_charges', 'atomic_partial_charges', 'energies', 'identities'])
    """

    with open(dict_file, "r") as f:
        string = f.read()

    database = json.loads(string.replace("'", "\""))

    n_systems = 2455
    # Each system has 21 calculations (assume 11 is the equilibrium volume)
    n_points_for_systems = database['n_points_for_systems']
    shell_charges = database['shell_charges']
    atomic_charges = database['atomic_charges']
    atomic_partial_charges = database['atomic_partial_charges']
    energies = database['energies']
    identities = database['identities']

    repacked_data = {}
    for i, formula_id in enumerate(identities):
        formula, id = formula_id.split('@')
        repacked_data[id] = {'formula': formula,
                             'n_calculations': n_points_for_systems[i],
                             'shell_charges': shell_charges[i],
                             'atomic_charges': atomic_charges[i],
                             'atomic_partial_charges': atomic_partial_charges[i],
                             'energies': energies[i]
                             }
    return repacked_data


