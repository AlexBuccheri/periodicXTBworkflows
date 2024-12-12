""" Generate a Quantum Espresso PX Input
"""
from tb_lite.src.utils import replace_item


def set_espresso_input(material: str, **kwargs) -> dict:
    """ Basic Espresso input options for an SCF calculation.

    https://www.quantum-espresso.org/Doc/INPUT_PW.html

    Assume one does not need to specify:
        SYSTEM
        ATOMIC_SPECIES
        ATOMIC_POSITIONS
        K_POINTS
    as passed separately.

    NOTE Still a problem for degenerate keys.

    :param material: Material name.
    :return: Input dictionary for pw.x.
    """
    # Initial settings, with some choice of defaults
    espresso_settings = {'control': {'prefix': f'{material}_PBESOL',
                                     'calculation': 'scf',
                                     'pseudo_dir': '',
                                     'verbosity': 'high'},
                         'electrons': {'conv_thr': 1.e-6},
                         'system': {'ecutwfc': 0}
                         }

    for key, value in kwargs.items():
        for top_level_key in ['control', 'electrons', 'system']:
            espresso_settings[top_level_key] = replace_item(espresso_settings[top_level_key], key, value)

    return espresso_settings
