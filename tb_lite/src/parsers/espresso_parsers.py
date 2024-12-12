""" Quantum Espresso Parsers
"""
import re
import os

def parse_espresso_total_energy(run_dir) -> float:
    """ Parse QE total energy in eV.

    :param run_dir:
    :return:
    """
    file = os.path.join(run_dir, 'espresso.pwo')
    try:
        with open(file) as fid:
            string = fid.read()
    except FileNotFoundError:
        raise FileNotFoundError(f'{file} not found')

    ry_to_ev = 13.6056980659
    total_energy_match = re.findall(r'total energy\s*=.*$', string, flags=re.MULTILINE)[0]
    energy_in_ry = float(total_energy_match.split()[-2])

    return energy_in_ry * ry_to_ev
