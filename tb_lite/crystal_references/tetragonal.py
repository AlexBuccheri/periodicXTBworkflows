"""Module containing tetragonal crystal dictionaries.

Rutile and Anatase details are tabulated.
"""
import os
import ase
import numpy as np

from tb_lite.crystal_references import cif_root
from tb_lite.src.utils import FileUrl


# Cubic crystals by bravais lattice
# Space groups: 75 - 142.

root = os.path.join(cif_root, 'tetragonal') + '/'


# Any space group beginning with P
simple_tetragonal_cifs = {}

# Any space group beginning with I
body_centred_tetragonal_cifs = {}


# Crystals without cif files
# Avoid additional tabulating and use cif_parser_wrapper to generate
# these dictionaries.

def tio2_rutile() -> ase.atoms.Atoms:
    """TiO2 Rutile.

    Space group P42/mnm (136)
    Direct band gap: 1.781 eV
    Ref: https://materialsproject.org/materials/mp-2657/
    """
    fractional_positions = [[0.000000, 0.000000, 0.000000],
                            [0.500000, 0.500000, 0.500000],
                            [0.695526, 0.695526, 0.000000],
                            [0.304474, 0.304474, 0.000000],
                            [0.195526, 0.804474, 0.500000],
                            [0.804474, 0.195526, 0.500000]]
    # Angstrom
    a = 4.6068
    c = 2.9916

    atoms = ase.atoms.Atoms(symbols=['Ti', 'Ti', 'O', 'O', 'O', 'O'],
                            scaled_positions=fractional_positions,
                            cell=np.array([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, c]]),
                            pbc=True)

    return atoms


def tio2_anatase() -> ase.atoms.Atoms:
    """TiO2 Anatase.

    Space group:  I4_1/amd (141)
    Indirect band gap: 2.062 eV
    Ref: https://materialsproject.org/materials/mp-390/
    TODO(Alex) Looks like I've run it through SPGLib - maybe should check this structure
    """
    fractional_positions = [[0.500000, 0.500000, 0.000000],
                            [0.250000, 0.750000, 0.500000],
                            [0.456413, 0.956413, 0.500000],
                            [0.706413, 0.706413, 0.000000],
                            [0.043587, 0.543587, 0.500000],
                            [0.293587, 0.293587, 0.000000]]

    a = 5.55734663
    c = 5.55734663

    atoms = ase.atoms.Atoms(symbols=['Ti', 'Ti', 'O', 'O', 'O', 'O'],
                            scaled_positions=fractional_positions,
                            cell= 0.5 * a * np.array([[-1.0,  1.0,  c/a],
                                                       [1.0, -1.0,  c/a],
                                                       [1.0,  1.0, -c/a]]),
                            pbc=True)

    return atoms
