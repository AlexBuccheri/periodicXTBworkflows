"""Module containing trigonal crystal dictionaries:
  key : str
    crystal name
  value : str
    file path to cif
"""
import os
import numpy as np
from ase.atoms import Atoms

from tb_lite.crystal_references import cif_root
from tb_lite.src.utils import FileUrl

# Cubic crystals by bravais lattice
# Space groups: 143 - 167.

root = os.path.join(cif_root, 'trigonal') + '/'


# Any space group beginning with P
hexagonal_cifs = {}

# Any space group beginning with R
rhomohedral_cifs = {}


# Crystals without cif files
# Avoid additional tabulating and use cif_parser_wrapper to generate
# these dictionaries.

# TODO(Alex)
# Looks like I ran it through SPGLib or something, as the positions don't
# match the Materials Project reference. Need to get the lattice vectors
# in a consistent way.
#
# def MoS2() -> Atoms:
#     """Molybdenum Disulfide
#
#     Rhombohedral setting. Space group: 160
#     Indirect band gap: 1.228 eV
#     https://materialsproject.org/materials/mp-1434/
#     """
#     positions = [[0.000000, 0.000000, 0.000000],
#                  [0.254176, 0.254176, 0.254176],
#                  [0.412458, 0.412458, 0.412458]]
#
#     # Angstrom
#     a = 6.856
#     # Degrees
#     alpha = 26.94
#
#     cell = np.array([[], [], []])
#
#     atoms = Atoms(symbols=['Mo', 'S', 'S'],
#                   scaled_positions=positions,
#                   cell=cell,
#                   pbc=True)
#
#     return atoms
