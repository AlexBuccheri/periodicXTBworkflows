"""
 Module containing cubic crystal dictionaries with the signature:
  key : str
    crystal name
  value : str
    file path to cif

  boron nitride is tabulated.
"""
import os
import ase
import numpy as np
from ase.atoms import Atoms

from tb_lite.crystal_references import cif_root
from tb_lite.src.utils import FileUrl


# Hexagonal crystals by bravais lattice
# Space groups: 168 - 194
# All space groups in this system begin with P

root = os.path.join(cif_root, 'hexagonal') + '/'


hexagonal_cifs = {'zinc_oxide': FileUrl(root + 'ZnO-WZ/ZnO_mp-2133_primitive.cif', "https://materialsproject.org/materials/mp-2133/"),
                  'molybdenum_disulfide': FileUrl(root + 'MoS2/MoS2_mp-2815_primitive.cif', 'https://materialsproject.org/materials/mp-2815/'),
                  'boron_nitride': {'wurzite': FileUrl(root + 'BN-WZ/BN_mp-2653_primitive.cif', "https://materialsproject.org/materials/mp-2653/")},  #TODO CHECK ME
                  'unbuckled_graphite': FileUrl(root + 'Unbluckled_graphite/A_hP4_194_bc.cif', 'UNKNOWN'),
                  'tungsten_disulfide': FileUrl(root + 'WS2/WS2_mp-224_primitive.cif', 'https://materialsproject.org/materials/mp-224/'),
                  'cadmium_selenide': FileUrl(root + 'CdSe-hexagonal/CdSe_mp-1070_primitive.cif', 'https://materialsproject.org/materials/mp-1070/'),
                  'graphite': FileUrl(root + 'graphite/C_mp-48_primitive.cif', "https://materialsproject.org/materials/mp-48/")
                  }


# Crystals without cif files
# Avoid additional tabulating and use cif_parser_wrapper to generate
# these dictionaries.

def boron_nitride_hexagonal() -> ase.atoms.Atoms:
    """Hexagonal boron nitride.

    Space group: P63/mmc [194] 'hexagonal'
    Primitive lattice vectors and atomic basis
    Indirect and gap: 4.482 eV
    https://materialsproject.org/materials/mp-984/
    """
    fractional_positions = [[1/3, 2/3, 1/4],
                            [2/3, 1/4, 3/4],
                            [1/3, 2/3, 3/4],
                            [2/3, 1/3, 1/4]]

    # Angstrom
    a = 2.51242804
    c = 7.70726501
    cell = np.array([[a * 0.5, - a * 0.5 * np.sqrt(3), 0.0],
                     [a * 0.5,   a * 0.5 * np.sqrt(3), 0.0],
                     [0.0, 0.0, c]])

    atoms = Atoms(symbols=['B', 'B', 'N', 'N'],
                  scaled_positions=fractional_positions,
                  cell=cell,
                  pbc=True)

    return atoms
