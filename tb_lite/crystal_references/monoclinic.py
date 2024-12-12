"""
 Module containing cubic crystal dictionaries with the signature:
  key : str
    crystal name
  value : str
    file path to cif

"""
import os

from tb_lite.crystal_references import cif_root
from tb_lite.src.utils import FileUrl

# Monoclinic crystals by bravais lattice
# Space groups: 3-15.

root = os.path.join(cif_root, 'monoclinic') + '/'

# Any space group beginning with P
simple_monoclinic_cifs = {"tungsten_oxide": FileUrl(root + 'Simple/WO3/WO3_mp-19033_primitive.cif',
                                                    "https://materialsproject.org/materials/mp-19033/")}

# Any space group beginning with C:
# Need in C-configuration to be consistent with qCore lattice vectors
# and tabulated k-points
base_centred_c_monoclinic_cifs = {}


