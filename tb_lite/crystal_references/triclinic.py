"""
 Module containing triclinic crystal dictionaries:
  key : str
    crystal name
  value : str
    file path to cif
"""
import os

from tb_lite.crystal_references import cif_root

# Cubic crystals by bravais lattice
# Space groups: 1 and 2
root = os.path.join(cif_root, 'triclinic') + '/'

triclinic_cifs = {}