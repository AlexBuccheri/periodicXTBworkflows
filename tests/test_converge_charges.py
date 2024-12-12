from tb_lite.crystal_references import cubic
from tb_lite.src.parsers.parsers import cif_to_ase_atoms

# Routines under test
from tb_lite.src.band_utils import monkhorst_pack_k_grid_sampling

import numpy as np

def test_k_grid_sampling():
    atoms = cif_to_ase_atoms(cubic.fcc_cifs.get('zirconium_dioxide').file)
    #atoms = cubic.diamond()
    lattice = atoms.get_cell()
    lattice = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 4]])
    print(lattice)
    k_grids = monkhorst_pack_k_grid_sampling(4, lattice)
    print(k_grids)
