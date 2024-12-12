import pathlib

import numpy as np

from tb_lite.molecular_crystals.run_molecular_evsv import parse_x23_xyz_to_ase


def test_parse_x23_xyz_to_ase(tmp_path):
    data = """8
100 0 0 
0  100 0
0  0 100
H	11.4581	0.0862	2.8742	0.592867
H	12.7157	0.1799	1.5982	0.583613
H	12.4376	-1.3554	2.4309	0.586326
H	10.7201	-1.7525	-0.6058	0.622374
C	11.94	-0.4521	2.0533	0.749122
C	10.9077	-0.8192	1.0143	0.813875
O	9.7364	-0.5044	1.0232	0.960555
O	11.4509	-1.5732	0.0124	0.880852

    """
    file = tmp_path / 'example.xyz'
    file.write_text(data)
    atoms = parse_x23_xyz_to_ase(file)

    expected_positions = np.array([[1.14581e+01,  8.62000e-02,  2.87420e+00],
                                   [1.27157e+01,  1.79900e-01,  1.59820e+00],
                                   [1.24376e+01, -1.35540e+00,  2.43090e+00],
                                   [1.07201e+01, -1.75250e+00, -6.05800e-01],
                                   [1.19400e+01, -4.52100e-01,  2.05330e+00],
                                   [1.09077e+01, -8.19200e-01,  1.01430e+00],
                                   [9.73640e+00, -5.04400e-01,  1.02320e+00],
                                   [1.14509e+01, -1.57320e+00,  1.24000e-02]])

    expected_lattice = np.array([[100.,   0.,   0.],
                                 [  0., 100.,   0.],
                                 [  0.,   0., 100.]])

    assert atoms.get_global_number_of_atoms() == 8
    assert atoms.get_chemical_symbols() == ['H', 'H', 'H', 'H', 'C', 'C', 'O', 'O']
    assert np.allclose(atoms.get_positions(), expected_positions)
    assert np.allclose(atoms.get_cell().array, expected_lattice)
