from tb_lite.src.parsers.dftb_parsers import parse_dftb_output
from tb_lite.src.parsers.qcore_parsers import parse_qcore_structure, parse_qcore_settings

def test_parse_qcore_input():
    # Assumes pytest run from the root
    structure: dict = parse_qcore_structure('molecular_crystals/entos/acetic.cif.in')
    # TODO(Alex) Add an assert
    # print(structure)


def test_parse_qcore_settings():
    data = parse_qcore_settings('molecular_crystals/entos/acetic.cif.in')
    assert set(data.keys()) == {'k_points', 'electronic_temperature'}
    assert data['k_points'] == [4, 4, 4]
    assert data['electronic_temperature'] == 0.0


partial_tblite_output = """
    4   1   2   1       1.94699059  d_xz
    4   1   2   2       1.95944185  d_x2-y2
    4   2   0   0       0.53217851  s
    4   3   1  -1       0.23598867  p_y
    4   3   1   0       0.23598867  p_z
    4   3   1   1       0.23598867  p_x
 
Fermi level:                        -0.4495587982 H          -12.2331 eV
Band energy:                       -21.9227015154 H         -596.5471 eV
TS:                                  0.0000949682 H            0.0026 eV
Band free energy (E-TS):           -21.9227964836 H         -596.5496 eV
Extrapolated E(0K):                -21.9227489995 H         -596.5484 eV
Input / Output electrons (q):     44.0000000000     44.0000000000
 
Energy H0:                         -18.2466149624 H         -496.5157 eV
Energy SCC:                         -0.0415167324 H           -1.1297 eV
Total Electronic energy:           -18.2881316948 H         -497.6454 eV
Repulsive energy:                    0.0000000000 H            0.0000 eV
Total energy:                      -18.2881316948 H         -497.6454 eV
Extrapolated to 0:                 -18.2881791788 H         -497.6467 eV
Total Mermin free energy:          -18.2882266629 H         -497.6480 eV
Force related energy:              -18.2882266629 H         -497.6480 eV
 
SCC converged
 
Full geometry written in geo_end.{xyz|gen}
 
Geometry converged
"""


def test_parse_tb_output():
    result: dict = parse_dftb_output(partial_tblite_output)
    assert result['energy_h0'] == -496.5157
    assert result['energy_scc'] == -1.1297
    assert result['total_electronic_energy'] == -497.6454
    assert result['repulsive_energy'] == 0.0000
    assert result['total_energy'] == -497.6454
