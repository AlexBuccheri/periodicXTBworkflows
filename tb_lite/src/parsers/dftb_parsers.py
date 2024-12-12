""" DFTB+ and TB Lite Parsers
"""
import re
import numpy as np
import pathlib
import os
import subprocess
from typing import Union


def parse_dftb_output(input: str) -> dict:
    """
    Parse selected DFTB+ TB lite outputs (from detailed.out) in eV.

    End of file has the typical structure:
    '''
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
    '''

    :param input: File string contents
    :return: results dictionary
    """
    n_electrons_up = re.findall(r'^Nr\. of electrons \(up\): .*$', input, flags=re.MULTILINE)[0].split(':')[-1]
    fermi_level = re.findall(r'^Fermi level: .*$', input, flags=re.MULTILINE)[0].split()[-2]
    energy_h0 = re.findall(r'^Energy H0: .*$', input, flags=re.MULTILINE)[0].split()[-2]
    energy_scc = re.findall(r'^Energy SCC: .*$', input, flags=re.MULTILINE)[0].split()[-2]
    total_electronic_energy = re.findall(r'^Total Electronic energy: .*$', input, flags=re.MULTILINE)[0].split()[-2]
    repulsive_energy = re.findall(r'^Repulsive energy: .*$', input, flags=re.MULTILINE)[0].split()[-2]
    total_energy = re.findall(r'^Total energy: .*$', input, flags=re.MULTILINE)[0].split()[-2]

    results = {'n_electrons_up': n_electrons_up, 'fermi_level': fermi_level, 'energy_h0': energy_h0,
               'energy_scc': energy_scc, 'total_electronic_energy': total_electronic_energy,
               'repulsive_energy': repulsive_energy, 'total_energy': total_energy}

    # Convert all strings to floats
    for key, value in results.items():
        results[key] = float(value)

    return results


def parse_number_of_occupied_bands(detailed_out_str: str) -> float:
    """Parse the number of electrons from DFTB+'s output.

    TODO Extend to parse spin polarisation.
    :return: Number of occupied bands at zero kelvin.
    """
    spin_polarised = False
    electrons_per_band = 1 if spin_polarised else 2
    n_electrons = parse_dftb_output(detailed_out_str)['n_electrons_up']
    n_occupied_bands = float(n_electrons) / float(electrons_per_band)
    return n_occupied_bands


def parse_dftb_bands(directory) -> np.ndarray:
    """
    Return band energies from a DFTB+ TB Lite Output

    Need DFTB+ output to be in a sensible format, so apply this script: dp_bands band.out band
    Went to dftb-19/tools folder on my and installed dp_tools via the terminal, to this virtual env:
    `python setup.py install --prefix /Users/alexanderbuccheri/Python/pycharm_projects/tb_benchmarking/venv/`

    bands.shape = (n_k_points, n_bands),
    where n_k_points is the total number of k-points used across all bands paths

    :param directory: Directory containing band structure file.
    :return: band energies. Looks like always in eV.
    """
    if not pathlib.Path(directory).is_dir():
        raise NotADirectoryError(f'Directory does not exist: {directory}')

    processed_file = os.path.join(directory, 'band_tot.dat')

    if not os.path.isfile(processed_file):
        result = subprocess.run(['dp_bands', 'band.out', 'band'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                cwd=directory)
        if result.returncode != 0:
            raise subprocess.SubprocessError(f'Not able to convert band.out: {result.stderr}')

    with open(processed_file, 'r') as fid:
        lines = fid.readlines()

    n_k_points = len(lines)
    n_bands = len(lines[0].split()[1:])

    # File signature:  ith_kpoint eigenvalue1, eigenvalue2, ..., eigenvalue_ith_band\
    bands = np.empty(shape=(n_k_points, n_bands))

    for ik, line in enumerate(lines):
        band_energies = [float(x) for x in line.split()[1:]]
        bands[ik, :] = band_energies

    return bands


def parse_geometry_gen(file_name: Union[str, pathlib.Path]):
    """ Parse DFTB+'s geometry gen file.

    Assumes no header

    :return: Dict of data in gen file.
    Lattice vectors returned row-wise.
    """
    with open(file_name, "r") as fid:
        lines = fid.readlines()

    n_atoms, boundary_conditions = lines[0].split()
    n_atoms = int(n_atoms)
    unique_species = lines[1].split()

    positions = np.empty(shape=(n_atoms, 3))
    species = []
    for i in range(0, n_atoms):
        arb_index, species_index, x, y, z = lines[2 + i].split()
        species_symbol = unique_species[int(species_index) - 1]
        species.append(species_symbol)
        positions[i, :] = [float(x), float(y), float(z)]

    origin = np.asarray([float(r) for r in lines[2 + n_atoms].split()])

    lattice = np.empty(shape=(3, 3))
    for i in range(0, 3):
        j = 3 + n_atoms + i
        lattice[i, :] = np.asarray([float(r) for r in lines[j].split()])

    return {"n_atoms": n_atoms,
            "boundary_conditions": boundary_conditions,
            'species': species,
            "origin": origin,
            "lattice": lattice,
            "positions": positions}
