"""
Main workflow operations:
    generate TB lite inputs for a range of

TODO Alex. The directory generation should be done separately to the file generation
- Really makes routines inflexible.
"""
import os
from pathlib import Path

import ase
import numpy as np
from ase.io.dftb import write_dftb

from tb_lite.src.dftb_input import DftbInput
from tb_lite.src.parsers.parsers import parse_qcore_structure


def directory_name(output_directory: str, material: str, multiplier: float) -> str:
    """ Define directory name
    :param output_directory:
    :param material:
    :param multiplier:
    :return:
    """
    return os.path.join(output_directory, material.split('.')[0], str(round(multiplier, 4)))


def generate_inputs(input_directory: str,
                    output_directory: str,
                    material: str,
                    dftb_input: DftbInput,
                    lattice_multipliers=np.arange(0.8, 1.2, step=0.025)):
    """DFTB+ TBLite and geometry strings written to file

    For writing DFTB+ with ASE, see:
    https://wiki.fysik.dtu.dk/ase/_modules/ase/io/dftb.html#write_dftb
    Annoyingly no option to write to .gen in fractional (even though DFTB+ accepts it)

    ASE Atoms():
    https://wiki.fysik.dtu.dk/ase/ase/atoms.html#ase.Atoms.cell

    :param input_directory: Location of input file
    :param output_directory: Run/output directory
    :param material: Material file
    :param dftb_input: DFTB input objects
    :param lattice_multipliers: Lattice multipliers
    :return: DFTB+ TBLite and geometry strings written to file
    """
    # QCore
    qcore_file = os.path.join(input_directory, material)
    structure = parse_qcore_structure(qcore_file)
    # TBLite
    dftb_input_str = dftb_input.generate_dftb_hsd()

    for multiplier in lattice_multipliers:
        # Run/output directory
        dir = directory_name(output_directory, material, multiplier)
        Path(dir).mkdir(parents=True, exist_ok=True)

        # DFTB input file
        with open(dir + "/dftb_in.hsd", "w") as fid:
            fid.write(dftb_input_str)

        # Output the structure:
        cell = []
        for vector in structure['lattice']:
            cell.append([multiplier * r for r in vector])

        assert structure['lattice_vectors_unit'] == 'angstrom'
        atoms = ase.Atoms(symbols=structure['species'],
                          scaled_positions=structure['fractional_positions'],
                          cell=cell,
                          pbc=True)

        write_dftb(dir + "/geometry.gen", atoms)

