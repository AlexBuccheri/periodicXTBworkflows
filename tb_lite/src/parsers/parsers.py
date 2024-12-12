"""
Parsers
"""
import os
import glob
import io
from pathlib import PurePath, Path
import warnings

import ase
from pymatgen.io.cif import CifParser


def reader(func):
    """Decorate func so it can receive a file name, a file ID, or string from file.
    """

    def modified_func(input, *args, **kwargs):
        if isinstance(input, io.TextIOWrapper):
            file_string = input.read()
            input.close()
        elif isinstance(input, (str, PurePath)):
            if Path(input).is_file():
                with open(input, "r") as fid:
                    file_string = fid.read()
            # Assume the string is the file string, and not an erroneous file name (could be a big source of error here)
            else:
                file_string = input
        else:
            raise ValueError(f"Input is neither an IO handle, nor a string: {type(input)}")
        return func(file_string, *args, **kwargs)

    return modified_func


def cif_to_ase_atoms(file: str) -> ase.atoms.Atoms:
    """Convert CIF to ASE Atoms.

    ASE's read_cif() doesn't seem robust, or I'm using it wrong, so use pymatgen instead.
    :param file:
    :return:
    """
    structure = CifParser(file).get_structures()[0]
    # Suppress pymatgen/io/cif.py:1224: UserWarning: Issues encountered while parsing CIF:
    # 8 fractional coordinates rounded to ideal values to avoid issues with finite precision.
    warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF: .*", category=UserWarning)
    atoms = ase.atoms.Atoms(numbers=structure.atomic_numbers,
                            cell=structure.lattice.matrix,
                            scaled_positions=structure.frac_coords,
                            pbc=True)
    return atoms


def clear_directory(directory: str):
    """ Remove all files from a directory

    :param directory: Directory to clear
    """
    files = glob.glob(os.path.join(directory, '*'))
    for file in files:
        os.remove(file)


def parse_x23_xyz_to_ase(file) -> ase.atoms.Atoms:
    """Parser specific for xyz files downloaded from SI of X23 paper.
    Format does not appear to be standard:
    Natoms
    a1
    a2
    a3
    symbol  x  y  z partial-charge
    ...
    """
    with open(file, mode='r') as fid:
        lines = fid.readlines()

    n_atoms = int(lines[0])
    lattice_vectors = [[float(x) for x in v.split()] for v in lines[1:4]]
    symbols = []
    positions = []

    for line in lines[4:n_atoms + 4]:
        symbol, x, y, z, partial_charge = line.split()
        symbols.append(symbol)
        positions.append((float(x), float(y), float(z)))

    atoms = ase.atoms.Atoms(symbols=symbols,
                            cell=lattice_vectors,
                            positions=positions,
                            pbc=True)

    return atoms
