"""
Utility Classes
"""
from ase.data import chemical_symbols, atomic_names
import numpy as np
import enum
from pathlib import Path


@enum.unique
class Units(enum.Enum):
    Hartree = enum.auto()
    eV = enum.auto()


class Value:
    """ Container for value and unit  """

    def __init__(self, value: float, unit='') -> None:
        self.value = value
        self.unit = unit

    def to_dict(self):
        unit = self.unit
        if isinstance(self.unit, Units):
            unit = self.unit.name
        return {'value': self.value, 'unit': unit}


class FileUrl:
    """ Container for local file location and URL reference  """

    def __init__(self, file, url=None) -> None:
        self.file = file
        self.url = url


def replace_item(obj, key, replace_value):
    """
    https://stackoverflow.com/questions/45335445/how-to-recursively-replace-dictionary-values-with-a-matching-key

    :param obj:
    :param key:
    :param replace_value:
    :return:
    """
    for k, v in obj.items():
        if isinstance(v, dict):
            obj[k] = replace_item(v, key, replace_value)
    if key in obj:
        obj[key] = replace_value
    return obj


def get_names_to_symbols() -> dict:
    """ Dictionary of (key, value) = (element names, element symbols)

    Should build once, but the overhead is small enough
    :return:
    """
    name_to_symbol = {}
    for i in range(len(chemical_symbols)):
        name_to_symbol[atomic_names[i]] = chemical_symbols[i]
    return name_to_symbol


def material_to_symbol(material: str) -> str:
    # In some keys, polymorphs are defined as "name_poly"
    name = material.split('_')[0]
    name_to_symbol = get_names_to_symbols()
    symbol = name_to_symbol[name.capitalize()]
    return symbol


def reciprocal_lattice_vectors(a: np.ndarray):
    """Get the reciprocal lattice vectors of real-space lattice vectors {a}
    :param a: lattice vectors, stored column-wise
    :return: Reciprocal lattice vectors, stored row-wise
    """
    volume = np.dot(a[0, :], np.cross(a[1, :], a[2, :]))
    b = np.empty(shape=(3, 3))
    b[0, :] = 2 * np.pi * np.cross(a[1, :], a[2, :]) / volume
    b[1, :] = 2 * np.pi * np.cross(a[2, :], a[0, :]) / volume
    b[2, :] = 2 * np.pi * np.cross(a[0, :], a[1, :]) / volume
    return b


def get_project_root() -> Path:
    """ Get the project's root directory.

    Assumes a fixed directory structure at least a fixed position of utils.py w.r.t.
    the top level folder.
    Ref: https://stackoverflow.com/questions/25389095/python-get-path-of-root-project-structure
    :return Absolute path to root.
    """
    tb_lite_root = Path(__file__).parent.parent
    tb_benchmarking_root = tb_lite_root.parent
    return tb_benchmarking_root


def serve_unique_directory(directory: Path, n_unique=500) -> Path:
    """Given a directory,
    return if unique, else append an integer until unique
    """
    if not directory.exists():
        return directory

    new_dir = directory / '_'
    for i in range(1, n_unique + 1):
        new_dir = new_dir / str(i)
        if not new_dir.exists():
            return new_dir

    raise IsADirectoryError(f'{directory}_1 - {n_unique} already taken')
