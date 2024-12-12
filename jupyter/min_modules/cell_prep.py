"""Cell Preparation
"""
import copy
import math

import numpy as np
from typing import Callable, List


from ase.atoms import Atoms

from delta_factor.fit_eos import fit_birch_murnaghan_relation, parse_birch_murnaghan_relation, EVCurveData
from tb_lite.crystal_references.crystal_systems import bulk_materials


def qe_cell_string(atoms: Atoms) -> str:
    """ CELL string block for QE Input.
    Espresso wants:
    ```
    CELL_PARAMETERS angstrom
     v1(1)  v1(2)  v1(3)    # ... 1st lattice vector
     v2(1)  v2(2)  v2(3)    # ... 2nd lattice vector
     v3(1)  v3(2)  v3(3)    # ... 3rd lattice vector
    ```
    """
    string = "CELL_PARAMETERS angstrom\n"
    for vector in atoms.cell:
        for element in vector:
            string += f"{element:.14f} "
        string += '\n'
    return string


def qe_atomic_positions_string(atoms: Atoms) -> str:
    """ ATOMIC_POSITIONS string block for QE Input
    """
    string = "ATOMIC_POSITIONS angstrom\n"
    for atom in atoms:
        string += atom.symbol + " " + "".join(f"{x:.14f} " for x in atom.position) + "\n"
    return string


def min_energy_cell(qcore_results: dict, material: str, use_discrete: List[str]) -> Atoms:
    """ Produce an atoms object (cell) with lattice vectors and atomic positions
    that correspond to the relaxed structure.

    * Find the lowest-energy cell predicted by QCore, using E vs V data.
      * This only provides the volume (not the lattice vectors).
      * Take this volume, and the volume from the original CIF file to create a
        volume multiplier.
    * Use the volume multiplier on atomic positions and cell vectors from the original
      CIF to create the minimum-energy cell.
    * If the E vs V fit is not a good one (see systems listed in `use_discrete`),
      fall back to the discrete data point, rather than the interpolated model minimum.

    :return qcore_atoms: Atoms object with cell vectors that correspond to QCore's
    minimum-energy cell.
    """
    inconsistent_name_map = {'bn_cubic': 'bn_cubic', 'bn_hex': 'bn_hex', 'cdse': 'cdse', 'diamond': 'diamond',
                             'gaas': 'gaas', 'gan': 'gan',
                             'ge': 'germanium', 'graphite': 'graphite', 'mgo': 'mgo', 'mos2': 'mos2', 'nacl': 'nacl',
                             'pbs': 'pbs',
                             'si': 'silicon', 'tio2_ana': 'tio2_ana', 'tio2_rutile': 'tio2_rutile', 'ws2': 'ws2',
                             'zno': 'zinc_oxide', 'zro2': 'zro2', 'wo3_monoclinic': 'wo3_monoclinic'}

    # Volume associated with the bulk cif files
    atoms: Atoms = bulk_materials[inconsistent_name_map[material]]
    V_input = atoms.get_volume()

    # Minimum volume found from model fit. More precise than using the discrete data
    # Note, this only makes sense if the model fit is good
    fit_data: dict = fit_birch_murnaghan_relation(qcore_results[material].volume, qcore_results[material].energy)
    energy_model: Callable = parse_birch_murnaghan_relation(fit_data)
    linearly_sampled_volume = np.linspace(qcore_results[material].volume[0],
                                          qcore_results[material].volume[-1],
                                          100)
    energy_fit = energy_model(linearly_sampled_volume)
    V_min_fit = linearly_sampled_volume[np.argmin(energy_fit)]

    # Copy bulk system and modify lattice vectors and atomic positions
    qcore_atoms = copy.deepcopy(atoms)
    lattice = math.pow(V_min_fit / V_input, 1/3.0) * atoms.get_cell()
    qcore_atoms.set_cell(lattice, scale_atoms=True)

    # Sanity check - should never be triggered
    max_percentage_diff = 5
    diff = np.abs(qcore_atoms.get_volume() - V_min_fit)
    if (diff * 100 / qcore_atoms.get_volume()) > max_percentage_diff:
        print(f'Greater than a {max_percentage_diff} % discrepancy between desired volume '
              f'and the final cell volume for {material}')
        print(qcore_atoms.get_volume(), V_min_fit)

    # For suboptimal fits, fall back to discrete data
    if material in use_discrete:
        i_min = np.argmin(qcore_results[material].energy)
        V_min_discrete = qcore_results[material].volume[i_min]

        # Load bulk system and modify lattice vectors and atomic positions
        qcore_atoms = copy.deepcopy(atoms)
        lattice = math.pow(V_min_discrete / V_input, 1/3.0) * atoms.get_cell()
        qcore_atoms.set_cell(lattice, scale_atoms=True)

    return qcore_atoms
