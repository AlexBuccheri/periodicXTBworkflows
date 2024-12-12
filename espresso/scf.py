""" Run an Espresso SCF calculation with ASE
"""
from ase.calculators.espresso import Espresso
from ase.atoms import Atoms
from typing import List, Tuple, Union


def run_scf_calculation(run_dir, input_data: dict, atoms: Atoms, kpts: List[int], pseudos: dict) -> Tuple[Atoms, Espresso]:
    """ Run an SCF Calculation with Espresso.

    Perform a regular, self-consistent calculation, saving the wave functions at the end.

    :param run_dir: Run directory.
    :param input_data: QE input file options, in a dict.
    :param atoms: ASE atomic structure object.
    :param kpts: Monkhorst Pack k-grid. If not passed, put into input_data['kpts']
    :return: Atoms and an evaluated Espresso calculator.
    """
    calculator = Espresso(directory=run_dir, input_data=input_data, kspacing=None, kpts=kpts, pseudopotentials=pseudos)
    # Attach the calculator to the atoms object
    atoms.calc = calculator
    # Run an SCF calc
    atoms.get_potential_energy()
    return atoms, calculator


def get_scf_input(results: dict) -> Union[dict, ValueError]:
    """ Get final (converged) input settings from results.

    :param results: Results dict for ONE material.
    """
    if results['converged'] == "Not Run":
        return ValueError("This system was not run")

    # Assumes ordered dict, python >3.7
    key = list(results)[-1]
    return results[key]['input']
