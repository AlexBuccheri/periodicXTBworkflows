""" Run Espresso Band structure calculations with the converged settings from
 TB LITE k-grid
 Espresso PW cut-off.
"""
import json
import os.path
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pickle
import bz2

from ase.atoms import Atoms
from ase.calculators.espresso import Espresso
from ase.dft.kpoints import BandPath
from ase.spectrum.band_structure import BandStructure

from espresso.espresso_settings import EspressoSettings, get_dune_qe_settings, number_of_bands
from espresso.inputs import set_espresso_input
from espresso.scf import run_scf_calculation, get_scf_input

from tb_lite.crystal_references.crystal_systems import bulk_materials
from tb_lite.src.band_utils import flatten_whole_kpath
from tb_lite.src.band_gaps import BandGap, direct_bandgap_alt, fundamental_bandgap_alt, get_spin_unpolarised_occupation


# class QEBandStructure(BandStructure):
#     """
#     Extend ASE's BandStructure class
#     """
#     def __init__(self, path, energies, reference=0.0):
#         super(QEBandStructure, self).__init__(path, energies, reference=reference)


def band_structure_calculation(input_data: dict, atoms: Atoms, calculator: Espresso, bz_path: BandPath, n_bands: int,
                               env: EspressoSettings) -> Tuple[BandStructure, BandGap, BandGap, int]:
    """ Run an Espresso Band Structure Calculation.

    Perform a non-self-consistent 'band structure' run.
    To plot the ASE object, `bs.plot()`

    :param input_data: QE input options from an SCF calculation.
    :param atoms: ASE atoms, with the QE calculator attached.
    :param calculator: QE calculator, evaluated for an SCF calculation.
    :param bz_path: BZ path for band structure.
    :param n_bands: Optional number of bands to compute.
    :return: ASE band structure instance
    """
    n_bands_local = n_bands
    with open(env.run_dir / "espresso.pwo") as fid:
        contents = fid.read()
    n_occupied = get_spin_unpolarised_occupation(contents)

    # Ensure there's n_conduction_states, hence a band gap
    if n_occupied <= n_bands:
        material = input_data['control']['prefix'].split('_')[0]
        n_conduction_states = 10
        n_bands_local = n_occupied + n_conduction_states
        message = f'Number of requested bands {n_bands} is <= the number of occupied states {n_occupied}' \
                  f'for {material}.\n Updating n_bands to {n_bands_local} such that a gap can be observed.'
        warnings.warn(message)

    # Update SCF calculator for band structure
    fermi_level = calculator.get_fermi_level()
    print('Fermi level parsed from QE', fermi_level)
    input_data['control'].update({'calculation': 'bands',
                                  'restart_mode': 'restart',
                                  'verbosity': 'high'})
    input_data['system'].update({'nbnd': n_bands_local})

    calculator.set(kpts=bz_path, input_data=input_data)
    calculator.calculate(atoms)
    bs: BandStructure = calculator.band_structure()
    bs._reference = fermi_level

    # Band gaps
    assert bs.energies.ndim == 3, "bs.energies.shape = (nspins, nkpoints, nbands)"
    fundamental: BandGap = fundamental_bandgap_alt(bs.energies[0, :, :], bz_path.kpts, n_occupied)
    direct: BandGap = direct_bandgap_alt(bs.energies[0, :, :], bz_path.kpts, n_occupied)

    return bs.subtract_reference(), fundamental, direct, n_occupied


def serialise_band_structure(bs: BandStructure, output):
    """ For easy GNUPLOTing.
    ADD description.

    :param bs:
    :param output:
    :return:
    """

    # Else the routine needs refactoring
    assert bs.energies.shape[0] == 1, "Expect spin unpolarised calculations"
    # print(f'spin, k_points and bands: {bs.energies.shape}')

    n_k = bs.energies.shape[1]
    n_bands = bs.energies.shape[2]
    bands = np.empty(shape=(n_k, n_bands + 1))

    # Put flattened k-points in first column, followed by band 1, band2, ...
    bands[:, 0] = flatten_whole_kpath(bs.path.kpts)
    bands[:, 1:] = bs.energies[0, :, :]
    np.savetxt(output, bands)


if __name__ == "__main__":

    # Paths
    dune_root = Path('/users/sol/abuccheri/packages/tb_benchmarking')
    run_dir_root = dune_root / 'outputs/espresso_bands'
    scf_json_file = dune_root / 'outputs/espresso_scf/results.json'

    # Compute Espresso band structures
    print('Espresso band structure calculation')
    print(f'Running in {run_dir_root}')
    print('NOTE. It is essential to set `npoints` such that the k-path is consistent with other calculations.')

    # Load all SCF results
    if os.path.isfile(scf_json_file):
        with open(scf_json_file) as fid:
            scf_results = json.load(fid)
    else:
        raise FileNotFoundError('Cannot find SCF results.json')

    materials = [name for name, result in scf_results.items() if result['converged'] is True]
    print('Producing QE band structures for materials listed as converged:', materials, '\n')

    band_structures = {}
    for material in materials:

        # Env and input settings
        env_settings = get_dune_qe_settings(run_dir_root, material)
        converged_settings = get_scf_input(scf_results[material])

        # Skip non-converged values
        if converged_settings is ValueError():
            print(f'{material} SCF did not converge. Skipping.')
            continue

        specific_settings = {'ecutwfc': converged_settings['ecut'],
                             'pseudo_dir': env_settings.pseudo_dir,
                             'conv_thr': 1.e-6}
        k_grid = converged_settings['k_grid']
        qe_input = set_espresso_input(material, **specific_settings)
        atoms: Atoms = bulk_materials[material]

        # 1. (Re)-Run SCF with converged settings
        print(f'Running converged SCF for {material}')
        atoms, calculator = run_scf_calculation(env_settings.run_dir,
                                                qe_input,
                                                atoms,
                                                k_grid,
                                                env_settings.pseudos)

        # 2. Band Structure
        npoints, n_bands = 100, number_of_bands[material]
        print(f'Running band structure for {material} using {npoints} k-points and {n_bands} bands.')

        band_path = atoms.get_cell().bandpath(npoints=npoints)
        bs, eg_fundamental, eg_direct, n_occupied = \
            band_structure_calculation(qe_input, atoms, calculator, band_path, n_bands, env_settings)
        band_structures[material] = {'instance': bs,
                                     'fundamental_gap': eg_fundamental,
                                     'direct_gap': eg_direct,
                                     'n_occupied': n_occupied}

        print(f'Fundamental gap: {eg_fundamental.gap()} (eV) from k = {eg_fundamental.kv} to {eg_fundamental.kc}')
        print(f'Direct gap: {eg_direct.gap()} (eV) at k = {eg_direct.kv}')

        # 3. Serialise result
        pickle_file = run_dir_root / 'espresso_bandstructures.pickle'

        # Load data if file already exists
        if pickle_file.is_file():
            with bz2.open(pickle_file, 'rb') as fid:
                results = pickle.load(fid)
            band_structures.update()

        # Dumpy band structure results to pickle
        with bz2.BZ2File(pickle_file, 'wb') as fid:
            pickle.dump(band_structures, fid, protocol=pickle.HIGHEST_PROTOCOL)

        # Convenient for fast plotting
        serialise_band_structure(band_structures[material]['instance'], env_settings.run_dir / 'band_test.out')
