"""
Issues and Points

    Rather than run relaxation with QCore, Rui's done E vs V, and will then find the minimum energy from that.
    One can fit this and find the minimum volume, then compare with DFT
    However:
    One cannot interpolate the shell charges, or any other quantity.
    What Rui should have done was interpolate to find the volume corresponding to the optimised cell parameters
    then run that to get the shell and atomic charges
    - This is not a problem if the discrete volume increments are small (i.e. < 1 %)
    - They're not, they 2%, which means there's an error of ± 1% per dimension.

    How did Rui plot some shell charge quantity vs the difference in total energies?
    Total energy is not an observable, it's essentially meaningless to compare total energies computed with
    two codes (perhaps unless the basis is all-electron).

    - To plot |E - E_GS|, E would need to be the lowest-energy structure from QCore used as an input
    for the DFT calculation. And E_GS would be the total-energy of the DFT-relaxed system.
    => This would require a couple-thousand QE runs

    Rui used Qcore to:
    - Use E vs V to find minimum-energy Qcore structure
    - Pass DFT-relaxed structure as input to Qcore and compute total energy
    - Compare the two
"""
import gzip
import os
import pickle
import warnings
from typing import List, Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt

from emmet.core.mpid import MPID
from pymatgen.core.structure import Structure

from delta_factor.fit_eos import fit_birch_murnaghan_relation, parse_birch_murnaghan_relation
from materials_project.parse_qcore_results import repackage_qcore_data
from materials_project.query_mp_properties import query_properties
from materials_project.properties import get_pair_interactions, diff_electronegativities, Pairs, label_interaction, \
    BondType

# Each user needs their own `MAPI_KEY` in their shell env.
api_key = os.getenv('MAPI_KEY')


def query_dft_volumes(data: dict) -> dict:
    """
    Query MP database for volumes.

    """
    material_ids = [MPID(mp_id) for mp_id in data.keys()]
    # Assume these correspond to relaxed structures, as fields=['initial_structures'] would
    # return input structures  List[Structure]
    docs = query_properties(api_key, material_ids=material_ids, fields=["volume", "material_id"])
    dft_volumes = {doc.material_id: doc.volume for doc in docs}

    return dft_volumes


def fit_e_vs_v(vol, energy, n_points=100) -> Tuple[np.ndarray, np.ndarray]:
    """ Wrapper to fit energy vs volume curves.

    :return linearly_sampled_volume, energy_fit: Model volume and energies.
    """
    fit_data: dict = fit_birch_murnaghan_relation(vol, energy)

    energy_model_func: Callable = parse_birch_murnaghan_relation(fit_data)
    linearly_sampled_volume = np.linspace(vol[0], vol[-1], n_points)
    energy_fit = energy_model_func(linearly_sampled_volume)
    return linearly_sampled_volume, energy_fit


def add_energy_and_volume_data(data: dict) -> tuple:
    """
    Mutate input data to contain equilibrium volumes and energies.
    """
    # Parse relaxed DFT volume DFT from MP
    dft_volumes = query_dft_volumes(data)

    qcore_vol_multipliers = np.power(np.linspace(0.8, 1.2, 21), 3)

    # Reduce the range over which the data is fit
    i1, i2 = 5, 16

    print('Adding energy and volume data:')
    exceptions = []
    for mp_id, attributes in data.items():
        print(mp_id)

        dft_volume = dft_volumes[mp_id]
        qcore_volumes = dft_volume * qcore_vol_multipliers

        try:
            # Min energy and volume from interpolated data points
            denser_qcore_volumes, qcore_energies = fit_e_vs_v(qcore_volumes[i1: i2], attributes['energies'][i1: i2])
            interpolated_qcore_vol_min = True
        except RuntimeError:
            exceptions.append(f'Could not fit birch_murnaghan_relation for {mp_id}')
            # Min energy and volume from computed data points
            denser_qcore_volumes, qcore_energies = qcore_volumes, attributes['energies']
            interpolated_qcore_vol_min = False

        i_min = np.argmin(qcore_energies)
        # Minimum QCore volume
        qcore_volume = denser_qcore_volumes[i_min]
        # # Qcore total energy for Qcore's lowest-energy structure, found with E vs V analysis
        qcore_min_energy = qcore_energies[i_min]

        # Qcore total energy for relaxed DFT structure
        # i.e. a lattice multiplier of 1 (index 10 of 20) corresponds to the DFT ground state structure
        qcore_energy_dft_struct = attributes['energies'][10]

        data[mp_id].update(
            {'qcore_vol_min': qcore_volume,
             'interpolated_qcore_vol_min': interpolated_qcore_vol_min,
             'dft_vol_min': dft_volume,
             'E_relaxed': qcore_energy_dft_struct,
             'E_min': qcore_min_energy
             })

    return data, exceptions


def add_electronegativity(data: dict) -> dict:
    """
    Mutate input data to contain avg and max electronegativities
    per system
    """
    # Get MP structures
    material_ids = [str(MPID(mp_id)) for mp_id in data.keys()]
    docs = query_properties(api_key, material_ids=material_ids, fields=["material_id", "structure"])

    print('Adding electronegativity data')
    for doc in docs:
        print(doc.material_id, doc.structure.formula)
        pair_indices = get_pair_interactions(doc.structure, Pairs.NN)
        average_en, missed_interactions = diff_electronegativities(doc.structure, pair_indices)
        # Note, for max(EN_NN), I'd need to modify what the `diff_electronegativities` returns
        data[doc.material_id].update({'average_EN_NN': (average_en, missed_interactions),
                                      'n_atoms': len(doc.structure.cart_coords),
                                      'ionicity': label_interaction(average_en, as_string=True)
                                      })
    return data


def add_shell_charges(data: dict) -> dict:
    """
    To add to data[mp_id].update( {'average_shell_charge': value,
                                   'max_shell_charge': value} )
    """
    raise NotImplementedError('Implement some descriptor for shell charge of a system')


def sort_wrt_ionicity(data: dict) -> dict:
    """

    """
    sorted_data_keys = {str(BondType.IONIC): [],
                   str(BondType.COVALENT): [],
                   str(BondType.POLAR_COVALENT): []
                   }

    for mp_ip, attributes in data.items():
        label = attributes['ionicity']
        sorted_data_keys[label].append(mp_ip)

    return sorted_data_keys


def plot_vol_distribution(data: dict):
    """
    Plot distribution of |deltaV/V|, and replot as cumulative distribution
        * Mean should be half-way along the plot
        * Again, can split this according to  covalent, polar and ionic labelled structures
    """
    vol_err = []
    for attributes in data.values():
        error = np.abs(attributes['qcore_vol_min'] - attributes['dft_vol_min']) / attributes['dft_vol_min']
        vol_err.append(error)

    count, bins_count = np.histogram(vol_err, bins=100)
    # PDF of the histogram
    pdf = count / sum(count)
    # CDF
    cdf = np.cumsum(pdf)

    # plotting PDF and CDF
    # PDF looks junk
    # plt.plot(bins_count[1:], pdf, color="red", label="PDF")
    plt.plot(bins_count[1:], cdf, label="CDF")
    plt.legend()
    plt.show()


def plot_energy_energy_correlation(data: dict):
    """ Plot the Qcore total-energy for the relaxed DFT structure vs the QCore total-energy for
    Qcore's lowest-energy volume.

        * Qcore E_min / n-atoms on y-axis
        * Qcore energy @ v_multiplier = 1 on x-axis
        * See how well they correlate
        * Compute R, R^2, MAE
        * Have 3 plots: Systems labelled covalent, polar and ionic.

    Note, it would be more meaningful if these were DFT energies, but that would require us to
      a) identify the lowest-energy structure computed by QCore (we did this using E vs V)
      b) Get the DFT total-energy for the relaxed structure (should be able to query MP for this)
      c) Compute the DFT total-energy for the minimum-energy QCore cell - we don't have the workflow
      set up to run single-point DFT on ~ 2500 systems.
    """
    sorted_data_keys = sort_wrt_ionicity(data)

    x = np.empty(shape=len(sorted_data_keys[str(BondType.COVALENT)]))
    y = np.empty(shape=len(sorted_data_keys[str(BondType.COVALENT)]))

    for i, key in enumerate(sorted_data_keys[str(BondType.COVALENT)]):
        x[i] = data[key]['E_relaxed'] / float(data[key]['n_atoms'])
        y[i] = data[key]['E_min'] / float(data[key]['n_atoms'])

    # TODO(Alex) Need to add some further criteria for exluding
    # Create a bunch of index maps and combine
    # if max(x_value, y_value) / min(x_value, y_value) > 100:

    plt.scatter(x, y)
    plt.show()


def plot_delta_energy_vs_mean_en(data: dict):
    """ Plot abs( Qcore total-energy for the relaxed DFT structure  -
    QCore total-energy for min(vol) ) / n-atoms, vs average electronegativity

    * Use different colours for covalent, polar and ionic plots
    """
    sorted_data_keys = sort_wrt_ionicity(data)

    def set_x_y(bond_type):
        e_diff = np.empty(shape=len(sorted_data_keys[bond_type]))
        en_diff = np.empty(shape=len(sorted_data_keys[bond_type]))

        for i, key in enumerate(sorted_data_keys[bond_type]):
            e_diff[i] = np.abs(data[key]['E_relaxed'] - data[key]['E_min']) / float(data[key]['n_atoms'])
            en_diff[i] = data[key]['average_EN_NN'][0]

        return en_diff, e_diff



    # Want to remove all energy differences where the difference is >= an order
    # of magnitude
    # TODO(Alex) These are giving the same numbers every time => Check sorted data keys
    x, y = set_x_y(str(BondType.IONIC))
    plt.scatter(x, y, marker='o')

    x2, y2 = set_x_y(str(BondType.COVALENT))
    plt.scatter(x2, y2, marker='^')

    x3, y3 = set_x_y(str(BondType.POLAR_COVALENT))
    plt.scatter(x3, y3, marker='*')

    plt.ylim((0,0.5))
    plt.show()

    # Should just run a cutoff and only retain things where abs error < 0.1 eV per atom
    # voliume error is <= 10 %



def clean_entries_prior_to_mp_queries(input_data: dict) -> dict:
    """

    """
    # Remove calcs that have no clear middle-index (i.e. they
    # weren't computed for the standard 21 volumes)
    expected_num_calcs = 21
    materials_to_remove = []

    for id, data in input_data.items():
        if data['n_calculations'] != expected_num_calcs:
            materials_to_remove.append(id)

    for id in materials_to_remove:
        del input_data[id]

    # Remove any systems where data cannot be retrieved from the MP-id
    n_materials = len(input_data.keys())
    material_ids = [MPID(mp_id) for mp_id in input_data.keys()]
    docs = query_properties(api_key, material_ids=material_ids, fields=["material_id"])

    if len(docs) != n_materials:
        returned_materials = {doc.material_id for doc in docs}
        missing_materials = set(list(input_data.keys())) - returned_materials
        warnings.warn(f'These materials were not returned from MP query {missing_materials}')
        for id in missing_materials:
            input_data.pop(id)

    return input_data


def compute_descriptors(input_file: str):

    # Completely repack QCore data from Rui
    repacked_data: dict = repackage_qcore_data(input_file)

    repacked_data = clean_entries_prior_to_mp_queries(repacked_data)

    # Subset to test on
    # subset_data = {key: value for key, value in list(repacked_data.items())[:10]}
    subset_data = repacked_data

    subset_data, exceptions = add_energy_and_volume_data(subset_data)
    subset_data = add_electronegativity(subset_data)

    if exceptions:
        print('Exceptions: exceptions adding energy and volume data')
        for e in exceptions:
            print(e)
    print(f'Corresponds to {len(exceptions)} / {len(subset_data.keys())} materials')

    # Dump
    with gzip.open("materials_project/structures/data_and_descriptors.pkl.gz", "wb") as file:
        pickle.dump(subset_data, file)

    return


if __name__ == '__main__':

    # Call to parse the raw Qcore data, compute some properties/descriptors
    # and write back to a new file with clean dict structure
    # compute_descriptors('tb_results/qcore/binary_system.dict')

    # Plotting

    # Read the data from the compressed pickle file
    with gzip.open("materials_project/structures/data_and_descriptors.pkl.gz", "rb") as file:
        loaded_data = pickle.load(file)

    # plot_vol_distribution(loaded_data)
    # plot_energy_energy_correlation(loaded_data)
    plot_delta_energy_vs_mean_en(loaded_data)

