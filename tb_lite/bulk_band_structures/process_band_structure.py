""" Band structure plotting and comparison
"""
import matplotlib.pyplot as plt
from scipy.constants import physical_constants
import pickle
import bz2

from tb_lite.src.dataclasses.band_structure import BandData
from tb_lite.src.ase_bandpath import ASEBandPath
from tb_lite.src.utils import Units

ha_to_ev = physical_constants['Hartree energy in eV'][0]


def plot_band_structure(material: str, bands_data: BandData, save_image=False):
    """
    """
    print(f'Generating Band Structure Plot for {material}')

    path_data = bands_data.band_path
    assert isinstance(path_data, ASEBandPath), \
        "Plotting expects band_path attribute to be of type(ASEBandPath)"

    # Always want band structure in eV
    conversions = {Units.eV: 1., Units.Hartree: ha_to_ev}

    # Initialise figure
    fig, ax = plt.subplots(figsize=(9, 9))

    # x range and labels
    k_vector = path_data.flattened_k
    xticks = path_data.xticks
    labels = path_data.labels

    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)

    # Font sizes
    ax.yaxis.label.set_size(20)
    ax.tick_params(axis='both', which='major', labelsize=20)

    # y range and label
    plt.ylim(-10, 10)
    plt.ylabel('Energy (eV)')

    # Set zero-point of plot
    bands_data.set_bands_zeropoint(bands_data.fermi_level)

    # Colour valence and conduction bands differently
    line_colour = {key: 'blue' for key in range(0, bands_data.n_occupied_bands)}
    line_colour.update({key: 'red' for key in range(bands_data.n_occupied_bands, bands_data.n_bands)})

    # Plot band structure
    conversion = conversions[path_data.unit]
    for ib in range(0, bands_data.n_bands):
        plt.plot(k_vector, conversion * bands_data.bands[:, ib], color=line_colour[ib])

    # Vertical lines at high symmetry points
    for x in xticks:
        plt.axvline(x, linestyle='--', color='black')

    plt.show()

    if save_image:
        plt.savefig(f'{material}_bandstructure.jpeg', dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', transparent=True, bbox_inches=None, pad_inches=0.1)


if __name__ == "__main__":

    # Read serialised result
    with bz2.open('tblite_bandstructures.pickle', 'rb') as fid:
        results = pickle.load(fid)

    # Post-process
    for name, p_c in results.items():
        print(f'Plotting band structure for {name}')
        process_result = p_c['process']
        band_structure = p_c['calculator']
        band_structure.print()
        plot_band_structure(name, band_structure, save_image=True)
