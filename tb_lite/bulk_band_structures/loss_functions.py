"""

"""
import numpy as np


def valence_loss_function(tb_bands, qe_bands) -> float:
    """ Function to minimise in band structure fit.

    Evaluate the root-mean-square deviation.
    Optimise the fit for the valence bands.

    :param tb_bands: All TB valence bands
    :param qe_bands: All QE valence bands
    :return loss_value: root-mean-square deviation for valence bands.
    """
    assert tb_bands.shape[0] == qe_bands.shape[0], 'k-sampling is not consistent'
    n_kpt = tb_bands.shape[0]

    # DFT expected to contain more valence states than TB, so only
    # evaluate loss function for bands available in both cases
    n_valence = min(tb_bands.shape[1], qe_bands.shape[1])
    n_total = n_valence * n_kpt

    i = tb_bands.shape[1] - n_valence
    j = qe_bands.shape[1] - n_valence

    # All (valence) bands and k-points weighted equally
    loss_value = np.sqrt(np.sum(np.square(tb_bands[:, i:] - qe_bands[:, j:])) / n_total)

    return loss_value


def band_edges_loss_function():
    pass
