from typing import NamedTuple

import numpy as np
import rastermap.utils
from rastermap import Rastermap

__all__ = ['run_rastermap']


class RastermapResult(NamedTuple):
    """
    `Dimension parameters`:

        N = Number of neurons/pixel

        T = Number of image pulse

        C = Number of clusters = N / binsize

    """
    isort: np.ndarray
    """`Array[int, N]`"""
    embedding: np.ndarray
    """`Array[float, [N, 1]]`"""
    super_neurons: np.ndarray
    """super neuron activity. `Array[float, [C, T]]`"""


def run_rastermap(activity: np.ndarray,
                  bin_size: int = 20,
                  *,
                  n_clusters: int = 50,
                  n_pcs: int = 128,
                  locality: float = 0.75,
                  time_lag_window: int = 5,
                  grid_upsample: int = 10,
                  **kwargs) -> RastermapResult:
    """

    :param activity: Neural activity. `Array[float, [N, T]]`
    :param bin_size: Bin size for number of total neurons
    :param n_clusters: Number of clusters created from data before upsampling and creating embedding
    :param n_pcs: Number of PCs to use during optimization
    :param locality: How local should the algorithm be -- set to 1.0 for highly local + sequence finding,
        and 0.0 for global sorting
    :param time_lag_window: Number of time points into the future to compute cross-correlation,
        useful to set to several timepoints for sequence finding
    :param grid_upsample: How much to upsample clusters, if set to 0.0 then no upsampling
    :param kwargs: Additional keyword arguments pass to ``Rastermap()``
    :return: ``RastermapResult``
    """
    model = Rastermap(
        n_clusters=n_clusters,
        n_PCs=n_pcs,
        locality=locality,
        time_lag_window=time_lag_window,
        grid_upsample=grid_upsample,
        **kwargs
    ).fit(activity)

    embedding = model.embedding
    isort = model.isort
    sn = rastermap.utils.bin1d(activity[isort], bin_size=bin_size, axis=0)

    return RastermapResult(isort, embedding, super_neurons=sn)
