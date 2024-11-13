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

    :param activity:
    :param bin_size:
    :param n_clusters:
    :param n_pcs:
    :param locality:
    :param time_lag_window:
    :param grid_upsample:
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
