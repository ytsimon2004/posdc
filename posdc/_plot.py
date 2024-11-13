import numpy as np
from matplotlib.axes import Axes
from scipy.ndimage import gaussian_filter1d

__all__ = [
    'plot_decode_actual_position',
    'plot_firing_rate',
    'plot_decoding_err'
]


def plot_decode_actual_position(ax: Axes,
                                time: np.ndarray,
                                predicted_pos: np.ndarray,
                                actual_pos: np.ndarray):
    """
    Plot decoded/actual animal's position

    :param ax: ``Axes``
    :param time: Time for animal's position. `Array[float, T]`
    :param predicted_pos: Predicted animal's position. `Array[float, T]`
    :param actual_pos: Actual animal's position. `Array[float, T]`
    """
    ax.plot(time, predicted_pos, 'r.', label='decoded', alpha=0.5, markerfacecolor=None)
    ax.plot(time, actual_pos, 'k.', label='actual position', alpha=0.3)
    ax.set(ylabel='position(cm)')
    ax.legend()


def plot_firing_rate(ax: Axes, time: np.ndarray, fr: np.ndarray, **kwargs):
    """
    Heatmap for activity

    :param ax: ``Axes``
    :param time: Activity time. `Array[float, T]`
    :param fr: Neural activity. `Array[float, [N, T]]`
    :param kwargs: Additional arguments pass to ``ax.set()``
    """
    ax.imshow(fr.T,
              aspect='auto',
              cmap='binary',
              interpolation='none',
              origin='lower',
              extent=(0, np.max(time), 0, fr.shape[1]))
    ax.set(**kwargs)


def plot_decoding_err(ax: Axes, time: np.ndarray, decode_err: np.ndarray, cutoff: float):
    """
    Plot decoding error as a function of temporal bins

    :param ax: ``Axes``
    :param time: Time for decoding error. `Array[float, T]`
    :param decode_err: Decoding error. `Array[float, T]`
    :param cutoff: Axvline cutoff for the different recording session (condition)
    """
    ax.plot(time, decode_err, 'k.', alpha=0.3, label='frame-wise')
    smooth_err = gaussian_filter1d(decode_err, 10)
    ax.plot(time, smooth_err, 'k.', label='smooth')
    ax.axvline(cutoff, color='r', linestyle='--')
    ax.set(xlabel='time(sec)', ylabel='decoding error (cm)', ylim=(0, 70))
    ax.legend()

