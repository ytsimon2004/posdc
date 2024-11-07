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
                                actual_pos: np.ndarray,
                                **kwargs):
    t = np.linspace(time[0], time[-1], len(predicted_pos))

    ax.plot(t, predicted_pos, color='r', label='decoded', alpha=0.6, **kwargs)
    ax.plot(t, actual_pos, color='k', label='actual position', alpha=0.4, **kwargs)
    ax.set(ylabel='cm')
    ax.legend()


def plot_firing_rate(ax: Axes, time: np.ndarray, fr: np.ndarray, rate_map: np.ndarray):
    """heatmap for sorted firing rate of all cells"""
    sort_idx = _sort_neuron(rate_map.T)
    fr = fr[:, sort_idx]
    ax.imshow(fr.T,
              aspect='auto',
              cmap='magma',
              interpolation='none',
              origin='lower',
              extent=(0, np.max(time), 0, fr.shape[1]))
    ax.set(ylabel='# neurons')


def plot_decoding_err(ax: Axes, time: np.ndarray, decode_err: np.ndarray, cutoff: float):
    """
    plot decoding error as a function of temporal bins

    :param ax:
    :param time:
    :param decode_err
    :param cutoff:
    :return:
    """
    t = np.linspace(time[0], time[-1], len(decode_err))
    ax.plot(t, decode_err, color='k', alpha=0.3, label='frame-wise')
    smooth_err = gaussian_filter1d(decode_err, 10)
    ax.plot(t, smooth_err, color='k', label='smooth')
    ax.axvline(cutoff, color='r', linestyle='--')
    ax.set(xlabel='time(sec)', ylabel='decoding error (cm)', ylim=(0, 70))
    ax.legend()


def _sort_neuron(data: np.ndarray) -> np.ndarray:
    """
    sort neurons based on maximal activity along the belt

    :param data: 2d binned calactivity data. (N, B)
    :return: sorted indices
    """
    m_filter = gaussian_filter1d(data, 3, axis=1)
    m_argmax = np.argmax(m_filter, axis=1)
    return np.argsort(m_argmax)
