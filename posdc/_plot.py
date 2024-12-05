import numpy as np
from matplotlib.axes import Axes
from scipy.ndimage import gaussian_filter1d

__all__ = [
    'plot_decode_actual_position',
    'plot_firing_rate',
    'plot_decoding_error',
    'plot_binned_decoding_error'
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


def plot_decoding_error(ax: Axes, time: np.ndarray, error: np.ndarray, cutoff: float):
    """
    Plot decoding error as a function of temporal bins

    :param ax: ``Axes``
    :param time: Time for decoding error. `Array[float, T]`
    :param error: Decoding error. `Array[float, T]`
    :param cutoff: Axvline cutoff for the different recording session (condition)
    """
    ax.plot(time, error, 'k.', alpha=0.3, label='frame-wise')
    smooth_err = gaussian_filter1d(error, 10)
    ax.plot(time, smooth_err, 'k.', label='smooth')
    ax.axvline(cutoff, color='r', linestyle='--')
    ax.set(xlabel='time(sec)', ylabel='decoding error (cm)', ylim=(0, 70))
    ax.legend()


def plot_binned_decoding_error(ax: Axes,
                               trial_length: float,
                               error: np.ndarray,
                               error_sem: np.ndarray):
    """

    :param ax: ``Axes``
    :param trial_length: Trial length in cm
    :param error: Trial-averaged binned decoding error. `Array[float, B]`
    :param error_sem: sem binned decoding error across trials. `Array[float, B]`
    :return:
    """
    n = len(error)
    x = np.linspace(0, trial_length, n)
    ax.plot(x, error)
    ax.fill_between(x, error + error_sem, error - error_sem, color='grey', alpha=0.5)
    ax.set(xlabel='position(cm)', ylabel='decoding error(cm)')
