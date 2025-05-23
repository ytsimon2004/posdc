import numpy as np
from matplotlib.axes import Axes
from scipy.ndimage import gaussian_filter1d

__all__ = [
    'plot_decode_actual_position',
    'plot_train_position',
    'plot_firing_rate',
    'plot_decoding_error',
    'plot_binned_decoding_error'
]


def plot_decode_actual_position(ax: Axes,
                                time: np.ndarray,
                                predicted_pos: np.ndarray,
                                actual_pos: np.ndarray,
                                time_mask: tuple[float, float] | None = None):
    """
    Plot decoded/actual animal's position

    :param ax: ``Axes``
    :param time: Time for animal's position. `Array[float, T]`
    :param predicted_pos: Predicted animal's position. `Array[float, T]`
    :param actual_pos: Actual animal's position. `Array[float, T]`
    :param time_mask: Time masking (START, END)
    """
    plot_kw = {'markerfacecolor': None, 'markersize': 3}
    ax.plot(time, predicted_pos, 'r.', label='decoded', alpha=0.5, **plot_kw)
    ax.plot(time, actual_pos, 'k.', label='actual position', alpha=0.3, **plot_kw)

    if time_mask is not None:
        ax.set(ylabel='position(cm)', xlim=time_mask)
    else:
        ax.set(ylabel='position(cm)')  # keep correct extend

    ax.legend()

def plot_train_position(ax: Axes, time: np.ndarray, pos: np.ndarray):
    kwargs = {'markerfacecolor': None, 'markersize': 3}
    ax.plot(time, pos, 'g.', label='train', alpha=0.5, **kwargs)
    ax.set(ylabel='position(cm)')
    ax.legend()


def plot_firing_rate(ax: Axes, time: np.ndarray, fr: np.ndarray, time_mask: tuple[float, float] | None, **kwargs):
    """
    Heatmap for activity

    :param ax: ``Axes``
    :param time: Activity time. `Array[float, T]`
    :param fr: Neural activity. `Array[float, [N, T]]`
    :param time_mask: Time masking (START, END)
    :param kwargs: Additional arguments pass to ``ax.set()``
    """
    if time_mask is None:
        time_mask = (0, np.max(time))

    ax.imshow(fr.T, aspect='auto',
              cmap='Greys',
              # interpolation='none',
              origin='lower',
              extent=(*time_mask, 0, fr.shape[1]))
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
    Plot decoding error (mean+-sem) as a function of position bins

    :param ax: ``Axes``
    :param trial_length: Trial length in cm
    :param error: Trial-averaged binned decoding error. `Array[float, B]`
    :param error_sem: sem binned decoding error across trials. `Array[float, B]`
    """
    n = len(error)
    x = np.linspace(0, trial_length, n)
    ax.plot(x, error)
    ax.fill_between(x, error + error_sem, error - error_sem, color='grey', alpha=0.5)
    ax.set(xlabel='position(cm)', ylabel='decoding error(cm)')
