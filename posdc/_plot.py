import numpy as np
from matplotlib.axes import Axes

__all__ = ['plot_decode_actual_position']

def plot_decode_actual_position(ax: Axes,
                                time: np.ndarray,
                                predicted_pos: np.ndarray,
                                actual_pos: np.ndarray,
                                **kwargs):
    ax.plot(time, predicted_pos, color='r', label='decoded', alpha=0.6, **kwargs)
    ax.plot(time, actual_pos, color='k', label='actual position', alpha=0.4, **kwargs)
    ax.set(ylabel='cm')
    ax.legend()
