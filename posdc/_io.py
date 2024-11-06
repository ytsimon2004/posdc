from typing import NamedTuple

import pandas as pd
from neuralib.locomotion import CircularPosition
from scipy.interpolate import interp1d
from typing_extensions import Self
from neuralib.typing import PathLike

import numpy as np

__all__ = ['PositionDecodeInput']


class PositionDecodeInput(NamedTuple):
    activity: np.ndarray
    """`Array[float, [N, T]]`"""
    act_time: np.ndarray
    """`Array[float, T]`"""
    position: np.ndarray
    """`Array[float, P]`"""
    position_time: np.ndarray
    """`Array[float, P]`"""
    lap_time: np.ndarray
    """`Array[float, L]`"""
    light_off_lap: int
    """Lap index for light off epoch"""
    light_off_time: float
    """Time for light off epoch"""

    pos_norm: int

    # noinspection PyUnresolvedReferences
    @classmethod
    def load_hdf(cls, file: PathLike,
                 use_deconv: bool = False,
                 pos_norm: int = 150) -> Self:
        dat = pd.read_hdf(file)
        act = dat.deconv if use_deconv else dat.df_f
        position = dat.position_raw * pos_norm if pos_norm is not None else dat.position_raw

        return cls(act, dat.frametimes,
                   position, dat.position_time,
                   dat.laptimes, dat.lights_off_lap, dat.lights_off_time, pos_norm)

    @property
    def n_neurons(self) -> int:
        return self.activity.shape[0]

    @property
    def n_samples(self) -> int:
        return self.activity.shape[1]

    @property
    def activity_sampling_rate(self) -> float:
        """approximate frame rate"""
        return np.median(1 / np.diff(self.act_time))

    @property
    def n_trials(self) -> int:
        return len(self.lap_time)

    @property
    def trial_index(self) -> np.ndarray:
        return np.arange(self.n_trials)

    def get_interp_position(self, sampling_rate: int = 100) -> CircularPosition:
        from neuralib.locomotion import interp_pos1d
        return interp_pos1d(self.position_time, self.position, sampling_rate=sampling_rate)

