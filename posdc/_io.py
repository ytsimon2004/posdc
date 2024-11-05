from typing import NamedTuple

import pandas as pd
from scipy.interpolate import interp1d
from typing_extensions import Self
from neuralib.typing import PathLike

import numpy as np

__all__ = ['PositionDecodeInput']


class PositionDecodeInput(NamedTuple):
    activity: np.ndarray
    """`Array[float, [N, T]]`"""
    time: np.ndarray
    """`Array[float, T]`"""
    position: np.ndarray
    """`Array[float, T]`"""
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
        position = dat.position * pos_norm if pos_norm is not None else dat.position

        return cls(act, dat.frametimes, position, dat.laptimes, dat.lights_off_lap, dat.lights_off_time, pos_norm)

    @property
    def n_neurons(self) -> int:
        return self.activity.shape[0]

    @property
    def n_samples(self) -> int:
        return self.activity.shape[1]

    @property
    def activity_sampling_rate(self) -> float:
        """approximate frame rate"""
        return np.median(1 / np.diff(self.time))

    @property
    def n_trials(self) -> int:
        return len(self.lap_time)

    @property
    def trial_index(self) -> np.ndarray:
        return np.arange(self.n_trials)

    @property
    def velocity(self) -> np.ndarray:
        dp = np.diff(self.position, append=self.position[0])

        # circular continuity
        x1 = dp + self.pos_norm * 0.5
        x2 = self.pos_norm
        dp = np.mod(x1, x2) - self.pos_norm * 0.5
        dt = np.diff(self.time, append=self.time[-1] + (self.time[-1] - self.time[-2]))

        return dp / dt

    def with_interpolation(self) -> Self:
        # TODO test
        from neuralib.locomotion import interp_pos1d
        pos = interp_pos1d(self.time, self.position, sampling_rate=30, norm_max_value=self.pos_norm)

        # activity
        act = interp1d(self.time, self.activity, bounds_error=False, fill_value=0)(pos.t)

        return self._replace(activity=act, position=pos.p, time=pos.t)
