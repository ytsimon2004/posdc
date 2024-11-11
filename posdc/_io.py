from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from neuralib.locomotion import CircularPosition
from neuralib.typing import PathLike
from typing_extensions import Self

__all__ = ['PositionDecodeInput']


class PositionDecodeInput(NamedTuple):
    filepath: Path
    """For cache temporal data"""
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
    trial_length: int
    """Trial length in cm"""

    # noinspection PyUnresolvedReferences
    @classmethod
    def load_hdf(cls, file: PathLike,
                 use_deconv: bool = False,
                 trial_length: int = 150) -> Self:
        dat = pd.read_hdf(file)
        act = dat.deconv if use_deconv else dat.df_f

        return cls(Path(file), act, dat.frametimes,
                   dat.position_raw, dat.position_time,
                   dat.laptimes, dat.lights_off_lap, dat.lights_off_time, trial_length)

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

    def get_light_trange(self) -> tuple[int, int]:
        x = self.lap_time < self.light_off_time
        ret = self.trial_index[x]

        return int(ret[0]), int(ret[-1])

    def get_dark_trange(self, tol: int = 4) -> tuple[int, int]:
        """

        :param tol: tolerance (delay) buffer trials after lights-off
        :return:
        """
        x = self.lap_time >= self.light_off_time
        ret = self.trial_index[x]

        return int(ret[0]) + tol, int(ret[-1])

    @property
    def interp_cache(self) -> Path:
        return self.filepath.with_name(self.filepath.stem + '_position_cache').with_suffix('.npz')

    def load_interp_position(self, sampling_rate: int = 100, force_compute: bool = False) -> CircularPosition:
        """

        :param sampling_rate:
        :param force_compute:
        :return:
        """
        from neuralib.locomotion import interp_pos1d

        if not self.interp_cache.exists() or force_compute:
            pos = interp_pos1d(self.position_time,
                               self.position,
                               norm_max_value=self.trial_length,
                               sampling_rate=sampling_rate)
            np.savez(self.interp_cache, t=pos.t, p=pos.p, d=pos.d, v=pos.v, trial_time_index=pos.trial_time_index)
            return pos
        else:
            pos = np.load(self.interp_cache, allow_pickle=True)
            return CircularPosition(pos['t'], pos['p'], pos['d'], pos['v'], pos['trial_time_index'])
