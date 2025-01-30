from pathlib import Path
from typing import NamedTuple, Literal

import numpy as np
import pandas as pd
from neuralib.locomotion import CircularPosition
from neuralib.typing import PathLike
from typing_extensions import Self

__all__ = ['PositionDecodeInput']

LOAD_INPUT_BACKEND = Literal['numpy', 'pandas']


class PositionDecodeInput(NamedTuple):
    """
    `Dimension parameters`:

        N = Number of neurons

        T = Number of image pulse

        P = Number of position signal

        L = Number of laps(trials)

    """

    filepath: Path
    """For cache temporal data"""

    activity: np.ndarray
    """Neural activity. `Array[float, [N, T]]`"""

    act_time: np.ndarray
    """Time for neural activity. `Array[float, T]`"""

    position: np.ndarray
    """Animal's position. `Array[float, P]`"""

    position_time: np.ndarray
    """Time for animal's position. `Array[float, P]`"""

    lap_time: np.ndarray
    """Time for each laps(trial). `Array[float, L]`"""

    light_off_lap: int
    """Lap index for light off epoch"""

    light_off_time: float
    """Time for light off epoch"""

    trial_length: int
    """Trial length in cm"""


    @classmethod
    def load(cls, file: PathLike,
             *,
             use_deconv: bool = False,
             trial_length: int | None = None,
             backend: LOAD_INPUT_BACKEND = 'pandas') -> Self:
        """
        Load data

        :param file: Filepath
        :param use_deconv: Whether deconvolved, otherwise use df/f
        :param trial_length: Trial length in cm
        :param backend: Backend for loading data, default is 'pandas'
        :return: ``PositionDecodeInput``
        """

        if backend == 'pandas':
            return cls._load_hdf(file, use_deconv, trial_length)
        elif backend == 'numpy':
            return cls._load_npy(file, use_deconv)
        else:
            raise ValueError(f'Unsupported backend: {backend}')

    # noinspection PyUnresolvedReferences
    @classmethod
    def _load_hdf(cls, file, use_deconv, trial_length) -> Self:
        """Use only in Joao's dataset"""
        dat = pd.read_hdf(file)
        act = dat.deconv if use_deconv else dat.df_f

        return cls(
            Path(file),
            act,
            dat.frametimes,
            dat.position_raw,
            dat.position_time,
            dat.laptimes,
            dat.lights_off_lap,
            dat.lights_off_time,
            trial_length
        )

    @classmethod
    def _load_npy(cls, file, use_deconv) -> Self:
        """Formal load"""
        dat = np.load(file)
        return cls(
            Path(file),
            dat['spks'] if use_deconv else dat['df_f'],
            dat['act_time'],
            dat['position'],
            dat['position_time'],
            dat['lap_time'],
            dat['lights_off_lap'],
            dat['lights_off_time'],
            dat['trial_length']
        )

    @property
    def n_neurons(self) -> int:
        """Number of neurons"""
        return self.activity.shape[0]

    @property
    def n_samples(self) -> int:
        """Number of activity samples"""
        return self.activity.shape[1]

    @property
    def activity_sampling_rate(self) -> float:
        """Approximate frame rate"""
        return np.median(1 / np.diff(self.act_time))

    @property
    def n_trials(self) -> int:
        return len(self.lap_time)

    @property
    def trial_index(self) -> np.ndarray:
        return np.arange(self.n_trials)

    def get_light_trange(self) -> tuple[int, int]:
        """Trial range of the light session (START, STOP)"""
        x = self.lap_time < self.light_off_time
        ret = self.trial_index[x]

        return int(ret[0]), int(ret[-1])

    def get_dark_trange(self, tol: int = 4) -> tuple[int, int]:
        """
        Trial range of the dark session (START, STOP)

        :param tol: tolerance (delay) buffer trials after lights-off
        :return: Trial range of the dark session (START, STOP)
        """
        x = self.lap_time >= self.light_off_time
        ret = self.trial_index[x]

        return int(ret[0]) + tol, int(ret[-1])

    @property
    def position_cache_file(self) -> Path:
        return self.filepath.with_name(self.filepath.stem + '_position_cache').with_suffix('.npz')

    def load_interp_position(self, sampling_rate: int = 100, force_compute: bool = False) -> CircularPosition:
        """
        Compute or load the ``CircularPosition``

        :param sampling_rate: Sampling rate for interpolation
        :param force_compute: Force recompute local cache
        :return: ``CircularPosition``
        """
        from neuralib.locomotion import interp_pos1d

        if not self.position_cache_file.exists() or force_compute:
            pos = interp_pos1d(self.position_time,
                               self.position,
                               norm_max_value=self.trial_length,
                               sampling_rate=sampling_rate)
            np.savez(self.position_cache_file, t=pos.t, p=pos.p, d=pos.d, v=pos.v,
                     trial_time_index=pos.trial_time_index)
            return pos
        else:
            pos = np.load(self.position_cache_file, allow_pickle=True)
            return CircularPosition(pos['t'], pos['p'], pos['d'], pos['v'], pos['trial_time_index'])
