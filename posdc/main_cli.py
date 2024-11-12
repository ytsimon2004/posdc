import random
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
from neuralib.argp import AbstractParser, argument
from neuralib.calimg.suite2p import normalize_signal
from neuralib.model.bayes_decoding import place_bayes
from neuralib.plot import plot_figure, ax_merge
from neuralib.typing import PathLike
from scipy.interpolate import interp1d

from ._io import PositionDecodeInput
from ._plot import *
from ._ratemap import PositionRateMap
from ._trial import TrialSelection

__all__ = ['PositionDecodeOptions']

SESSION = Literal[
    'odd', 'even',
    'light', 'light-odd', 'light-even',
    'dark', 'dark-odd', 'dark-even',
    'random-split',
    'light-cv', 'dark-cv'  # Train model on a specific session, and test on the rest
]


class PositionDecodeOptions(AbstractParser):
    DESCRIPTION = "Bayes decoding of animal's position in an linear environment"

    file: PathLike = argument(
        '-F', '--file',
        help='hdf data file path',
    )

    use_deconv: bool = argument(
        '--deconv',
        help='use deconv activity, otherwise, df_f',
    )

    # ================= #
    # Decoder Parameter #
    # ================= #

    trial_length: int = argument(
        '--length',
        metavar='VALUE',
        type=int,
        default=150,
        help='trial length in cm',
    )

    spatial_bin_size: float = argument(
        '--spatial-bin',
        metavar='VALUE',
        default=3,
        help='spatial bin size in cm',
    )

    temporal_bin_size: float | None = argument(
        '--temporal-bin',
        metavar='VALUE',
        default=None,
        help='temporal bin size in second, CURRENTLY NOT USE, DIRECTLY USE THE SAMPLING RATE OF THE DF/F',
    )

    running_epoch: bool = argument(
        '--run',
        help='whether select only the running epoch',
    )

    # ================ #
    # Train-Test Split #
    # ================ #

    train_session: SESSION = argument(
        '--train-session',
        metavar='NAME',
        default='light',
        help='train the decoder in which behavioral session'
    )

    cross_validation: int | None = argument(
        '--cv',
        type=int,
        default=None,
        help='nfold for model cross validation',
    )

    no_shuffle: bool = argument(
        '--no-shuffle',
        help='whether without shuffle the data for non-repeated cv',
    )

    n_repeats: int | None = argument(
        '--repeats',
        type=int,
        default=None,
        help='run as repeat kfold cv, make number of results to `n_cv * n_repeats`'
    )

    train_fraction: float = argument(
        '--train',
        type=float,
        default=0.8,
        help='fraction of data for train set if `random_split` in cv, the rest will be utilized in test set'
    )

    # ============== #
    # Random neurons #
    # ============== #

    neuron_random: int | None = argument(
        '--random',
        metavar='NUMBER',
        type=int,
        default=None,
        help='number of random neurons'
    )

    seed: int | None = argument(
        '--seed',
        type=int,
        metavar='VALUE',
        default=None,
        help='seed for random number generator'
    )

    # runtime set
    train_test_list: list[tuple[TrialSelection, TrialSelection]] | None
    dat: PositionDecodeInput | None
    number_iter: int | None
    _current_train_test_index: int | None

    def run(self):
        self.dat = PositionDecodeInput.load_hdf(self.file, use_deconv=self.use_deconv, trial_length=self.trial_length)
        trial = TrialSelection(self.dat)
        self.train_test_list = self.train_test_split(trial)
        self.set_number_iter()

        for i in range(self.number_iter):
            self._current_train_test_index = i
            self.run_decode(trial, self.neuron_list)

    @cached_property
    def neuron_list(self) -> np.ndarray:
        n_neurons = self.dat.n_neurons
        ret = np.full(n_neurons, 0, dtype=bool)

        match self.neuron_random:
            case int(n) if n < n_neurons:
                n = self.neuron_random
            case int(n) if n >= n_neurons:
                n = n_neurons
            case None:
                n = n_neurons
            case _:
                raise ValueError('')

        if self.seed is not None:
            random.seed(self.seed)
        ret[random.sample(range(n_neurons), n)] = 1

        return np.nonzero(ret)[0]

    @property
    def bayes_posterior_cache(self) -> Path:
        file = self.dat.filepath
        return file.with_name(file.stem + '_bayes_posterior_cache').with_suffix('.npy')

    @property
    def train_test(self) -> tuple[TrialSelection, TrialSelection]:
        sz = len(self.train_test_list)
        return self.train_test_list[self._current_train_test_index % sz]

    def set_number_iter(self):
        match self.cross_validation, self.n_repeats:
            case (int(cv), None) if cv > 0:
                self.number_iter = cv
            case (int(cv), int()) if cv > 0:
                self.number_iter = cv * self.n_repeats
            case (str(), _):
                self.number_iter = 1
            case _:
                raise RuntimeError(f'cv invalid: {self.cross_validation=}')

    # noinspection PyTypeChecker
    def train_test_split(self, trial: TrialSelection) -> list[tuple[TrialSelection, TrialSelection]]:
        """Train test split based on ``cross_validation`` instance"""

        if self.train_session is not None:
            match self.train_session:
                case 'light':
                    train_trial = self.dat.get_light_trange()
                    train_set = trial.select_range(train_trial)
                    test_set = train_set.invert()
                case 'light-odd':
                    train_trial = self.dat.get_light_trange()
                    train_set = trial.select_odd_in_range(train_trial)
                    test_set = train_set.invert()
                case 'light-even':
                    train_trial = self.dat.get_light_trange()
                    train_set = trial.select_even_in_range(train_trial)
                    test_set = train_set.invert()
                case 'dark':
                    train_trial = self.dat.get_dark_trange()
                    train_set = trial.select_range(train_trial)
                    test_set = train_set.invert()
                case 'even':
                    train_set = trial.select_even()
                    test_set = train_set.invert()
                case 'odd':
                    train_set = trial.select_odd()
                    test_set = train_set.invert()
                case 'random-split':
                    train_set, test_set = trial.select_fraction(self.train_fraction)
                case 'light-cv':
                    train_trial = self.dat.get_light_trange()
                    return trial.kfold_cv_in_range(train_trial, self.cross_validation, not self.no_shuffle)
                case 'dark-cv':
                    train_trial = self.dat.get_dark_trange()
                    return trial.kfold_cv_in_range(train_trial, self.cross_validation, not self.no_shuffle)
                case _:
                    raise ValueError('')

            return [(train_set, test_set)]

        elif isinstance(self.cross_validation, int) and isinstance(self.n_repeats, int):
            match self.cross_validation, self.n_repeats:
                case (0, None):  # no cv
                    return [(trial, trial)]
                case (i, None) if i > 0:
                    return trial.kfold_cv(self.cross_validation, self.no_shuffle)
                case (i, int()) if i > 0:
                    return trial.repeat_kfold_cv(self.cross_validation, n_repeats=self.n_repeats, state=self.seed)
                case _:
                    raise ValueError('')
        else:
            raise RuntimeError('unsupported format train-test split')

    def run_decode(self, trial: TrialSelection, neuron_list: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        dat = self.dat
        index = trial.session_range

        train, test = self.train_test

        # rate map
        n_bins = int(self.trial_length / self.spatial_bin_size)
        rate_map = (
            PositionRateMap(dat, n_bins=n_bins, sig_norm=True)
            .load_binned_data(running_epoch=self.running_epoch)
        )  # (N, L, X)
        rate_map = train.masking_trial_matrix(rate_map, 1)  # (N, L', X)
        rate_map = np.nanmean(rate_map, axis=1)[neuron_list]  # (N, X)

        trial_index = np.zeros((index[1] - index[0]) + 1, dtype=int)  # (L')
        trial_index[train.selected_trials - index[0]] += 1  # train
        trial_index[test.selected_trials - index[0]] += 2  # test

        #
        fr = dat.activity[neuron_list]  # (N, T)
        fr = fr_raw = normalize_signal(fr)

        pos = dat.load_interp_position()
        if self.running_epoch:
            fr, time, position = self._running_epoch_masking(pos.t, pos.p, pos.v, fr, dat.act_time)
        else:
            time = dat.act_time
            position = pos.p
            # TODO interp
            raise NotImplementedError('')

        # actual (test set)
        t_mask = test.masking_time(time)
        fr = fr[:, t_mask]
        time = time[t_mask]
        actual_pos = position[t_mask]

        # predict
        fr = fr.T  # (T, N)
        rate_map = rate_map.T  # (X, N)
        pr = self.load_bayes_posterior(fr, rate_map)
        predict_pos = np.argmax(pr, axis=1) * self.spatial_bin_size

        self.plot_decode_result(time, predict_pos, actual_pos, fr_raw.T, rate_map, self.dat.light_off_time)

        return pr, predict_pos

    def plot_decode_result(self, time, pred_pos, actual_pos, fr_raw, rate_map, light_off_time):
        """

        :param time: `Array[float, T]`
        :param pred_pos: `Array[float, T]`
        :param actual_pos: `Array[float, T]`
        :param fr_raw: Raw firing. `Array[float, [Traw, N]]`
        :param rate_map: `Array[float, [T, N]]`
        :param light_off_time: Time of light off in sec
        """
        with plot_figure(None, 4, 1, figsize=(15, 8)) as _ax:
            ax = _ax[0]
            plot_decode_actual_position(ax, time, pred_pos, actual_pos)

            ax = ax_merge(_ax)[1:3]
            plot_firing_rate(ax, time, fr_raw, rate_map)
            ax.sharex(_ax[0])

            ax = _ax[3]
            err = self._calc_wrap_distance(pred_pos, actual_pos, self.dat.trial_length)
            plot_decoding_err(ax, time, err, light_off_time)
            ax.sharex(_ax[0])

    def load_bayes_posterior(self, fr: np.ndarray,
                             rate_map: np.ndarray,
                             force_compute: bool = True) -> np.ndarray:
        if not self.bayes_posterior_cache.exists() or force_compute:
            pr = place_bayes(fr, rate_map, self.spatial_bin_size)
            np.save(self.bayes_posterior_cache, pr)
        else:
            pr = np.load(self.bayes_posterior_cache)
        return pr

    @staticmethod
    def _calc_wrap_distance(x: np.ndarray,
                            y: np.ndarray,
                            upper_bound: int = 150) -> np.ndarray:
        """calculate the distance between two points in the wrapped environment"""

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError('')

        points = np.sort([*zip(x, y)])
        distances = points[:, 1] - points[:, 0]
        distances_wrap = upper_bound - distances

        return np.minimum(distances, distances_wrap)

    @staticmethod
    def _running_epoch_masking(position_time: np.ndarray,
                               position: np.ndarray,
                               velocity: np.ndarray,
                               fr: np.ndarray,
                               act_time: np.ndarray,
                               position_down_sampling: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """

        :param position_time:
        :param position:
        :param velocity:
        :param fr:
        :param act_time:
        :param position_down_sampling: If True, interpolate position to neural activity shape, otherwise, vice versa
        :return:
        """
        from neuralib.locomotion import running_mask1d

        if position_down_sampling:
            interp_pos = interp1d(position_time, position, bounds_error=False, fill_value='extrapolate')(act_time)
            interp_vel = interp1d(position_time, velocity, bounds_error=False, fill_value='extrapolate')(act_time)

            x = running_mask1d(act_time, interp_vel)
            fr = fr[:, x]
            position = interp_pos[x]
            time = act_time[x]

        else:
            x = running_mask1d(position_time, velocity)
            time = position_time[x]
            fr = interp1d(act_time, fr, axis=fr.ndim - 1, bounds_error=False, fill_value=0)(time)
            position = position[x]

        return fr, time, position


if __name__ == '__main__':
    PositionDecodeOptions().main()
