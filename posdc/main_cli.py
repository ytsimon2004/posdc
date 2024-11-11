import random
from functools import cached_property
from pathlib import Path
from typing import Literal, Final, TypeVar

import numpy as np
from neuralib.argp import AbstractParser, argument, union_type, int_tuple_type
from neuralib.calimg.suite2p import normalize_signal
from neuralib.model.bayes_decoding import place_bayes
from neuralib.plot import plot_figure
from neuralib.typing import PathLike
from scipy.interpolate import interp1d

from posdc._plot import plot_decode_actual_position, plot_firing_rate, plot_decoding_err
from ._io import PositionDecodeInput
from ._ratemap import PositionRateMap
from ._trial import TrialSelection

__all__ = ['PositionDecodeOptions']

SESSION = Literal['light', 'dark']
TRAIN_TEST_SPLIT_METHOD = Literal['odd', 'even', 'random_split']

# odd, even | light, dark | x-fold cv | certain range of trials
CrossValidateType = TRAIN_TEST_SPLIT_METHOD | SESSION | int | tuple[int, int]


# TODO temporal bins adjustment

class PositionDecodeOptions(AbstractParser):
    DESCRIPTION = ''

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
        help='temporal bin size in second',
    )

    running_epoch: bool = argument(
        '--run',
        help='whether select only the running epoch',
    )

    # ================ #
    # Train-Test Split #
    # ================ #

    cross_validation: CrossValidateType = argument(
        '--CV', '--cv-type',
        type=union_type(int, str, tuple),
        default='odd',
        help='int type for nfold for model cross validation, otherwise, str type',
    )

    train_fraction: float = argument(
        '--train',
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

    #
    train_test_list: list[tuple[TrialSelection, TrialSelection]]
    trial_length: Final[int] = 150
    """in cm"""
    dat: PositionDecodeInput

    #
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
        match self.cross_validation:
            case int(cv) if cv > 0:
                self.number_iter = cv
            case str():
                self.number_iter = 1
            case _:
                raise RuntimeError(f'cv invalid: {self.cross_validation=}')

    def train_test_split(self, trial: TrialSelection) -> list[tuple[TrialSelection, TrialSelection]]:
        """Train test split based on ``cross_validation`` instance"""

        match self.cross_validation:
            case str():
                match self.cross_validation:
                    case 'light':
                        train_trial = self.dat.get_light_trange()
                        train_set = trial.select_range(train_trial)
                        test_set = train_set.invert()

                    case 'light-odd':
                        train_trial = self.dat.get_light_trange()
                        train_set = trial.select_odd_in_range(train_trial)
                        test_set = train_set.invert()

                        print(f'{train_set.selected_trials=}')
                        print(f'{test_set.selected_trials=}')

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

                    case 'random_split':
                        train_set, test_set = trial.select_fraction(self.train_fraction)
                    case _:
                        raise ValueError('')

                return [(train_set, test_set)]

            case int():
                match self.cross_validation:
                    case 0:  # no cv
                        return [(trial, trial)]
                    case _:
                        return trial.kfold_cv(int(self.cross_validation))

            case _:
                raise TypeError('')

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

        trial_index = np.zeros((index[1] - index[0]), dtype=int)  # (L')
        trial_index[train.selected_trials - index[0]] += 1  # train
        trial_index[test.selected_trials - index[0]] += 2  # test

        #
        fr = dat.activity[neuron_list]  # (N, T)
        fr = normalize_signal(fr)

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

        self.plot(time, predict_pos, actual_pos, fr, rate_map, self.dat.light_off_time)

        return pr, predict_pos

    def plot(self, time, pred_pos, actual_pos, fr, rate_map, light_off_time):
        with plot_figure(None, 3, 1, figsize=(15, 8)) as _ax:
            ax = _ax[0]
            plot_decode_actual_position(ax, time, pred_pos, actual_pos)

            ax = _ax[1]
            plot_firing_rate(ax, time, fr, rate_map)
            ax.sharex(_ax[0])

            ax = _ax[2]
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
            interp_pos = interp1d(position_time, position, bounds_error=False, fill_value=0)(act_time)
            interp_vel = interp1d(position_time, velocity, bounds_error=False, fill_value=0)(act_time)

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
