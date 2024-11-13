import random
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import seaborn as sns
from neuralib.argp import AbstractParser, argument
from neuralib.calimg.suite2p import normalize_signal
from neuralib.io import csv_header
from neuralib.model.bayes_decoding import place_bayes
from neuralib.plot import plot_figure
from neuralib.typing import PathLike
from neuralib.util.verbose import fprint
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

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
    'light-cv', 'dark-cv', 'all-cv'
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

    GROUP_DECODING = 'Decoding'

    trial_length: int = argument(
        '--length',
        metavar='VALUE',
        type=int,
        default=150,
        group=GROUP_DECODING,
        help='trial length in cm',
    )

    spatial_bin_size: float = argument(
        '--spatial-bin',
        metavar='VALUE',
        default=3,
        group=GROUP_DECODING,
        help='spatial bin size in cm',
    )

    temporal_bin_size: float | None = argument(
        '--temporal-bin',
        metavar='VALUE',
        default=None,
        group=GROUP_DECODING,
        help='temporal bin size in second, CURRENTLY NOT USE, DIRECTLY USE THE SAMPLING RATE OF THE DF/F',
    )

    running_epoch: bool = argument(
        '--run',
        group=GROUP_DECODING,
        help='whether select only the running epoch',
    )

    invalid_cache: bool = argument(
        '--invalid-cache',
        help='invalid all the cache, and recompute'
    )

    # ================ #
    # Train-Test Split #
    # ================ #

    GROUP_TRAINING_SET = 'Training Set'

    train_session: SESSION = argument(
        '--train',
        metavar='NAME',
        required=True,
        group=GROUP_TRAINING_SET,
        help='train the decoder in which behavioral session'
    )

    cross_validation: int | None = argument(
        '--cv',
        type=int,
        default=None,
        group=GROUP_TRAINING_SET,
        help='nfold for model cross validation (require for --train=light|dark-cv)',
    )

    no_shuffle: bool = argument(
        '--no-shuffle',
        group=GROUP_TRAINING_SET,
        help='whether without shuffle the data for non-repeated cv',
    )

    n_repeats: int | None = argument(
        '--repeats',
        type=int,
        default=None,
        group=GROUP_TRAINING_SET,
        help='run as repeat kfold cv, make number of results to `n_cv * n_repeats`. '
             '(optional for --train=light|dark-cv)'
    )

    train_fraction: float = argument(
        '--train-fraction',
        type=float,
        default=0.8,
        group=GROUP_TRAINING_SET,
        help='fraction of data for train set if `random_split` in cv, the rest will be utilized in test set.'
             '(require for --train=random-split)'
    )

    # ============== #
    # Random neurons #
    # ============== #

    GROUP_SHUFFLING = 'Shuffling'

    neuron_random: int | None = argument(
        '--random-neuron',
        metavar='NUMBER',
        type=int,
        default=None,
        group=GROUP_SHUFFLING,
        help='number of random neurons'
    )

    seed: int | None = argument(  # TODO this may not work properly. need review all random function calls.
        '--seed',
        type=int,
        metavar='VALUE',
        default=None,
        group=GROUP_SHUFFLING,
        help='seed for random number generator'
    )

    # ================= #
    # RasterMap Options #
    # ================= #

    GROUP_RASTER = 'Rastermap Plotting Options'

    rastermap_sort: bool = argument(
        '--rastermap',
        group=GROUP_RASTER,
        help='sort activity heatmap using rastermap'
    )

    rastermap_bin_size: int = argument(
        '--rastermap-bin',
        metavar='VALUE',
        default=20,
        group=GROUP_RASTER,
        help='bin size for number of total neurons',
    )

    # ============ #
    # Plot Options #
    # ============ #

    GROUP_PLOTTING = 'Plotting Options'

    ignore_foreach_plot: bool = argument(
        '--ignore-foreach',
        group=GROUP_PLOTTING,
        help='whether to ignore foreach cv plot, and only plot the summary result',
    )

    output_dir: Path = argument(
        '-o', '--output',
        group=GROUP_PLOTTING,
        help='output figures to a directory',
    )

    # runtime set
    train_test_list: list[tuple[TrialSelection, TrialSelection]] | None
    """List of train/test trial_selection"""

    dat: PositionDecodeInput | None
    """``PositionDecodeInput``"""

    number_iter: int | None
    """Number of iteration for running"""

    _current_train_test_index: int | None
    """Train test index of the iteration"""

    def run(self):
        self.dat = PositionDecodeInput.load_hdf(self.file, use_deconv=self.use_deconv, trial_length=self.trial_length)
        trial = TrialSelection(self.dat)
        self.train_test_list = self.train_test_split(trial)
        self.number_iter = self.get_number_iter()
        assert self.number_iter == len(self.train_test_list)

        rate_map = self.get_ratemap()

        if self.csv_output.exists():
            self.csv_output.unlink()
            fprint(f'Auto delete existed csv due to the append mode: {self.csv_output}', vtype='io')

        for i in range(self.number_iter):
            self._current_train_test_index = i
            fprint(f'Validate iteration: {i}')
            self.run_decode(rate_map, trial, self.neuron_list)

        self.plot_decode_cv()

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
    def csv_output(self) -> Path:
        file = self.dat.filepath
        return file.with_name(file.stem + '_test_result').with_suffix('.csv')

    @property
    def current_train_test(self) -> tuple[TrialSelection, TrialSelection]:
        sz = len(self.train_test_list)
        return self.train_test_list[self._current_train_test_index % sz]

    def get_number_iter(self) -> int:
        match self.cross_validation, self.n_repeats:
            case (int(cv), None) if cv > 0:
                return cv
            case (int(cv), int()) if cv > 0:
                return cv * self.n_repeats
            case _:
                return 1

    def train_test_split(self, trial: TrialSelection) -> list[tuple[TrialSelection, TrialSelection]]:
        match self.train_session, self.cross_validation, self.n_repeats:

            # train decoder on all the even trials and test on all the odd trials
            case ('even', _, _):
                train = trial.select_even()
                test = train.invert()

            # train decoder on all the odd trials, and test on all the even trials
            case ('odd', _, _):
                train = trial.select_odd()
                test = train.invert()

            # train decoder on all the light session, and test on dark session
            case ('light', _, _):
                train_trial = self.dat.get_light_trange()
                train = trial.select_range(train_trial)
                test = train.invert()

            # train decoder on the light-odd, and test on all the rest
            case ('light-odd', _, _):
                train_trial = self.dat.get_light_trange()
                train = trial.select_odd_in_range(train_trial)
                test = train.invert()  # TODO maybe to ``diffall()``?

            # train decoder on the light-even, and test on all the rest
            case ('light-even', _, _):
                train_trial = self.dat.get_light_trange()
                train = trial.select_even_in_range(train_trial)
                test = train.invert()

            # train decoder on the light session with k-fold (or repeated) validation
            case ('light-cv', int(), int() | None):
                train_trial = self.dat.get_light_trange()
                return trial.select_range(train_trial).kfold_cv(
                    self.cross_validation,
                    not self.no_shuffle,
                    self.n_repeats,
                    self.seed,
                    test_across_session=True
                )

            # train decoder on all the dark session, and test on light session
            case ('dark', _, _):
                train_trial = self.dat.get_dark_trange()
                train = trial.select_range(train_trial)
                test = train.invert()

            # train decoder on the dark-odd, and test on all the rest
            case ('dark-odd', _, _):
                train_trial = self.dat.get_dark_trange()
                train = trial.select_odd_in_range(train_trial)
                test = train.invert()  # TODO maybe to ``diffall()``?

            # train decoder on the dark-even, and test on all the rest
            case ('dark-even', _, _):
                train_trial = self.dat.get_dark_trange()
                train = trial.select_even_in_range(train_trial)
                test = train.invert()

            # train decoder on the dark session with k-fold (or repeated) validation
            case ('dark-cv', int(), int() | None):
                train_trial = self.dat.get_dark_trange()
                return trial.select_range(train_trial).kfold_cv(
                    self.cross_validation,
                    not self.no_shuffle,
                    self.n_repeats,
                    self.seed,
                    test_across_session=True
                )

            # train decoder on all with k-fold (or repeated) validation
            case ('all-cv', int(), int() | None):
                return trial.kfold_cv(
                    self.cross_validation,
                    not self.no_shuffle,
                    self.n_repeats,
                    state=self.seed,
                    test_across_session=False
                )

            # train decoder by selecting fraction of the trials, and test on the rest
            case ('random-split', _, _):
                train, test = trial.select_fraction(self.train_fraction)

            # same train/test
            case _:
                fprint('unsupported format train-test split, then use the same train/test', vtype='warning')
                return [(trial, trial)]

        return [(train, test)]

    def get_ratemap(self) -> np.ndarray:
        n_bins = int(self.trial_length / self.spatial_bin_size)
        pos_ratemap = PositionRateMap(self.dat, n_bins=n_bins, sig_norm=True, force_compute=self.invalid_cache)
        return pos_ratemap.load_binned_data(running_epoch=self.running_epoch, force_compute=self.invalid_cache)

    def run_decode(self, rate_map: np.ndarray,
                   trial: TrialSelection,
                   neuron_list: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the decoding analysis.

        :param rate_map: `Array[float, [N, L, B]]`
        :param trial: ``TrialSelection``
        :param neuron_list: Neuronal bool mask. `Array[bool , N]`
        :return:
        - pr: matrix of posterior probabilities. `Array[float, [T, B]]`.
        - predict_pos: Predicted position. `Array[float, T]`
        """

        dat = self.dat
        index = trial.session_range

        train, test = self.current_train_test

        # rate map
        rate_map = train.take_along_trial_axis(rate_map, 1)
        rate_map = np.nanmean(rate_map, axis=1)[neuron_list]

        trial_index = np.zeros((index[1] - index[0]) + 1, dtype=int)
        trial_index[train.selected_trials - index[0]] += 1  # train
        trial_index[test.selected_trials - index[0]] += 2  # test

        # fr
        fr = dat.activity[neuron_list]  # (N, T)
        fr = fr_raw = normalize_signal(fr)

        if self.rastermap_sort:
            from ._rastermap import run_rastermap
            fr_raw = run_rastermap(fr_raw, self.rastermap_bin_size).super_neurons
            ylabel = '#super_neurons'
        else:
            sort_idx = self._sort_position(rate_map)
            fr_raw = fr_raw[sort_idx]
            ylabel = '#neurons'

        # position
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
        rate_map = rate_map.T  # (B, N)
        pr = self.load_bayes_posterior(fr, rate_map)
        predict_pos = np.argmax(pr, axis=1) * self.spatial_bin_size

        if not self.ignore_foreach_plot:
            self.plot_decode_foreach(time, predict_pos, actual_pos, fr_raw.T, self.dat.light_off_time, ylabel)

        self.log_output_csv(test, time, predict_pos, actual_pos)

        return pr, predict_pos

    def log_output_csv(self,
                       test: TrialSelection,
                       time: np.ndarray,
                       pred_pos: np.ndarray,
                       actual_pos: np.ndarray,
                       agg_func: Literal['median', 'mean'] = 'median'):
        headers = ['n_cv', 'session', 'n_trials', f'decode_err']

        with csv_header(self.csv_output, headers, append=True) as csv:
            mask = time < self.dat.light_off_time  # TODO if add tol 4 trials?
            err = self._wrap_diff(pred_pos, actual_pos, self.dat.trial_length)

            func = getattr(np, agg_func)
            for session in ('light', 'dark'):
                if session == 'light':
                    error = func(err[mask])
                    n_trials = np.count_nonzero(test.selected_trials < self.dat.light_off_lap)
                else:
                    error = func(err[~mask])
                    n_trials = np.count_nonzero(test.selected_trials >= self.dat.light_off_lap)

                csv(self._current_train_test_index, session, n_trials, error)

    def plot_decode_foreach(self, time, pred_pos, actual_pos, fr_raw, light_off_time, ylabel):
        """
        Plot decode results foreach iteration

        :param time: `Array[float, T]`
        :param pred_pos: `Array[float, T]`
        :param actual_pos: `Array[float, T]`
        :param fr_raw: Raw firing. `Array[float, [Traw, N | N']]`
        :param light_off_time: Time of light off in sec
        :param ylabel: ylabel for the firing activity
        """
        # TODO take self.output_dir and generate figure filenames
        with plot_figure(None, 3, 1, figsize=(15, 8), height_ratios=(1, 2, 1), sharex=True) as ax:
            plot_decode_actual_position(ax[0], time, pred_pos, actual_pos)

            plot_firing_rate(ax[1], self.dat.act_time, fr_raw, ylabel=ylabel)

            err = self._wrap_diff(pred_pos, actual_pos, self.dat.trial_length)
            plot_decoding_err(ax[2], time, err, light_off_time)

    def load_devode_cv(self) -> pl.DataFrame:
        return pl.read_csv(self.csv_output)

    def plot_decode_cv(self):
        """Plot decode summary cross-validation testset results"""
        df = self.load_devode_cv()
        print(f'cv dataframe: {df}')
        with plot_figure(None, figsize=(3, 8)) as ax:
            sns.boxplot(df, x='session', y='decode_err', ax=ax)
            sns.stripplot(df, x='session', y='decode_err', color='black', jitter=False, alpha=0.7, ax=ax)
            ax.set(ylabel='decode_error(cm)')

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
    def _wrap_diff(x: np.ndarray,
                   y: np.ndarray,
                   upper_bound: int = 150) -> np.ndarray:
        """
        Calculate the distance between two points in the wrapped environment

        :param x: Position x. `Array[float, T]`
        :param y: Position y. `Array[float, T]`
        :param upper_bound: Upper bound of the distance in cm
        :return: Distance between x and y in cm. `Array[float, T]`
        """

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

        :param position_time: Position time. `Array[float, P]`
        :param position: Position value. `Array[float, P]`
        :param velocity: Velocity value in cm/s. `Array[float, P]`
        :param fr: Neural activity. `Array[float, [N, T]]`
        :param act_time: Activity time. `Array[float, T]`
        :param position_down_sampling: If True, interpolate position to neural activity shape, otherwise, vice versa
        :return: After running epoch masking and interpolation
            - fr: Neural activity.`Array[float, [N, T']]`.
            - time: Neural activity time. `Array[float, T]`.
            - position: Animal's position. `Array[float, T]`.
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

    @staticmethod
    def _sort_position(data: np.ndarray) -> np.ndarray:
        """sort neurons based on maximal activity along the track position"""
        m_filter = gaussian_filter1d(data, 3, axis=1)
        m_argmax = np.argmax(m_filter, axis=1)
        return np.argsort(m_argmax)


def main():
    PositionDecodeOptions().main()


if __name__ == '__main__':
    main()
