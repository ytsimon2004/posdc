import random
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import scipy
from argclz import AbstractParser, argument, float_tuple_type
from neuralib.imaging.suite2p import normalize_signal
from neuralib.io import csv_header
from neuralib.model.bayes_decoding import place_bayes
from neuralib.plot import plot_figure, violin_boxplot, ax_merge
from neuralib.typing import PathLike
from neuralib.util.utils import ensure_dir
from neuralib.util.verbose import fprint, print_save
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from tqdm import trange

from ._io import PositionDecodeInput, LOAD_INPUT_BACKEND
from ._plot import *
from ._ratemap import PositionRateMap, PositionBinnedSig
from ._trial import TrialSelection

__all__ = ['PositionDecodeOptions']

TRAIN_OPTIONS = Literal[
    'odd', 'even',
    'light', 'light-odd', 'light-even',
    'dark', 'dark-odd', 'dark-even',
    'random-split',
    'light-cv', 'dark-cv', 'all-cv'
]


class PositionDecodeOptions(AbstractParser):
    DESCRIPTION = "Bayes decoding of animal's position in an linear environment"

    # ============= #
    # IO Input Load #
    # ============= #

    file: PathLike = argument(
        '-F', '--file',
        help='hdf data file path',
    )

    use_deconv: bool = argument(
        '--deconv',
        help='use deconv activity, otherwise, df_f',
    )

    load_backend: LOAD_INPUT_BACKEND = argument(
        '--load-backend',
        default='pandas',
        help='backend for loading data, default is `pandas`'
    )

    # ================= #
    # Decoder Parameter #
    # ================= #

    GROUP_DECODING = 'Decoding Option'

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

    no_position_down_sampling: bool = argument(
        '--no-position-ds',
        group=GROUP_DECODING,
        help='not interpolate position to neural activity shape, then interpolate neural activity to position',
    )

    invalid_cache: bool = argument(
        '--invalid-cache',
        group=GROUP_DECODING,
        help='invalid all the cache, and recompute'
    )

    # ================ #
    # Train-Test Split #
    # ================ #

    GROUP_TRAIN_TEST = 'Train/Test Option'

    train_option: TRAIN_OPTIONS = argument(
        '--train',
        metavar='NAME',
        required=True,
        group=GROUP_TRAIN_TEST,
        help='train the decoder in which behavioral session'
    )

    cross_validation: int | None = argument(
        '--cv',
        type=int,
        default=None,
        validator=lambda it: it is None or 0 < it <= 10,
        group=GROUP_TRAIN_TEST,
        help='N-fold for model cross validation',
    )

    no_shuffle: bool = argument(
        '--no-shuffle',
        group=GROUP_TRAIN_TEST,
        help='whether without shuffle the data for non-repeated cv',
    )

    n_repeats: int | None = argument(
        '--repeats',
        type=int,
        default=None,
        group=GROUP_TRAIN_TEST,
        help='run as repeated kfold cv, make number of results to `n_cv * n_repeats`'
    )

    train_fraction: float = argument(
        '--train-fraction',
        type=float,
        default=0.8,
        validator=lambda it: 0 < it < 1,
        group=GROUP_TRAIN_TEST,
        help='fraction of data for train set if `random_split` in cv, the rest will be utilized in test set.'
             '(require for --train=random-split)'
    )

    # ============= #
    # Randomization #
    # ============= #

    GROUP_RANDOM = 'Random Option'

    neuron_random: int | None = argument(
        '--random-neuron',
        metavar='NUMBER',
        type=int,
        default=None,
        group=GROUP_RANDOM,
        help='number of random neurons. If None, then use all neurons'
    )

    seed: int | None = argument(
        '--seed',
        type=int,
        metavar='VALUE',
        default=None,
        group=GROUP_RANDOM,
        help='seed for random number generator'
    )

    # ================= #
    # RasterMap Options #
    # ================= #

    GROUP_RASTERMAP = 'Rastermap sorting Options'

    rastermap_sort: bool = argument(
        '--rastermap',
        group=GROUP_RASTERMAP,
        help='sort activity heatmap using rastermap'
    )

    rastermap_bin_size: int = argument(
        '--rastermap-bin',
        metavar='VALUE',
        default=15,
        group=GROUP_RASTERMAP,
        help='bin size for number of total neurons',
    )

    # ============ #
    # Plot Options #
    # ============ #

    GROUP_PLOTTING = 'Plotting Options'

    smooth_fr: bool = argument(
        '--smooth',
        group=GROUP_PLOTTING,
        help='Do smooth of raw firing (N, T) heatmap'
    )

    time_mask: tuple[float, float] | None = argument(
        '--time-mask',
        type=float_tuple_type,
        default=None,
        validator=lambda it: it is None or len(it) == 2,
        group=GROUP_PLOTTING,
        help='time mask for plotting. (START, END) in seconds',
    )

    perc_norm: tuple[float, float] | None = argument(
        '--perc-norm',
        type=float_tuple_type,
        default=None,
        validator=lambda it: it is None or len(it) == 2,
        group=GROUP_PLOTTING,
        help='Lower and upper percentile bounds for the fr_raw, for the visualization of population activity'
    )

    with_train_position: bool = argument(
        '--with-train',
        group=GROUP_PLOTTING,
        help='plot the train set trials'
    )

    ignore_foreach_plot: bool = argument(
        '--ignore-foreach',
        group=GROUP_PLOTTING,
        help='whether to ignore foreach cv plot, and only plot the summary result',
    )

    no_interactive: bool = argument(
        '--no-qt',
        group=GROUP_PLOTTING,
        help='Do not render any interactive qt plot'
    )

    output_dir: Path | None = argument(
        '-O', '--output',
        metavar='PATH',
        type=Path,
        group=GROUP_PLOTTING,
        default=None,
        help='output figures to a directory',
    )

    # =========== #
    # Runtime set #
    # =========== #

    train_test_list: list[tuple[TrialSelection, TrialSelection]] | None
    """List of train/test trial_selection"""

    dat: PositionDecodeInput | None
    """``PositionDecodeInput``"""

    number_iter: int | None
    """Number of iteration for running"""

    _current_train_test_index: int | None
    """Train test index of the iteration"""

    _current_train_trials: np.ndarray | None
    """Train trials of the iteration"""

    _current_test_trials: np.ndarray | None
    """Test trials of the iteration"""

    __rastermap_sn: np.ndarray | None = None
    """Rastermap super neuron cache"""

    def run(self):
        # set attrs
        self.dat = PositionDecodeInput.load(self.file,
                                            use_deconv=self.use_deconv,
                                            trial_length=self.trial_length,
                                            backend=self.load_backend)
        trial = TrialSelection(self.dat)
        self.train_test_list = self.train_test_split(trial)

        self.number_iter = self.get_number_iter()
        assert self.number_iter == len(self.train_test_list)
        rate_map = self.get_ratemap()

        # io
        if self._csv_output.exists():
            self._csv_output.unlink()
            fprint(f'Auto delete existed csv due to the append mode: {self._csv_output}', vtype='io')

        if self.output_dir is not None:
            ensure_dir(self.output_dir)

        # main
        for i in trange(self.number_iter, ncols=80, unit='cv', desc='number of cross validate'):
            self._current_train_test_index = i
            self.run_decode(rate_map, trial, self.neuron_list)

        self.csv_to_parquet()

        if not self.no_interactive:
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
    def _csv_output(self) -> Path:
        """for tmp append write"""
        file = self.dat.filepath
        return file.with_name(file.stem + '_test_result').with_suffix('.csv')

    @property
    def parquet_output(self) -> Path:
        file = self.dat.filepath
        return file.with_name(file.stem + '_test_result').with_suffix('.parquet')

    @property
    def current_train_test(self) -> tuple[TrialSelection, TrialSelection]:
        sz = len(self.train_test_list)
        return self.train_test_list[self._current_train_test_index % sz]

    def get_test_trials(self, session: Literal['light', 'dark', 'light_end']) -> np.ndarray:
        test_trials = self._current_test_trials[:-1]  # exclude last incomplete

        match session:
            case 'light':
                return test_trials[test_trials < self.dat.light_off_lap]
            case 'dark':
                mx = np.logical_and(self.dat.light_off_lap < test_trials, test_trials < self.dat.light_end_lap)
                return test_trials[mx]  # with tol
            case 'light_end':
                return test_trials[test_trials >= self.dat.light_end_lap]
            case _:
                raise ValueError(f'unknown session: {session}')

    def get_number_iter(self) -> int:
        match self.cross_validation, self.n_repeats:
            case (int(cv), None) if cv > 0:
                return cv
            case (int(cv), int()) if cv > 0:
                return cv * self.n_repeats
            case _:
                return 1

    def train_test_split(self, trial: TrialSelection) -> list[tuple[TrialSelection, TrialSelection]]:
        match self.train_option, self.cross_validation, self.n_repeats:

            # train decoder on all the even trials and test on all the odd trials
            case ('even', _, _):
                train = trial.select_even()
                test = train.diff_all()

            # train decoder on all the odd trials, and test on all the even trials
            case ('odd', _, _):
                train = trial.select_odd()
                test = train.diff_all()

            # train decoder on all the light session, and test on dark session
            case ('light', _, _):
                train_trial = self.dat.get_light_trange()
                train = trial.select_range(train_trial)
                test = train.diff_all()

            # train decoder on the light-odd, and test on all the rest
            case ('light-odd', _, _):
                train_trial = self.dat.get_light_trange()
                train = trial.select_odd_in_range(train_trial)
                test = train.diff_all()

            # train decoder on the light-even, and test on all the rest
            case ('light-even', _, _):
                train_trial = self.dat.get_light_trange()
                train = trial.select_even_in_range(train_trial)
                test = train.diff_all()

            # train decoder on the light session with k-fold (or repeated) validation
            case ('light-cv', int(), int() | None):
                train_trial = self.dat.get_light_trange()
                return trial.select_range(train_trial).kfold_cv(
                    self.cross_validation,
                    not self.no_shuffle,
                    self.n_repeats,
                    self.seed,
                    test_from_all=True
                )

            # train decoder on all the dark session, and test on light session
            case ('dark', _, _):
                train_trial = self.dat.get_dark_trange()
                train = trial.select_range(train_trial)
                test = train.diff_all()

            # train decoder on the dark-odd, and test on all the rest
            case ('dark-odd', _, _):
                train_trial = self.dat.get_dark_trange()
                train = trial.select_odd_in_range(train_trial)
                test = train.diff_all()

            # train decoder on the dark-even, and test on all the rest
            case ('dark-even', _, _):
                train_trial = self.dat.get_dark_trange()
                train = trial.select_even_in_range(train_trial)
                test = train.diff_all()

            # train decoder on the dark session with k-fold (or repeated) validation
            case ('dark-cv', int(), int() | None):
                train_trial = self.dat.get_dark_trange()
                return trial.select_range(train_trial).kfold_cv(
                    self.cross_validation,
                    not self.no_shuffle,
                    self.n_repeats,
                    self.seed,
                    test_from_all=True
                )

            # train decoder on all with k-fold (or repeated) validation
            case ('all-cv', int(), int() | None):
                return trial.kfold_cv(
                    self.cross_validation,
                    not self.no_shuffle,
                    self.n_repeats,
                    state=self.seed,
                    test_from_all=False
                )

            # train decoder by selecting fraction of the trials, and test on the rest
            case ('random-split', _, _):
                train, test = trial.select_fraction(self.train_fraction, seed=self.seed)

            # same train/test
            case _:
                fprint('unsupported format train-test split, then use the same train/test', vtype='warning')
                return [(trial, trial)]

        return [(train, test)]

    def get_ratemap(self) -> np.ndarray:
        n_bins = int(self.trial_length / self.spatial_bin_size)
        pos_ratemap = PositionRateMap(self.dat, n_bins=n_bins, sig_norm=True, force_compute=self.invalid_cache)
        return pos_ratemap.load_binned_data(running_epoch=self.running_epoch, force_compute=self.invalid_cache)

    def get_rastermap_sn(self, fr_raw: np.ndarray) -> np.ndarray:
        """Get rastermap superneurons and cache instance"""
        if self.__rastermap_sn is None:
            from posdc._rastermap import run_rastermap
            ret = run_rastermap(fr_raw, self.rastermap_bin_size).super_neurons
            self.__rastermap_sn = ret

        return self.__rastermap_sn

    def run_decode(self, rate_map: np.ndarray,
                   trial: TrialSelection,
                   neuron_list: np.ndarray) -> None:
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

        trial_index = np.zeros((index[1] - index[0]) + 1, dtype=int)  # 0-based
        trial_index[train.selected_trials - index[0]] += 1  # train
        trial_index[test.selected_trials - index[0]] += 2  # test

        self._current_train_trials = np.nonzero(trial_index == 1)[0]
        self._current_test_trials = np.nonzero(trial_index == 2)[0]

        # fr
        fr = dat.activity[neuron_list]  # (N, T)
        fr = fr_raw = normalize_signal(fr)

        # Compute lower and upper percentile bounds
        if self.perc_norm is not None:
            lower, upper = self.perc_norm
            lp = np.percentile(fr_raw, lower, axis=1, keepdims=True)
            up = np.percentile(fr_raw, upper, axis=1, keepdims=True)

            fr_raw = (fr_raw - lp) / (up - lp)
            fr_raw = np.clip(fr_raw, 0, 1)

        # Sort
        if self.rastermap_sort:
            from ._rastermap import run_rastermap
            fr_raw = self.get_rastermap_sn(fr_raw)
            ylabel = '#super_neurons'
        else:
            sort_idx = self._sort_position(rate_map)
            fr_raw = fr_raw[sort_idx]
            ylabel = '#neurons'

        # position
        pos = dat.load_interp_position()
        if self.running_epoch:
            fr, time, position = self._running_epoch_masking(
                pos.t, pos.p, pos.v, fr, dat.act_time,
                position_down_sampling=not self.no_position_down_sampling
            )
        else:
            fr, time, position = self._all_epoch_masking(
                pos.t, pos.p, fr, dat.act_time,
                position_down_sampling=not self.no_position_down_sampling
            )

        # actual (train set)
        if self.with_train_position:
            t_mask = train.masking_time(time)
            t_seg = train.segment_time(time)
            train_pos = position[t_mask]
            train_time = time[t_mask]
            train_seg_time = time[t_seg]
        else:
            train_pos = None
            train_time = None
            train_seg_time = None

        # actual (test set)
        t_mask = test.masking_time(time)
        fr = fr[:, t_mask]
        time = time[t_mask]
        actual_pos = position[t_mask]

        # predict
        fr = fr.T  # (T, N)
        rate_map = rate_map.T  # (B, N)
        pr = place_bayes(fr, rate_map, self.spatial_bin_size)
        predict_pos = np.argmax(pr, axis=1) * self.spatial_bin_size

        #
        binned_err_light, binned_err_dark, binned_err_light_end = self.calc_position_binned_error(
            time, predict_pos, actual_pos
        )

        if not self.ignore_foreach_plot and not self.no_interactive:
            self.plot_decode_foreach(
                time,
                predict_pos,
                actual_pos,
                fr_raw.T,
                binned_err_light,
                binned_err_dark,
                binned_err_light_end,
                dat.light_off_time,
                ylabel,
                time_mask=self.time_mask,
                train_pos=train_pos,
                train_time=train_time,
                train_seg_time=train_seg_time
            )

        self.log_output_csv(test, time, predict_pos, actual_pos,
                            binned_err_light[0], binned_err_dark[0], binned_err_light_end[0])

    def log_output_csv(self, test: TrialSelection,
                       time: np.ndarray,
                       pred_pos: np.ndarray,
                       actual_pos: np.ndarray,
                       binned_err_light: np.ndarray,
                       binned_err_dark: np.ndarray,
                       binned_err_light_end: np.ndarray,
                       *,
                       agg_func: Literal['median', 'mean'] = 'median'):
        """
        Log cv foreach results to csv

        :param test: ``TrialSelection``
        :param time: Time for test dataset. `Array[float, T]`
        :param pred_pos: Predicted position. `Array[float, T]`
        :param actual_pos: Actual position. `Array[float, T]`
        :param binned_err_light: Binned error in light. `Array[float, B]`
        :param binned_err_dark: Binned error in dark. `Array[float, B]`
        :param binned_err_light_end: Optional, Binned error in light_end. `Array[float, B]`
        :param agg_func: numpy attribute used for aggregate. by default use ``np.median()``
        :return:
        """
        dat = self.dat

        headers = ['n_cv', 'session', 'n_trials', 'decode_err', 'trial_indices', 'binned_err']
        with csv_header(self._csv_output, headers, append=True) as csv:
            err = self._wrap_diff(pred_pos, actual_pos, dat.trial_length)
            func = getattr(np, agg_func)

            def _array_to_str(array: np.ndarray) -> str:
                return ' '.join(array.astype(np.str_).tolist())

            #
            if dat.endswith_dark():
                fields = ['light', 'dark']
                dark_time = dat.light_off_time
                dark_lap = dat.light_off_lap
                light_again_time = dat.position_time[-1]
                light_again_lap = dat.n_trials - 1  # last
            else:
                fields = ['light', 'dark', 'light_end']
                dark_time = dat.light_off_time[0]
                dark_lap = dat.light_off_lap
                light_again_time = dat.light_off_time[1]
                light_again_lap = dat.get_light_end_trange()[0]

            #
            for session in fields:
                match session:
                    case 'light':
                        error = func(err[(time < dark_time)])
                        n_trials = np.count_nonzero(test.selected_trials < dark_lap)
                        binned_err = binned_err_light
                    case 'dark':
                        mx = np.logical_and(dark_time < time, time < light_again_time)
                        error = func(err[mx])
                        mxl = np.logical_and(dark_lap < test.selected_trials, test.selected_trials < light_again_lap)
                        n_trials = np.count_nonzero(mxl)
                        binned_err = binned_err_dark
                    case 'light_end':
                        error = func(err[light_again_time < time])
                        n_trials = np.count_nonzero(light_again_lap < test.selected_trials)
                        binned_err = binned_err_light_end

                trials = _array_to_str(self.get_test_trials(session))
                binned_err = _array_to_str(binned_err)

                csv(self._current_train_test_index, session, n_trials, error, trials, binned_err)

    def csv_to_parquet(self) -> pl.DataFrame:
        """parse tmp csv to parquet for array like table storage"""
        df = pl.read_csv(self._csv_output)
        df = (df.with_columns(pl.col('trial_indices').str.split(' ').cast(pl.List(pl.Int64)))
              .with_columns(pl.col('binned_err').str.split(' ').cast(pl.List(pl.Float64))))

        self._csv_output.unlink()
        df.write_parquet(self.parquet_output)
        print_save(self.parquet_output)
        print(df)

        return df

    def plot_decode_foreach(self, time, pred_pos, actual_pos, fr_raw, light_err, dark_err, light_end_err,
                            light_off_time, ylabel, *,
                            time_mask: tuple[float, float] | None = None,
                            train_pos: np.ndarray | None = None,
                            train_time: np.ndarray | None = None,
                            train_seg_time: np.ndarray | None = None):
        """
        Plot decode results foreach iteration

        :param time: Time Array in sec. `Array[float, T]`
        :param pred_pos: Predicted position. `Array[float, T]`
        :param actual_pos: Actual position. `Array[float, T]`
        :param fr_raw: Raw firing. `Array[float, [Traw, N | N']]`
        :param light_err: Binned decoding error in light session. `Array[float, [2, B]]`
        :param dark_err: Binned decoding error in dark session. `Array[float, [2, B]]`
        :param light_end_err: Optional, Binned decoding error in light_end session. `Array[float, [2, B]]`
        :param light_off_time: Time of light off in sec
        :param ylabel: ylabel for the firing activity
        :param time_mask: Time mask interval. (START, END) in sec.
        :param train_pos: Train position. `Array[float, TT]`
        :param train_time: Train time. `Array[float, TT]`
        """
        do_train_plot = False
        raw_time = self.dat.act_time

        if train_pos is not None and train_time is not None:
            do_train_plot = True

        # time masking
        if time_mask is not None:
            start, end = time_mask
            mx = np.logical_and(time > start, time < end)
            time = time[mx]
            pred_pos = pred_pos[mx]
            actual_pos = actual_pos[mx]
            if pred_pos.size == 0 or actual_pos.size == 0:
                raise RuntimeError('empty datapoint in the selected time interval')

            raw_mx = np.logical_and(raw_time > start, raw_time < end)
            raw_time = raw_time[raw_mx]
            fr_raw = fr_raw[raw_mx]

            if do_train_plot:
                mx = np.logical_and(train_time > start, train_time < end)
                train_time = train_time[mx]
                train_pos = train_pos[mx]

        #
        filename = self.dat.filepath.stem + f'_decode_foreach_cv{self._current_train_test_index}.pdf'
        output = self.output_dir / filename if self.output_dir else None
        nrow = 2 if light_end_err is None else 3

        with plot_figure(output, 7, nrow, figsize=(12, 8)) as _ax:
            ax0 = ax_merge(_ax)[:2, :]
            plot_decode_actual_position(ax0, time, pred_pos, actual_pos, time_mask=time_mask)

            if do_train_plot:
                plot_train_position(ax0, train_time, train_pos)

            ax1 = ax_merge(_ax)[2:4, :]
            plot_firing_rate(ax1, raw_time, fr_raw, time_mask=self.time_mask, smooth=self.smooth_fr, ylabel=ylabel)
            ax1.sharex(ax0)

            ax2 = ax_merge(_ax)[4:6, :]
            err = self._wrap_diff(pred_pos, actual_pos, self.dat.trial_length)
            plot_decoding_error(ax2, time, err, time_ann=light_off_time, train_seg_time=train_seg_time)
            ax2.sharex(ax0)

            # position binned
            ax = _ax[6, 0]
            plot_binned_decoding_error(ax, self.trial_length, light_err[0], light_err[1])
            ax.set_title('light')

            ax = _ax[6, 1]
            plot_binned_decoding_error(ax, self.trial_length, dark_err[0], dark_err[1])
            ax.sharey(_ax[6, 0])
            ax.set_title('dark')

            if light_end_err is not None:
                ax = _ax[6, 2]
                plot_binned_decoding_error(ax, self.trial_length, light_end_err[0], light_end_err[1])
                ax.sharey(_ax[6, 0])
                ax.set_title('light_end')

    def calc_position_binned_error(self,
                                   time, pred_pos, actual_pos) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Take all the cv results of trial-averaged decoding error as a function of position bins

        :param time:
        :param pred_pos:
        :param actual_pos:
        :return: light/dark mean and sem decoding error. (Array[float, [2, B]], Array[float, [2, B]])
        """
        nbins = int(self.trial_length / self.spatial_bin_size)
        pbs = PositionBinnedSig(self.dat, bin_range=(0, self.trial_length, nbins))

        def get_binned_error(trial: np.ndarray) -> np.ndarray:
            act = pbs.calc_binned_signal(time, actual_pos, trial, running_epoch=self.running_epoch)
            pred = pbs.calc_binned_signal(time, pred_pos, trial, running_epoch=self.running_epoch)
            binned_err = [self._wrap_diff(act[t], pred[t]) for t in range(len(trial))]
            return np.vstack([np.mean(binned_err, axis=0),
                              scipy.stats.sem(binned_err, axis=0)])

        light_trials = self.get_test_trials('light')
        dark_trials = self.get_test_trials('dark')
        light_ret = get_binned_error(light_trials)
        dark_ret = get_binned_error(dark_trials)

        if not self.dat.endswith_dark():
            light_end_trials = self.get_test_trials('light_end')
            light_end_ret = get_binned_error(light_end_trials)
        else:
            light_end_ret = None

        return light_ret, dark_ret, light_end_ret

    def plot_decode_cv(self):
        """Plot decode summary cross-validation testset results"""
        df = pl.read_parquet(self.parquet_output)
        filename = self.dat.filepath.stem + '_decode_summary.pdf'
        output = self.output_dir / filename if self.output_dir else None
        with plot_figure(output, figsize=(3, 8)) as ax:
            violin_boxplot(ax, df.to_pandas(), x='session', y='decode_err')
            ax.set(ylabel='decode_error(cm)', ylim=(0, 40))

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
        Running epoch masking and interpolation for the decoding parameters

        :param position_time: Position time. `Array[float, P]`
        :param position: Position value. `Array[float, P]`
        :param velocity: Velocity value in cm/s. `Array[float, P]`
        :param fr: Neural activity. `Array[float, [N, T]]`
        :param act_time: Activity time. `Array[float, T]`
        :param position_down_sampling: If True, interpolate position to neural activity shape, otherwise, vice versa
        :return: After running epoch masking and interpolation
            - fr: Neural activity.`Array[float, [N, T']]`.
            - time: Neural activity time. `Array[float, T']`.
            - position: Animal's position. `Array[float, T']`.
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
    def _all_epoch_masking(position_time: np.ndarray,
                           position: np.ndarray,
                           fr: np.ndarray,
                           act_time: np.ndarray,
                           position_down_sampling: bool = True):
        """
        Interpolation for decoding parameters

        :param position_time: Position time. `Array[float, P]`
        :param position: Position value. `Array[float, P]`
        :param fr: Neural activity. `Array[float, [N, T]]`
        :param act_time: Activity time. `Array[float, T]`
        :param position_down_sampling: If True, interpolate position to neural activity shape, otherwise, vice versa
        :return: After interpolation
            - fr: Neural activity.`Array[float, [N, T']]`.
            - time: Neural activity time. `Array[float, T']`.
            - position: Animal's position. `Array[float, T']`.
        """

        if position_down_sampling:
            position = interp1d(position_time, position, bounds_error=False, fill_value='extrapolate')(act_time)
            time = act_time
        else:
            fr = interp1d(position_time, position, bounds_error=False, fill_value=0)(position_time)
            time = position_time

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
