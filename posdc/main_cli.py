import random
from functools import cached_property
from typing import Literal, Final

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from neuralib.argp import AbstractParser, argument, union_type
from neuralib.model.bayes_decoding import place_bayes
from neuralib.typing import PathLike

from posdc._io import PositionDecodeInput
from posdc._ratemap import PositionRateMap
from posdc._trial import TrialSelection, random_split

TRAIN_TEST_SPLIT_METHOD = Literal['odd', 'even', 'random_split']
CrossValidateType = TRAIN_TEST_SPLIT_METHOD | int


# TODO if interpolated the position every lap
# TODO position 01 is total length
# TODO activity unit, negative value

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

    spatial_bin_size: float | None = argument(
        '--spatial-bin',
        metavar='VALUE',
        default=None,
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
        type=union_type(int, str),
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
        default=None,
        help='number of random neurons'
    )

    seed: int | None = argument(
        '--seed',
        metavar='VALUE',
        default=None,
        help='seed for random number generator'
    )

    #
    train_test_list: list[tuple[TrialSelection, TrialSelection]]
    trial_length: Final[int] = 150
    """in cm"""
    position: np.ndarray
    dat: PositionDecodeInput

    #
    number_iter: int | None
    _current_train_test_index: int | None

    def post_parsing(self):
        pass

    def run(self):
        self.dat = PositionDecodeInput.load_hdf(self.file, use_deconv=self.use_deconv, pos_norm=self.trial_length)
        trial = TrialSelection(self.dat)
        self.train_test_list = self.trial_cross_validation(trial)

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
            case _:
                raise ValueError('')

        if self.seed is not None:
            random.seed(self.seed)
        ret[random.sample(range(n_neurons), n)] = 1

        return np.nonzero(ret)[0]

    @property
    def train_test(self) -> tuple[TrialSelection, TrialSelection]:
        sz = len(self.train_test_list)
        return self.train_test_list[self._current_train_test_index % sz]

    def trial_cross_validation(self, trial: TrialSelection) -> list[tuple[TrialSelection, TrialSelection]]:

        match self.cross_validation:
            case str():
                match self.cross_validation:
                    case 'even':
                        train_set = trial.select_even()
                        test_set = trial.select_odd()
                    case 'odd':
                        train_set = trial.select_odd()
                        test_set = trial.select_even()
                    case 'random_split':
                        train_set, test_set = random_split(trial, self.train_fraction)
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
        fr = dat.activity  # (N, T)
        index = trial.session_range
        start_time = trial.trial_time[0]
        end_time = trial.trial_time[-1]

        if self.running_epoch:
            fr, fr_time, position = ...  # TODO
        else:
            fr_time = dat.time
            position = dat.position

        train, test = self.train_test
        t_mask = test.masking_time(fr_time)
        fr = fr[:, t_mask]
        fr_time = fr_time[t_mask]
        position = position[t_mask]

        # rate map
        n_bins = int(self.trial_length / self.spatial_bin_size)
        rate_map = PositionRateMap(dat, n_bins=n_bins).load_binned_data(running_epoch=self.running_epoch)  # (N, L, X)
        rate_map = train.masking_trial_matrix(rate_map, 1)  # (N, L', X)
        rate_map = np.nanmean(rate_map, axis=1)[neuron_list]

        trial_index = np.zeros((index[1] - index[0]), dtype=int)  # (L')
        trial_index[train.selected_trials - index[0]] += 1  # train
        trial_index[test.selected_trials - index[0]] += 2  # test

        pr = place_bayes(fr, rate_map, self.spatial_bin_size)
        predict_pos = np.argmax(pr, axis=1) * self.spatial_bin_size

        return pr, predict_pos


if __name__ == '__main__':
    PositionDecodeOptions().main()
