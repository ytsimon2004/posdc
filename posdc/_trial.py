import numpy as np
from typing_extensions import Self

from ._io import PositionDecodeInput

__all__ = [
    'TrialSelection',
    'random_split'
]


class TrialSelection:

    def __init__(self, dat: PositionDecodeInput,
                 selected_trial: np.ndarray | None = None):

        self.dat = dat

        if selected_trial is not None:
            self._selected_trials = selected_trial
        else:
            self._selected_trials = np.arange(self.dat.n_trials)

    @property
    def selected_trials(self) -> np.ndarray:
        return self._selected_trials

    @property
    def number_of_trials(self):
        return len(self.selected_trials)

    @property
    def trial_time(self) -> np.ndarray:
        return self.dat.lap_time[self.selected_trials]

    @property
    def session_range(self) -> tuple[int, int]:
        return int(self.selected_trials[0]), int(self.selected_trials[-1])

    def invert(self) -> Self:
        whole = np.arange(*self.session_range)
        ret = np.setdiff1d(whole, self.selected_trials)
        return TrialSelection(self.dat, ret)

    def select_odd(self) -> Self:
        odd_trials = np.arange(self.session_range[0] + 1, self.session_range[1], 2)
        return TrialSelection(self.dat, odd_trials)

    def select_even(self) -> Self:
        even_trials = np.arange(*self.session_range, 2)
        return TrialSelection(self.dat, even_trials)

    def selection_range(self, trial_range: tuple[int, int]) -> Self:
        select_trials = np.arange(*trial_range)
        return TrialSelection(self.dat, select_trials)

    def kfold_cv(self, fold: int = 5) -> list[tuple[Self, Self]]:
        """list of train/test TrialSelection, respectively"""
        from sklearn.model_selection import KFold
        kfold_iter = KFold(fold, shuffle=False)

        ret = []
        for train_index, test_index in kfold_iter.split(self.selected_trials):
            ret.append((TrialSelection(self.dat, train_index),
                        TrialSelection(self.dat, test_index)))

        return ret

    def masking_trial_matrix(self, data: np.ndarray, axis: int = 1) -> np.ndarray:
        """

        :param data: (..., L, ...)
        :param axis:
        :return:
            (..., L', ...)
        """
        return np.take(data, self.selected_trials, axis=axis)

    def masking_time(self, t: np.ndarray) -> np.ndarray:
        """

        :param t:  (T,) time array in sec
        :return:
            (T,) mask
        """

        time = self.dat.lap_time
        index = self.selected_trials

        # time index? find a trial index which interval include t
        trial_index = np.searchsorted(time, t) - 1  # (T,), ranging from 0 to L-1

        # trial index in selected_trial
        a = np.zeros_like(time, dtype=bool)  # (L+1,)
        a[index] = True
        ret = a[trial_index]  # (T)
        # two edge cases for trial index,
        # 1. t before first lap, trial_index = -1
        # 2. t after last lap , trial_index = L
        # a[L] always false since index's value range from 0 to L-1
        # a[edge_cases] always false

        return ret


def random_split(trial_select: TrialSelection, train_fraction: float = 0.8) -> tuple[TrialSelection, TrialSelection]:
    """randomized train test split based on the trial range"""
    total = trial_select.number_of_trials
    n_test = int(total * (1 - train_fraction))
    trial_start = np.random.randint(total - n_test) + trial_select.session_range[0]
    trial_range = (trial_start, trial_start + n_test)

    test = trial_select.selection_range(trial_range)
    train = test.invert()
    return train, test
