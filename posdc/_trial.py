import numpy as np
from typing_extensions import Self

from ._io import PositionDecodeInput

__all__ = ['TrialSelection']


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
    def selected_numbers(self) -> int:
        """Number of selected trials"""
        return len(self.selected_trials)

    @property
    def trial_time(self) -> np.ndarray:
        return self.dat.lap_time[self.selected_trials]

    @property
    def session_range(self) -> tuple[int, int]:
        r = self.dat.trial_index
        return int(r[0]), int(r[-1])

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

    def select_range(self, trial_range: tuple[int, int]) -> Self:
        select_trials = np.arange(*trial_range)
        return TrialSelection(self.dat, select_trials)

    def select_odd_in_range(self, trial_range: tuple[int, int]) -> Self:
        t = self.select_range(trial_range)
        odd_trials = np.arange(t.selected_trials[0] + 1, t.selected_trials[-1], 2)
        return TrialSelection(self.dat, odd_trials)

    def select_even_in_range(self, trial_range: tuple[int, int]) -> Self:
        t = self.select_range(trial_range)
        odd_trials = np.arange(t.selected_trials[0], t.selected_trials[-1], 2)
        return TrialSelection(self.dat, odd_trials)

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
        Masking data with the given ``selected_trials``

        :param data: `Array[float, [..., L, ...]]`
        :param axis: default is 1
        :return: `Array[float, [..., L', ...]]`
        """
        return np.take(data, self.selected_trials, axis=axis)

    def masking_time(self, t: np.ndarray) -> np.ndarray:
        """
        Create a time mask

        :param t: Time array in sec. `Array[float, T]`
        :return: Mask `Array[bool, T]`
        """
        time = self.dat.lap_time
        index = self.selected_trials

        # time index? find a trial index which interval include t
        trial_index = np.searchsorted(time, t) - 1  # (T,), ranging from 0 to L-1

        # trial index in selected_trial
        a = np.zeros_like(time, dtype=bool)  # (L+1,)
        a[index] = True
        ret = a[trial_index]  # (T)

        return ret

    def select_fraction(self, train_fraction: float) -> tuple[Self, Self]:
        """Select fraction of the trials for training"""
        total = self.selected_numbers
        n_test = int(total * (1 - train_fraction))
        start = np.random.randint(total - n_test) + self.session_range[0]
        trial_range = (start, start + n_test)

        test = self.select_range(trial_range)
        train = test.invert()

        return train, test
