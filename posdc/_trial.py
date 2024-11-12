import numpy as np
from typing_extensions import Self

from ._io import PositionDecodeInput

__all__ = ['TrialSelection']


class TrialSelection:
    """Trial selection class for cross validation"""
    __slots__ = ('dat', '_selected_trials')

    def __init__(self, dat: PositionDecodeInput,
                 selected_trial: np.ndarray | None = None):
        """
        :param dat: ``PositionDecodeInput``
        :param selected_trial:  `Array[int, L | L']`
        """

        self.dat = dat

        if selected_trial is not None:
            self._selected_trials = selected_trial
        else:
            self._selected_trials = np.arange(self.dat.n_trials)

    def __repr__(self):
        return f'SELECT: {self.selected_trials}'

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

    def kfold_cv(self, fold: int = 5, shuffle: bool = True) -> list[tuple[Self, Self]]:
        """

        :param fold: Number of folds
        :param shuffle: Whether to shuffle the data before splitting into batches
        :return:
        """
        from sklearn.model_selection import KFold
        kfold_iter = KFold(fold, shuffle=shuffle)

        ret = []
        for train_index, test_index in kfold_iter.split(self.selected_trials):
            train = TrialSelection(self.dat, train_index)
            test = TrialSelection(self.dat, test_index)
            ret.append((train, test))

        return ret

    def repeat_kfold_cv(self, fold: int = 5,
                        n_repeats: int = 10,
                        state: int | None = None) -> list[tuple[Self, Self]]:
        from sklearn.model_selection import RepeatedKFold
        kfold_iter = RepeatedKFold(n_splits=fold, n_repeats=n_repeats, random_state=state)

        ret = []
        for train_index, test_index in kfold_iter.split(self.selected_trials):
            train = TrialSelection(self.dat, train_index)
            test = TrialSelection(self.dat, test_index)
            ret.append((train, test))

        return ret

    def kfold_cv_in_range(self, trial_range: tuple[int, int],
                          fold: int = 5,
                          shuffle: bool = True) -> list[tuple[Self, Self]]:
        """
        Making K-Fold for a certain of trial range for **TRAINING** set,
        and test the model on the rest of the trials

        :param trial_range: Trial range of for **TRAINING** of the model
        :param fold: Number of folds
        :param shuffle: Whether to shuffle the data before splitting into batches
        :return:
        """
        from sklearn.model_selection import KFold
        kfold_iter = KFold(fold, shuffle=shuffle)

        ret = []
        t = self.select_range(trial_range)
        trials = t.selected_trials
        for train_index, _ in kfold_iter.split(trials):
            train_index = trials[train_index]  # map back for split index to actual trial index
            train = TrialSelection(self.dat, train_index)
            test_index = np.setdiff1d(self.dat.trial_index, train_index)  # test for the rest
            test = TrialSelection(self.dat, test_index)
            ret.append((train, test))

        return ret

    def repeat_kfold_cv_in_range(self):
        pass

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
