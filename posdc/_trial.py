import numpy as np
from typing_extensions import Self

from ._io import PositionDecodeInput

__all__ = ['TrialSelection']


class TrialSelection:
    """Trial selection class for cross validation"""
    __slots__ = ('dat', '_selected_trials', 'select_mode')

    def __init__(self, dat: PositionDecodeInput,
                 selected_trial: np.ndarray | None = None,
                 select_mode: str | None = None,
                 inherited_mode: tuple[str, ...] = ()):
        """
        :param dat: ``PositionDecodeInput``
        :param selected_trial: Selected trial indices, or None to select all trials. `Array[int, L]`
        :param select_mode: A string describing the selection mode
        :param inherited_mode: A tuple of inherited selection modes.
        """

        self.dat = dat

        if selected_trial is not None:
            self._selected_trials = selected_trial
        else:
            self._selected_trials = np.arange(self.dat.n_trials)

        self.select_mode = inherited_mode + ((select_mode,) if select_mode else ())

    def _new_instance(self, selected_trial: np.ndarray, select_mode: str) -> Self:
        """
        Create a new instance of TrialSelection with inherited select_mode.

        :param selected_trial: The trials to select.
        :param select_mode: The new selection mode to append.
        :return: A new TrialSelection instance.
        """
        return TrialSelection(self.dat, selected_trial, select_mode, self.select_mode)

    @property
    def selected_trials(self) -> np.ndarray:
        return np.copy(self._selected_trials)

    @property
    def n_selected_trials(self) -> int:
        """Number of selected trials"""
        return len(self.selected_trials)

    @property
    def trial_time(self) -> np.ndarray:
        """trial start time"""
        return self.dat.lap_time[self.selected_trials]

    @property
    def session_range(self) -> tuple[int, int]:
        """Get session range (start, end)"""
        r = self.dat.trial_index
        return int(r[0]), int(r[-1])

    def diff_all(self) -> Self:
        """Instance diff to all trials"""
        whole = np.arange(*self.session_range)
        ret = np.setdiff1d(whole, self.selected_trials)
        return self._new_instance(ret, 'diff_all')

    def select_odd(self) -> Self:
        """Select odd trials"""
        return self._new_instance(self._selected_trials[self._selected_trials % 2 == 1], 'select_odd')

    def select_even(self) -> Self:
        """Select even trials"""
        return self._new_instance(self._selected_trials[self._selected_trials % 2 == 0], 'select_even')

    def select_range(self, trial_range: tuple[int, int]) -> Self:
        """
        Select trial range

        :param trial_range: (start, end). inclusive
        """
        mask = (trial_range[0] <= self._selected_trials) & (self._selected_trials <= trial_range[1])
        return self._new_instance(self._selected_trials[mask], 'select_range')

    def select_odd_in_range(self, trial_range: tuple[int, int]) -> Self:
        """
        Select odd trials in range

        :param trial_range: (start, end). inclusive
        """
        selected_trials = self._selected_trials
        mask = (trial_range[0] <= selected_trials) & (self._selected_trials <= trial_range[1]) & (
                self._selected_trials % 2 == 1)
        return self._new_instance(self._selected_trials[mask], 'select_odd_in_range')

    def select_even_in_range(self, trial_range: tuple[int, int]) -> Self:
        """
        Select even trials in range

        :param trial_range: (start, end). inclusive
        :return:
        """
        selected_trials = self._selected_trials
        mask = (trial_range[0] <= selected_trials) & (selected_trials <= trial_range[1]) & (selected_trials % 2 == 0)
        return self._new_instance(selected_trials[mask], 'select_even_in_range')

    def kfold_cv(self, fold: int = 5,
                 shuffle: bool = True,
                 n_repeats: int | None = 10,
                 state: int | None = None,
                 test_from_all: bool = False) -> list[tuple[Self, Self]]:
        """
        K-Fold cross-validation from ``selected_trials``

        :param fold: Number of folds
        :param shuffle: Whether to shuffle the data before splitting into batches
        :param n_repeats: Number of repeats of K-fold
        :param state: Random state for K-Fold
        :param test_from_all: If True, test from all trials, otherwise, test from purely the ``test_index`` from split
        :return:
        """
        from sklearn.model_selection import KFold, RepeatedKFold

        if n_repeats is None:
            kfold_iter = KFold(fold, shuffle=shuffle, random_state=state)
        else:
            kfold_iter = RepeatedKFold(n_splits=fold, n_repeats=n_repeats, random_state=state)

        ret = []
        for train_index, test_index in kfold_iter.split(self.selected_trials):
            train = self._new_instance(self._selected_trials[train_index], 'kfold_cv-train')

            if test_from_all:
                t = np.setdiff1d(self.dat.trial_index, self._selected_trials[train_index])
                test = self._new_instance(t, 'kfold_cv-test-all')
            else:
                test = self._new_instance(self._selected_trials[test_index], 'kfold_cv-test')

            ret.append((train, test))

        return ret

    def take_along_trial_axis(self, data: np.ndarray, axis: int = 1) -> np.ndarray:
        """
        take data with the given ``selected_trials``

        :param data: `Array[float, [..., L, ...]]`
        :param axis: position of L, default is 1
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

        trial_index = np.searchsorted(time, t) - 1

        # trial index in selected_trial
        a = np.zeros_like(time, dtype=bool)
        a[index] = True
        ret = a[trial_index]

        return ret

    def select_fraction(self, train_fraction: float, seed: int | None = None) -> tuple[Self, Self]:
        """
        Select fraction of the trials for training and testing.

        testing data always continuous in selected trials.

        :param train_fraction: value between [0, 1]
        :param seed: Random seed
        :return: tuple of (train, test)
        """
        total = self.n_selected_trials
        n_test = int(total * (1 - train_fraction))

        if seed is not None:
            np.random.seed(seed)
        start = np.random.randint(total - n_test)

        test_index = np.arange(start, start + n_test)
        train_index = np.setdiff1d(np.arange(total), test_index)

        test = self._new_instance(self._selected_trials[test_index], 'select_fraction-test')
        train = self._new_instance(self._selected_trials[train_index], 'select_fraction-train')

        return train, test
