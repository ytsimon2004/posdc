from pathlib import Path

import numpy as np
from neuralib.locomotion import running_mask1d
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from ._io import PositionDecodeInput

__all__ = ['PositionRateMap',
           'PositionBinnedSig']


class PositionRateMap:

    def __init__(self, dat: PositionDecodeInput,
                 n_bins: int = 50,
                 sig_norm: bool = True,
                 force_compute: bool = False):
        """

        :param dat: ``PositionDecodeInput``
        :param n_bins: Number of bins for each lap(trial)
        :param sig_norm: Do 0 1 normalization
        :param force_compute: Force compute local position cache
        """

        self.dat = dat
        self.pbs = PositionBinnedSig(dat, bin_range=(0, dat.trial_length, n_bins), force_compute=force_compute)
        self.sig_norm = sig_norm

    @property
    def ratemap_cache(self) -> Path:
        file = self.dat.filepath
        return file.with_name(file.stem + '_ratemap_cache').with_suffix('.npy')

    def get_signal(self, lap_range: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        activities, baseline activities and imaging time in certain interval of laps (nlaps)

        :param lap_range: lap (trial) number from start to end (excluded)
        :return: image_time: `Array[float, T]` and signal: `Array[float, [N, T]|T]`
        """
        sig = self.dat.activity

        if self.sig_norm:
            from neuralib.calimg.suite2p import normalize_signal
            sig = normalize_signal(sig)

        imt = self.dat.act_time

        if lap_range is None:
            return imt, sig

        lt = self.dat.lap_time
        lt0 = lt[lap_range[0]]
        lt1 = lt[lap_range[1] - 1]
        ltx = np.logical_and(lt0 <= imt, imt <= lt1)

        if sig.ndim == 1:
            return imt[ltx], sig[ltx]
        else:
            return imt[ltx], sig[:, ltx]

    def load_binned_data(self, running_epoch: bool = False,
                         smooth: bool = False,
                         force_compute: bool = False) -> np.ndarray:
        """save or load the binned calcium activity data in all neurons.

        :param running_epoch: Whether calculate only running epoch
        :param smooth: Whether do the smoothing
        :param force_compute: Force compute local ratemap cache
        :return: `Array[float, [N, L, B]]`
        """

        if not self.ratemap_cache.exists() or force_compute:
            t, signal = self.get_signal(None)
            act = self.pbs.calc_binned_signal(t, signal, running_epoch=running_epoch, enable_tqdm=True, smooth=smooth)
            np.save(self.ratemap_cache, act)
        else:
            act = np.load(self.ratemap_cache)

        return act


# ================= #
# PositionBinnedSig #
# ================= #

class PositionBinnedSig:
    """
    Calculation of Position Binned Signal

    `Dimension parameters`:

        N = number of neurons

        L = number of trials (laps)

        B = number of position bins for each trials

        T = number of sample points for signal acquisition (i.e., neural signal)

        PT = number of sample points for position data acquisition (i.e., Encoder readout)

    """

    def __init__(
            self,
            dat: PositionDecodeInput,
            *,
            bin_range: int | tuple[int, int] | tuple[int, int, int] = (0, 150, 150),
            smooth_kernel: int = 3,
            force_compute: bool = False
    ):
        """
        :param dat: ``PositionDecodeInput``
        :param bin_range: END or tuple of (start, end, number)
        :param smooth_kernel: Smoothing gaussian kernel after binned
        :param force_compute: Force compute local position cache
        """

        self.dat = dat
        self.pos = dat.load_interp_position(force_compute=force_compute)

        match bin_range:
            case int():
                bin_range = (0, bin_range, bin_range)
            case [start, end] | (start, end):
                bin_range = (start, end, end - start)
            case [start, end, num] | (start, end, num):
                pass  # already right format
            case _:
                raise TypeError(f'wrong bin_range type or format, got {bin_range}')

        # running epoch
        self.running_velocity_threshold = 5
        self.running_merge_gap = 0.5
        self.running_duration_threshold = 1

        #
        self.smooth_kernel = smooth_kernel

        # cache
        self._bin_range = bin_range
        self._run_mask = None
        self._pos_mask_cache: dict[int, np.ndarray] = {}  # lap: pos_mask(bool arr)
        self._occ_map_cache: dict[(int, bool), np.ndarray] = {}  # (lap, running_epoch): occ_map

    @property
    def bin_range(self) -> tuple[int, int]:
        """Bin range (start, end). in cm"""
        return self._bin_range[0], self._bin_range[1]

    @property
    def n_bins(self) -> int:
        """Number of bins"""
        return self._bin_range[2]

    @property
    def position_time(self) -> np.ndarray:
        """Position Time"""
        return self.pos.t

    @property
    def position(self) -> np.ndarray:
        """Position in cm"""
        return self.pos.p

    @property
    def velocity(self) -> np.ndarray:
        """Velocity in cm/s"""
        return self.pos.v

    @property
    def run_mask(self) -> np.ndarray:
        if self._run_mask is None:
            self._run_mask = running_mask1d(self.position, self.velocity)
        return self._run_mask

    def _position_mask(self, lap: int) -> np.ndarray:
        """
        position mask based on initial lap number
        :param lap: 0-base lap index
        :return:
        """
        if lap not in self._pos_mask_cache:
            t = self.position_time

            t1 = self.dat.lap_time[lap]
            t2 = self.dat.lap_time[lap + 1]
            self._pos_mask_cache[lap] = np.logical_and(t1 < t, t < t2)

        return self._pos_mask_cache[lap]

    def _occupancy_map(self,
                       lap: int,
                       bins: np.ndarray,
                       pos: np.ndarray,
                       running_epoch=False) -> np.ndarray:
        """

        :param lap: 0-base lap_index
        :param bins:
        :param pos:
        :param running_epoch:
        :return:
        """
        key = (lap, running_epoch)
        if key not in self._occ_map_cache:
            self._occ_map_cache[key] = np.histogram(pos, bins)[0]

        return self._occ_map_cache[key]

    def calc_binned_signal(self,
                           t: np.ndarray,
                           signal: np.ndarray,
                           lap_range: tuple[int, int] | np.ndarray | None = None,
                           occ_normalize: bool = True,
                           smooth: bool = False,
                           running_epoch: bool = False,
                           enable_tqdm: bool = False,
                           norm: bool = False) -> np.ndarray:
        """
        Calculate binned signal

        :param t: Time array. `Array[float, T]`
        :param signal: Signal array. `Array[float, [N, T] | T]`
        :param lap_range:
            tuple type: lap (trial) index (0-based) from start to end (excluded)
            array type: trial index (0-based) array. i.e., for trial-wise cross validation
            None: all trials from all recording sessions
        :param occ_normalize: If do occupancy normalize
        :param smooth: If do the gaussian kernel smoothing after binned
        :param running_epoch: If only take running epoch
        :param enable_tqdm: enable tqdm progress bar
        :param norm: whether do the maximal normalization
        :return: Position binned signal. `Array[float, [N, L, B] | [L, B]]`
        """

        # activity on position time
        act = self._activity(t, signal)
        act_1d = act.ndim == 1
        run_mask = self.run_mask if running_epoch else None
        bins = np.linspace(*self.bin_range, num=self.n_bins + 1, endpoint=True)

        if lap_range is None or isinstance(lap_range, tuple):

            if lap_range is None:
                lap_index = self.dat.trial_index
                lap_range = lap_index[0], lap_index[-1]

            n_lap = lap_range[1] - lap_range[0] + 1
            enum = enumerate(range(*lap_range))

        elif isinstance(lap_range, np.ndarray):
            n_lap = len(lap_range)
            enum = enumerate(lap_range)
        else:
            raise TypeError(f'type:{type(lap_range)}')

        #
        if act_1d:
            ret = np.zeros((n_lap, self.n_bins))  # (L, B)
        else:
            ret = np.zeros((act.shape[0], n_lap, self.n_bins), dtype=np.float32)  # (N, L, B)

        if enable_tqdm:
            from tqdm import tqdm
            lap_iter = tqdm(enum, desc='get_binned_signal', unit='laps', ncols=80)
        else:
            lap_iter = enum

        occ = None
        for i, lap in lap_iter:
            pos_mask = self._position_mask(lap)
            if running_epoch:
                pos_mask = np.logical_and(pos_mask, run_mask)

            pos = self.position[pos_mask]

            if occ_normalize:  # occ_normalize is constant in this function
                occ = self._occupancy_map(lap, bins, pos, running_epoch=running_epoch)

            if act_1d:
                ret[i] = self._binned_signal(pos, bins, act[pos_mask], occ, smooth)  # (L, B)
            else:
                for n in range(act.shape[0]):
                    ret[n, i] = self._binned_signal(pos, bins, act[n, pos_mask], occ, smooth)  # (N, L, B)

        if norm:
            ret /= np.max(ret)

        return ret

    def _activity(self, t: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """
        Interpolation numbers of image time to the position time

        :param t: Time array. `Array[float, T]`
        :param signal: Signal array. `Array[float, [N, T] | T]`
        :return: Interpolated neural signal. `Array[float, [N, PT] | PT]`
        """
        if t.ndim != 1:
            raise RuntimeError(f'wrong time dimension : {t.ndim}')

        if signal.ndim == 1:
            n_samples = signal.shape[0]
        elif signal.ndim == 2:
            n_neuron, n_samples = signal.shape
            if n_neuron == 0:
                raise RuntimeError(f'empty signal. shape : {signal.shape}')
        else:
            raise RuntimeError(f'wrong signal dimension : {signal.ndim}')

        if len(t) != n_samples:
            raise RuntimeError(f't {t.shape} and signal {signal.shape} shape not matched')

        # activity on position time
        while True:
            try:
                ret = interp1d(
                    t,
                    signal,
                    axis=signal.ndim - 1,
                    bounds_error=False,
                    fill_value=0.0,
                    assume_sorted=True
                )(self.position_time)
                return ret
            except MemoryError as e:
                print(repr(e))
                input('press to continue')

    def _binned_signal(self, pos, bins, act, occ: np.ndarray | None, smooth: bool):
        """

        :param pos: position
        :param bins: bins
        :param act: weight
        :param occ: occupancy
        :param smooth:
        :return:
        """
        r, _ = np.histogram(pos, bins, weights=act)

        if occ is not None:
            r = r / occ

        if smooth:
            r[np.isnan(r)] = 0.0
            r = gaussian_filter1d(r, self.smooth_kernel, mode='wrap')

        return r
