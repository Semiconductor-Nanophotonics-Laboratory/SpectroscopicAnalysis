from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Iterable
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# =========================
# Standard data container
# =========================
@dataclass
class StandardSpectrumMap:
    spectra: np.ndarray   # (N, M)
    xy: np.ndarray        # (N, 2)
    axis: np.ndarray      # (M,)
    unit: str = ""

    def validate(self) -> None:
        if self.spectra.ndim != 2:
            raise ValueError("spectra must be 2D (N, M).")
        if self.xy.shape != (self.spectra.shape[0], 2):
            raise ValueError(f"xy must be (N,2); got {self.xy.shape}, N={self.spectra.shape[0]}")
        if self.axis.shape != (self.spectra.shape[1],):
            raise ValueError(f"axis must be (M,); got {self.axis.shape}, M={self.spectra.shape[1]}")
        if not np.isfinite(self.axis).all():
            raise ValueError("axis contains non-finite values.")
        if not np.isfinite(self.xy).all():
            raise ValueError("xy contains non-finite values.")

    @property
    def shape(self) -> Tuple[int, int]:
        return self.spectra.shape


# =========================
# Public API: IO
# =========================
def load_StandardSpectrumMapNpz(path: str) -> StandardSpectrumMap:
    data = np.load(path, allow_pickle=False)
    spectra = data["spectra"].astype(np.float64, copy=False)
    xy = data["xy"].astype(np.float64, copy=False)
    axis = data["axis"].astype(np.float64, copy=False)
    unit = ""
    if "unit" in data.files:
        try:
            unit = str(data["unit"])
        except Exception:
            unit = ""
    obj = StandardSpectrumMap(spectra=spectra, xy=xy, axis=axis, unit=unit)
    obj.validate()
    return obj


def save_StandardSpectrumMap_toNpz(obj: StandardSpectrumMap, npz_path: str, *, compressed: bool = True) -> None:
    obj.validate()
    saver = np.savez_compressed if compressed else np.savez
    saver(npz_path,
          spectra=obj.spectra.astype(np.float64, copy=False),
          xy=obj.xy.astype(np.float64, copy=False),
          axis=obj.axis.astype(np.float64, copy=False),
          unit=np.array(obj.unit))


# =========================
# Internal helpers
# =========================
def _ensure_sorted_unique_axis(obj: StandardSpectrumMap) -> StandardSpectrumMap:
    axis = obj.axis
    spectra = obj.spectra
    order = np.argsort(axis, kind="mergesort")
    axis_sorted = axis[order]
    spectra_sorted = spectra[:, order]
    uniq = np.unique(axis_sorted)
    if uniq.size == axis_sorted.size:
        return StandardSpectrumMap(spectra_sorted, obj.xy, axis_sorted, obj.unit)
    merged = np.empty((spectra.shape[0], uniq.size), dtype=np.float64)
    for k, xv in enumerate(uniq):
        mask = (axis_sorted == xv)
        with np.errstate(invalid="ignore"):
            merged[:, k] = np.nanmean(spectra_sorted[:, mask], axis=1)
    return StandardSpectrumMap(merged, obj.xy.copy(), uniq, obj.unit)


def _build_new_axis(xmin: Optional[float], xmax: Optional[float], xstep: Optional[float], axis_current: np.ndarray) -> np.ndarray:
    a_min = float(axis_current.min())
    a_max = float(axis_current.max())
    lo = a_min if xmin is None else float(xmin)
    hi = a_max if xmax is None else float(xmax)
    if lo > hi:
        lo, hi = hi, lo
    lo_clip = max(a_min, lo)
    hi_clip = min(a_max, hi)
    if lo_clip > hi_clip:
        return np.array([], dtype=np.float64)
    if xstep is None or xstep <= 0:
        return np.array([], dtype=np.float64)
    eps = 1e-12
    return np.arange(lo_clip, hi_clip + eps, float(xstep), dtype=np.float64)


def _interp_row(x_old: np.ndarray, y_old: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    mask = np.isfinite(y_old)
    x = x_old[mask]; y = y_old[mask]
    if x.size < 2:
        return np.full_like(x_new, np.nan, dtype=np.float64)
    xmin, xmax = x.min(), x.max()
    y_new = np.full_like(x_new, np.nan, dtype=np.float64)
    inside = (x_new >= xmin) & (x_new <= xmax)
    if inside.any():
        y_new[inside] = np.interp(x_new[inside], x, y)
    return y_new


# =========================
# 1) Spike filter (선형보간) — 전 행 적용
# =========================
def _partition_consecutive(idx: np.ndarray) -> Iterable[Tuple[int, int]]:
    if idx.size == 0:
        return []
    start = prev = int(idx[0])
    for v in idx[1:]:
        v = int(v)
        if v == prev + 1:
            prev = v
            continue
        yield (start, prev)
        start = prev = v
    yield (start, prev)

def spike_filter(spectrum: np.ndarray, threshold: int) -> np.ndarray:
    if not threshold or threshold <= 0:
        return spectrum
    signal = spectrum.copy()
    forward, backward = signal[:-1], signal[1:]
    difference = np.abs(forward - backward)
    noise_pos = np.where(difference > threshold)[0]
    if len(noise_pos) == 0:
        return signal
    try:
        for start, end in _partition_consecutive(noise_pos):
            if start == 0 or end >= len(signal) - 1:
                continue
            length = end - start + 1
            signal[start:end + 1] = np.linspace(signal[start - 1], signal[end + 1], num=length)
    except Exception as e:
        print(f'[warning] Could not apply spike filter: {e}')
    return signal

def process_SpikeFilter(obj: StandardSpectrumMap, threshold: int) -> StandardSpectrumMap:
    base = _ensure_sorted_unique_axis(obj)
    Y = base.spectra.copy()
    for i in range(Y.shape[0]):
        Y[i] = spike_filter(Y[i], threshold=threshold)
    return StandardSpectrumMap(Y, base.xy.copy(), base.axis.copy(), base.unit)

# --- Whitaker–Hayes despike helpers ---
def _modified_z_score(a: np.ndarray) -> np.ndarray:
    med = np.nanmedian(a)
    mad = np.nanmedian(np.abs(a - med))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(a, dtype=np.float64)
    return 0.6745 * (a - med) / mad

def process_Despike_WhitakerHayes(obj: StandardSpectrumMap,
                                  threshold: float = 7.0,
                                  m: int = 3) -> StandardSpectrumMap:
    """
    Whitaker–Hayes 방식:
      1) dy = diff(y)
      2) z = modified Z-score(dy)
      3) |z| > threshold 위치를 스파이크로 간주
      4) 해당 지점 y[i]를 주변 비스파이크 이웃 [i-m..i+m] 평균으로 치환
    """
    base = _ensure_sorted_unique_axis(obj)
    Y = base.spectra.copy()
    N, M = Y.shape
    if M < 3:  # 너무 짧으면 스킵
        return StandardSpectrumMap(Y, base.xy.copy(), base.axis.copy(), base.unit)

    m = max(1, int(m))
    for r in range(N):
        y = Y[r]
        dy = np.diff(y)                # 길이 M-1
        z  = _modified_z_score(dy)
        spike = np.abs(z) > float(threshold)
        if not np.any(spike):
            continue

        y_out = y.copy()
        idxs = np.where(spike)[0]      # dy 인덱스
        for i in idxs:
            tgt = i                    # y에서 치환할 인덱스
            a = max(0, tgt - m); b = min(M, tgt + m + 1)
            win_idx = np.arange(a, b)

            # 스파이크 근접 지점 제외(보수적으로 i,i+1 제외)
            bad = {i}
            if i + 1 < M: bad.add(i + 1)
            good = [j for j in win_idx if (j not in bad) and np.isfinite(y_out[j])]

            if len(good) > 0:
                y_out[tgt] = float(np.mean([y_out[j] for j in good]))
            else:
                # 보간 fallback
                L = tgt - 1
                while L >= 0 and not np.isfinite(y_out[L]): L -= 1
                R = tgt + 1
                while R < M and not np.isfinite(y_out[R]): R += 1
                if L >= 0 and R < M:
                    y_out[tgt] = np.interp(base.axis[tgt], [base.axis[L], base.axis[R]], [y_out[L], y_out[R]])
                elif L >= 0:
                    y_out[tgt] = y_out[L]
                elif R < M:
                    y_out[tgt] = y_out[R]
        Y[r] = y_out

    return StandardSpectrumMap(Y, base.xy.copy(), base.axis.copy(), base.unit)

def process_Despike(obj: StandardSpectrumMap,
                    mode: str = "simple",
                    *,
                    # simple(=spike_filter) params
                    sf_threshold: int = 50,
                    # Whitaker–Hayes params
                    wh_threshold: float = 7.0,
                    wh_m: int = 3) -> StandardSpectrumMap:
    """
    mode: 'simple' | 'wh' | 'none'
      - 'simple': spike_filter(차이 임계치 → 선형보간)
      - 'wh'    : Whitaker–Hayes(modified Z of diff)
      - 'none'  : 미적용
    """
    mode = (mode or "simple").lower()
    if mode == "none":
        return obj
    if mode == "simple":
        return process_SpikeFilter(obj, threshold=sf_threshold)
    if mode == "wh":
        return process_Despike_WhitakerHayes(obj, threshold=wh_threshold, m=wh_m)
    raise ValueError("Despike mode must be 'simple', 'wh', or 'none'.")


# =========================
# 2) ROI / Interpolate
# =========================
def process_Roi_Interpolate(obj: StandardSpectrumMap,
                            xmin: Optional[float] = None,
                            xmax: Optional[float] = None,
                            xstep: Optional[float] = None,
                            interpolate: bool = True,
                            *,
                            round_decimals: Optional[int] = None  # << 추가
                            ) -> StandardSpectrumMap:
    if xmin is None and xmax is None and xstep is None and round_decimals is None:
        return StandardSpectrumMap(obj.spectra.copy(), obj.xy.copy(), obj.axis.copy(), obj.unit)

    base = _ensure_sorted_unique_axis(obj)
    a_min, a_max = base.axis.min(), base.axis.max()
    lo_req = a_min if xmin is None else float(xmin)
    hi_req = a_max if xmax is None else float(xmax)
    if lo_req > hi_req:
        lo_req, hi_req = hi_req, lo_req
    lo_clip = max(a_min, lo_req)
    hi_clip = min(a_max, hi_req)
    if lo_clip > hi_clip:
        raise ValueError(f"ROI [{lo_req}, {hi_req}] is outside data range [{a_min}, {a_max}].")

    def _slice_only() -> StandardSpectrumMap:
        keep = (base.axis >= lo_clip) & (base.axis <= hi_clip)
        if not np.any(keep):
            raise ValueError("No samples remain after ROI slicing.")
        Y = base.spectra[:, keep].copy()
        if round_decimals is not None:
            Y = np.round(Y, decimals=int(round_decimals))
        return StandardSpectrumMap(Y, base.xy.copy(), base.axis[keep], base.unit)

    if not interpolate:
        return _slice_only()
    if xstep is None or (isinstance(xstep, (int, float)) and xstep <= 0):
        return _slice_only()

    axis_new = _build_new_axis(xmin, xmax, xstep, base.axis)
    if axis_new.size == 0:
        raise ValueError("No overlap between requested ROI and data axis.")
    if axis_new.shape == base.axis.shape and np.allclose(axis_new, base.axis):
        Y = base.spectra.copy()
        if round_decimals is not None:
            Y = np.round(Y, decimals=int(round_decimals))
        return StandardSpectrumMap(Y, base.xy.copy(), base.axis.copy(), base.unit)

    N = base.spectra.shape[0]
    out = np.empty((N, axis_new.size), dtype=np.float64)
    for i in range(N):
        out[i] = _interp_row(base.axis, base.spectra[i], axis_new)

    if round_decimals is not None:
        out = np.round(out, decimals=int(round_decimals))

    return StandardSpectrumMap(out, base.xy.copy(), axis_new, base.unit)


# =========================
# 3) Savitzky–Golay smoothing — 전 행 적용
# =========================
def smooth_spectrum(y: np.ndarray, window_length: int, polyorder: int = 3,
                    mode: str = 'interp', strength: int = 1) -> np.ndarray:
    if window_length == -1:
        window_length = 5
    elif window_length >= len(y):
        window_length = len(y) - 1
    elif window_length % 2 == 0:
        window_length += 1
    elif window_length <= 0:
        return y
    elif 0 < window_length < 5:
        window_length = 5
    elif not window_length:
        return y
    else:
        window_length = max(5, window_length)
    polyorder = max(0, int(polyorder))
    if window_length <= polyorder:
        window_length = max(polyorder + 1 + (polyorder % 2 == 0), 5)
        if window_length % 2 == 0:
            window_length += 1
    out = y.copy()
    for _ in range(int(max(1, strength))):
        out = savgol_filter(out, window_length=window_length, polyorder=polyorder, mode=mode)
    return out

def process_SmoothSavGol(obj: StandardSpectrumMap,
                         window_length: int,
                         polyorder: int = 3,
                         mode: str = 'interp',
                         strength: int = 1) -> StandardSpectrumMap:
    base = _ensure_sorted_unique_axis(obj)
    Y = base.spectra.copy()
    for i in range(Y.shape[0]):
        Y[i] = smooth_spectrum(Y[i], window_length=window_length, polyorder=polyorder, mode=mode, strength=strength)
    return StandardSpectrumMap(Y, base.xy.copy(), base.axis.copy(), base.unit)


# =========================
# 4) Normalize (shift to min=0 → scale to max=1)
# =========================
def process_Normalize(obj: StandardSpectrumMap,
                      mode: str = "global",
                      *,
                      round_decimals: Optional[int] = None  # << 추가
                      ) -> StandardSpectrumMap:
    """
    Normalize spectra to min=0 and max=1.
      - mode="global": 행 최소값으로 shift 후, 전역 최대값으로 공통 스케일.
      - mode="instance": 행 최소값으로 shift 후, 각 행 최대값으로 개별 스케일.
    """
    base = _ensure_sorted_unique_axis(obj)
    Y = base.spectra.copy()

    if mode not in ("global", "instance"):
        raise ValueError("mode must be 'global' or 'instance'.")

    # 1) per-row min shift -> 0 baseline
    for i in range(Y.shape[0]):
        rmin = np.nanmin(Y[i])
        if np.isfinite(rmin):
            Y[i] = Y[i] - rmin

    # 2) scale
    if mode == "global":
        g_max = np.nanmax(Y)
        if np.isfinite(g_max) and g_max > 0:
            Y = Y / g_max
    else:
        for i in range(Y.shape[0]):
            rmax = np.nanmax(Y[i])
            if np.isfinite(rmax) and rmax > 0:
                Y[i] = Y[i] / rmax

    # 3) optional rounding
    if round_decimals is not None:
        Y = np.round(Y, decimals=int(round_decimals))

    return StandardSpectrumMap(Y, base.xy.copy(), base.axis.copy(), base.unit)

# =========================
# 5) Baseline removal (selectable)
#    - mode: 'none' | 'original' | 'peakstrip'
# =========================
def _baseline_original(x: np.ndarray, y: np.ndarray,
                       window_size: int, step_size: int) -> tuple[np.ndarray, np.ndarray]:
    bx, by = [], []
    i, n = 0, len(x)
    while i < n:
        j = min(i + window_size, n)
        xw = x[i:j]; yw = y[i:j]
        if len(yw) == 0:
            break
        k = int(np.argmin(yw))
        bx.append(float(xw[k])); by.append(float(yw[k]))
        i += step_size
    if len(bx) < 2:
        bx = [float(x[0]), float(x[-1])]; by = [float(y[0]), float(y[-1])]
    bx_u, idx_u = np.unique(np.asarray(bx), return_index=True)
    by_u = np.asarray(by)[idx_u]
    f = interp1d(bx_u, by_u, kind='linear', fill_value='extrapolate', assume_sorted=True)
    baseline = f(x)
    corrected = y - baseline
    return corrected, baseline

def _baseline_peakstrip(y: np.ndarray,
                        window_min: int = 31, window_max: int = 151, polyorder: int = 0) -> tuple[np.ndarray, np.ndarray]:
    # iterative peak-stripping (padding + 점차 큰 윈도우 SavGol로 상향평준화)
    from scipy import integrate
    l = len(y)
    lp = int(np.ceil(0.5 * l))
    padded = np.concatenate([np.ones(lp) * y[0], y, np.ones(lp) * y[-1]])
    S = padded.copy()
    if window_max is None:
        window_max = l // 3
    if window_max % 2 == 0:
        window_max -= 1
    A = []
    stripped = []
    def _strip_once(spectrum, window, polyorder):
        if window % 2 == 0:
            window += 1
        if window >= len(spectrum):
            window = len(spectrum) - 1 if len(spectrum) % 2 == 0 else len(spectrum)
        y_smooth = savgol_filter(spectrum, window, polyorder=polyorder, mode='interp')
        return np.where(spectrum > y_smooth, y_smooth, spectrum)
    for window in range(window_min, window_max + 1, 2):
        b = _strip_once(S, window, polyorder)
        A.append(integrate.trapezoid(S - b))
        stripped.append(b)
        S = b
        if len(A) > 3 and A[-2] < A[-3] and A[-2] < A[-1]:
            best = stripped[-2]
            break
    else:
        best = stripped[-1]
    baseline = best[lp:lp + l]
    corrected = y - baseline
    return corrected, baseline

def process_Baseline(obj: StandardSpectrumMap,
                     mode: str = "peakstrip",
                     *,
                     orig_window_size: int = 500,
                     orig_step_size: int = 150,
                     ps_window_min: int = 31,
                     ps_window_max: int = 151,
                     ps_polyorder:  int = 0) -> StandardSpectrumMap:
    """
    mode: 'none' | 'original' | 'peakstrip'
    - original  : 윈도우 최소값을 선형보간한 베이스라인
    - peakstrip : iterative peak-stripping (기본)
    """
    base = _ensure_sorted_unique_axis(obj)
    X = base.axis
    Y = base.spectra.copy()
    if mode.lower().strip() == "none":
        return StandardSpectrumMap(Y, base.xy.copy(), X.copy(), base.unit)

    for i in range(Y.shape[0]):
        y = Y[i]
        if mode == "original":
            y_corr, _ = _baseline_original(X, y, window_size=orig_window_size, step_size=orig_step_size)
        elif mode == "peakstrip":
            y_corr, _ = _baseline_peakstrip(y, window_min=ps_window_min, window_max=ps_window_max, polyorder=ps_polyorder)
        else:
            raise ValueError("baseline mode must be 'none', 'original', or 'peakstrip'.")
        Y[i] = y_corr
    return StandardSpectrumMap(Y, base.xy.copy(), X.copy(), base.unit)


# =========================
# QA figure saver (공통)
# =========================
def save_QA_SpectrumGrid(before: StandardSpectrumMap,
                         after: StandardSpectrumMap,
                         out_png: str,
                         *,
                         n_samples: int = 9,
                         seed: Optional[int] = 42,
                         suptitle: Optional[str] = None,
                         yscale: str = "linear") -> None:
    before.validate(); after.validate()
    if yscale not in ("linear", "log"):
        raise ValueError("yscale must be 'linear' or 'log'.")

    N_b = before.shape[0]
    N = min(N_b, n_samples)
    rng = np.random.default_rng(seed)
    idxs = np.sort(rng.choice(N_b, size=N, replace=False)) if N_b > 0 else np.array([], dtype=int)

    nrows, ncols = 3, 3
    total = nrows * ncols
    fig = plt.figure(figsize=(12, 10))
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)

    def _mask_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = np.isfinite(y)
        if yscale == "log":
            m &= (y > 0)
        return x[m], y[m]

    for k in range(total):
        ax = fig.add_subplot(nrows, ncols, k + 1)
        ax.grid(True, alpha=0.3, linewidth=0.6)
        ax.set_yscale(yscale)

        if k < len(idxs):
            i = idxs[k]
            xb, yb = _mask_xy(before.axis, before.spectra[i])
            if xb.size > 1:
                ax.plot(xb, yb, lw=1.2, label="before")
            if i < after.shape[0]:
                xa, ya = _mask_xy(after.axis, after.spectra[i])
                if xa.size > 1:
                    ax.plot(xa, ya, lw=1.2, linestyle="--", label="after")

            unit = after.unit or before.unit or ""
            ax.set_xlabel(f"x-axis ({unit})" if unit else "x-axis")
            ax.set_ylabel("Intensity (a.u.)" + (" [log]" if yscale == "log" else ""))
            xyi = before.xy[i] if i < before.xy.shape[0] else np.array([np.nan, np.nan])
            ax.set_title(f"idx={i}, (x={xyi[0]:.3g}, y={xyi[1]:.3g})", fontsize=9)
            ax.legend(fontsize=8, loc="best", frameon=False)
        else:
            ax.axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.96) if suptitle else None)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
