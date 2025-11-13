from __future__ import annotations
# -*- coding: utf-8 -*-
"""
Andor SIF → StandardSpectrumMap NPZ converter

- Reads .sif (Andor/Andor Solis) mapping files using one of:
    pip install sif_parser        # or: conda install -c conda-forge sif-parser
    pip install sifreader
- Reshapes frames into a (Ygrid, Xgrid, spectrum_len) cube and emits a
  StandardSpectrumMap NPZ:
    spectra : (N, M) float64
    xy      : (N, 2) float64
    axis    : (M,)   float64
    unit    : str    (e.g., "nm" or "cm^-1")
- Coordinates are synthetic stage coordinates defined by the user.
  They do NOT alter the instrument data; they only label each frame with (x,y).
- If your SIF already encodes stage coordinates somewhere else, adapt `build_axes()`
  to read them instead of generating linear ranges.
"""

import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# =========================
# User settings (edit here)
# =========================
# Input .sif file (single) or directory containing SIFs
INPUTS: List[str] = [
    "./test.sif",   # example
    # add more paths here ...
]

# Coordinate layout (symmetric ranges)
Xsize: float = 12.0   # X in [-Xsize, +Xsize]
Ysize: float = 12.0   # Y in [-Ysize, +Ysize]
Xgrid: int   = 120    # number of X samples
Ygrid: int   = 120    # number of Y samples

# Traversal & orientation
FRAME_ORDER: str    = "YX"   # "YX" (Y-major) or "XY" (X-major)
FLIP_X_ORDER: bool  = False  # reverse X iteration order when mapping frames
NEGATE_X_COORD: bool = False # mirror X coordinates (if plot appears left-right flipped)
NEGATE_Y_COORD: bool = False # mirror Y coordinates

# Unit hint for the spectral axis (try "nm" for wavelength or "cm^-1" for Raman shift)
UNIT_HINT: str = "nm"

# Output directory
OUT_DIR = "./npz_out"

# Numeric formatting for log/prints only
MAX_DECIMALS: int = 7


# =========================
# StandardSpectrumMap spec
# =========================
@dataclass
class StandardSpectrumMap:
    spectra: np.ndarray   # (N, M) float64
    xy: np.ndarray        # (N, 2) float64
    axis: np.ndarray      # (M,)   float64
    unit: str = ""

    def validate(self) -> None:
        if self.spectra.ndim != 2:
            raise ValueError("spectra must be 2D (N, M).")
        N, M = self.spectra.shape
        if self.xy.shape != (N, 2):
            raise ValueError(f"xy must be (N,2); got {self.xy.shape}, expected N={N}")
        if self.axis.shape != (M,):
            raise ValueError(f"axis must be (M,); got {self.axis.shape}, expected M={M}")
        if not np.isfinite(self.axis).all():
            raise ValueError("axis contains non-finite values.")
        if not np.isfinite(self.xy).all():
            raise ValueError("xy contains non-finite values.")

def save_StandardSpectrumMap_toNpz(obj: StandardSpectrumMap, npz_path: str, *, compressed: bool = True) -> None:
    obj.validate()
    saver = np.savez_compressed if compressed else np.savez
    saver(npz_path,
          spectra=obj.spectra.astype(np.float64, copy=False),
          xy=obj.xy.astype(np.float64, copy=False),
          axis=obj.axis.astype(np.float64, copy=False),
          unit=np.array(obj.unit))


# =========================
# SIF backends
# =========================
_BACKEND = None
try:
    import sif_parser  # type: ignore
    _BACKEND = "sif_parser"
except Exception:
    try:
        import sifreader  # type: ignore
        _BACKEND = "sifreader"
    except Exception:
        _BACKEND = None

def _read_sif_with_sif_parser(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    returns: (data, axis)
      data: (frames, n_spec) float64
      axis: (n_spec,) float64
    """
    import sif_parser
    arr, info = sif_parser.np_open(path)  # arr: (frames, pixels) or (frames, h, w)
    arr = np.asarray(arr, dtype=float)

    # wavelength axis
    try:
        from sif_parser.utils import extract_calibration
        wl = extract_calibration(info)
        wl = np.asarray(wl, dtype=float)
    except Exception as e:
        raise RuntimeError(f"sif_parser: failed to read wavelength axis: {e}")

    # normalize to (frames, n_spec)
    if arr.ndim == 2:
        frames, n_spec = arr.shape
    elif arr.ndim == 3:
        f, a, b = arr.shape
        if a == 1 and b > 1:
            arr = arr.reshape(f, b)
        elif b == 1 and a > 1:
            arr = arr.reshape(f, a)
        else:
            arr = arr.reshape(f, a * b)
        frames, n_spec = arr.shape
    else:
        raise RuntimeError(f"sif_parser: unexpected ndim {arr.ndim}")

    if wl.size != n_spec:
        raise RuntimeError(f"sif_parser: wavelength length ({wl.size}) != spectral length ({n_spec})")
    return arr.astype(np.float64, copy=False), wl.astype(np.float64, copy=False)

def _read_sif_with_sifreader(path: str) -> Tuple[np.ndarray, np.ndarray]:
    from sifreader import SifReader
    sr = SifReader(path)
    arr = np.asarray(sr.get_data(), dtype=float)  # (frames, pixels)
    frames, n_spec = arr.shape

    wl = None
    if hasattr(sr, "get_wavelengths"):
        try:
            wl = sr.get_wavelengths()
        except Exception:
            wl = None
    if wl is None and hasattr(sr, "get_wavelength_calibration"):
        try:
            cal = sr.get_wavelength_calibration()
            if cal is not None:
                idx = np.arange(n_spec, dtype=float)
                wl = np.polyval(cal[::-1], idx)
        except Exception:
            wl = None
    if wl is None:
        raise RuntimeError("sifreader: no wavelength axis in SIF")

    wl = np.asarray(wl, dtype=float)
    if wl.size != n_spec:
        raise RuntimeError(f"sifreader: wavelength length ({wl.size}) != spectral length ({n_spec})")
    return arr.astype(np.float64, copy=False), wl.astype(np.float64, copy=False)

def read_sif(path: str) -> Tuple[np.ndarray, np.ndarray]:
    if _BACKEND == "sif_parser":
        return _read_sif_with_sif_parser(path)
    elif _BACKEND == "sifreader":
        return _read_sif_with_sifreader(path)
    else:
        raise SystemExit(
            "No SIF backend available.\n"
            "Install one of:\n"
            "  pip install sif_parser   # or conda install -c conda-forge sif-parser\n"
            "  pip install sifreader\n"
        )


# =========================
# Helpers
# =========================
def build_axes(xsize: float, ysize: float, xgrid: int, ygrid: int) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(-float(xsize), +float(xsize), int(xgrid), dtype=np.float64)
    ys = np.linspace(-float(ysize), +float(ysize), int(ygrid), dtype=np.float64)
    if NEGATE_X_COORD:
        xs = -xs
    if NEGATE_Y_COORD:
        ys = -ys
    return xs, ys

def fmt_number(x: float, max_decimals: int = 7) -> str:
    xf = float(x)
    if math.isfinite(xf) and xf.is_integer():
        return str(int(xf))
    s = f"{xf:.{max_decimals}f}".rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    return s


# =========================
# Core conversion
# =========================
def convert_one_sif_to_npz(path_sif: str,
                           xsize: float, ysize: float,
                           xgrid: int, ygrid: int,
                           frame_order: str = "YX",
                           flip_x_order: bool = False,
                           unit_hint: str = "nm") -> str:
    """Read a SIF file and write StandardSpectrumMap NPZ next to OUT_DIR"""
    data, axis = read_sif(path_sif)  # data: (frames, M), axis: (M,)
    frames, M = data.shape

    # sanity: grid size
    expected = int(xgrid) * int(ygrid)
    if frames != expected:
        raise RuntimeError(f"{Path(path_sif).name}: frames ({frames}) != Xgrid*Ygrid ({expected})")

    # reshape to canonical cube (Y, X, M)
    cube = data.reshape(int(ygrid), int(xgrid), M)

    xs, ys = build_axes(xsize, ysize, xgrid, ygrid)

    # Unroll cube -> (N, M) and parallel xy -> (N, 2)
    spectra_list: List[np.ndarray] = []
    xy_list: List[Tuple[float, float]] = []

    fo = (frame_order or "YX").upper()
    if fo == "YX":
        x_iter = range(xgrid - 1, -1, -1) if flip_x_order else range(xgrid)
        for yi in range(ygrid):
            for xi in x_iter:
                spectra_list.append(cube[yi, xi, :].astype(np.float64, copy=False))
                xy_list.append((float(xs[xi]), float(ys[yi])))
    elif fo == "XY":
        x_iter = range(xgrid - 1, -1, -1) if flip_x_order else range(xgrid)
        for xi in x_iter:
            for yi in range(ygrid):
                spectra_list.append(cube[yi, xi, :].astype(np.float64, copy=False))
                xy_list.append((float(xs[xi]), float(ys[yi])))
    else:
        raise ValueError("FRAME_ORDER must be 'YX' or 'XY'")

    spectra = np.vstack(spectra_list).astype(np.float64, copy=False)
    xy = np.asarray(xy_list, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)

    obj = StandardSpectrumMap(spectra=spectra, xy=xy, axis=axis, unit=unit_hint or "")
    obj.validate()

    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(path_sif).stem + ".npz")
    save_StandardSpectrumMap_toNpz(obj, str(out_path), compressed=True)
    return str(out_path)


# =========================
# Batch driver
# =========================
def _discover_inputs(inputs: List[str]) -> List[str]:
    found: List[str] = []
    for p in inputs:
        P = Path(p)
        if P.is_file() and P.suffix.lower() == ".sif":
            found.append(str(P))
        elif P.is_dir():
            for q in P.glob("*.sif"):
                found.append(str(q))
    # uniq & stable order
    seen = set(); result = []
    for f in found:
        if f not in seen:
            result.append(f); seen.add(f)
    return result

def main():
    paths = _discover_inputs(INPUTS)
    if not paths:
        print("No .sif inputs found. Edit INPUTS at top of file.")
        return

    print(f"[Backend] {_BACKEND or 'None'}")
    for p in paths:
        try:
            out = convert_one_sif_to_npz(
                p,
                xsize=Xsize, ysize=Ysize,
                xgrid=Xgrid, ygrid=Ygrid,
                frame_order=FRAME_ORDER,
                flip_x_order=FLIP_X_ORDER,
                unit_hint=UNIT_HINT,
            )
            print(f"[OK] {Path(p).name} -> {Path(out).name}")
        except SystemExit as e:
            raise
        except Exception as e:
            print(f"[ERR] {Path(p).name}: {e}")

if __name__ == "__main__":
    main()