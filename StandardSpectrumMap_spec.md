# STANDARD SPECTRUM MAP — Core Data & NPZ Format

> **Scope:** This document specifies only the in‑memory data structure and the `.npz` file schema required to share and reuse mapping spectroscopy data. Algorithmic details (preprocessing, QA, etc.) are intentionally excluded.

## 1) In‑Memory Structure

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class StandardSpectrumMap:
    spectra: np.ndarray   # shape: (N, M), dtype: float64
    xy:      np.ndarray   # shape: (N, 2), dtype: float64
    axis:    np.ndarray   # shape: (M,),   dtype: float64
    unit:    str = ""     # x-axis unit (e.g., "cm^-1", "nm")
```
**Field semantics**
- `N`: number of mapping points (spectra).
- `M`: number of x‑axis samples per spectrum.
- `spectra[i, j]`: intensity (a.u.) at x = `axis[j]` for point `i`.
- `xy[i] = [x_i, y_i]`: instrument (stage) coordinates of point `i` (units depend on export; often µm).
- `axis[j]`: physical x‑axis value (e.g., Raman shift in cm⁻¹, wavelength in nm). Monotonic increasing is recommended.
- `unit`: human‑readable label for `axis` (optional).

**Validation (recommended)**
- Shapes: `spectra.ndim == 2`, `xy.shape == (N, 2)`, `axis.shape == (M,)`.
- Types: all arrays stored and exchanged as `float64`.
- Values: `axis` and `xy` should contain finite values only.
- If `axis` has duplicates or is unsorted, sort and merge duplicates by mean before saving/using.

---

## 2) `.npz` File Schema

**Required keys**
- `spectra` : `float64`, shape `(N, M)`
- `xy`      : `float64`, shape `(N, 2)`
- `axis`    : `float64`, shape `(M,)`

**Optional keys**
- `unit`    : NumPy scalar array with a string (e.g., `np.array("cm^-1")`).

> No additional metadata is required by this spec.

**Save / Load reference**

```python
# save
np.savez_compressed(
    "standard_map.npz",
    spectra=spectra.astype(np.float64),
    xy=xy.astype(np.float64),
    axis=axis.astype(np.float64),
    unit=np.array(unit_str),  # optional
)

# load
d = np.load("standard_map.npz", allow_pickle=False)
spectra = d["spectra"].astype(np.float64, copy=False)
xy      = d["xy"].astype(np.float64, copy=False)
axis    = d["axis"].astype(np.float64, copy=False)
unit    = str(d["unit"]) if "unit" in d.files else ""
```

---

## 3) Compatibility Notes
- Use `float64` consistently to avoid precision drift across toolchains.
- Keep `xy` as exported by the instrument; downstream code should not assume a particular origin or orientation.
- `unit` is informational; consumers must not rely on its presence.
---

## FIT SPECTRUM MAP — Result NPZ Format (v1)

> **Scope:** This section specifies the **output NPZ** produced after peak fitting of a `StandardSpectrumMap`. It is independent of any particular fitting algorithm, but assumes a multi-peak parametric model (e.g., pseudo-Voigt, Gaussian, Lorentzian).

### 1) Dimensions and Symbols
- `N`: number of mapping points (spectra)
- `M`: number of x-axis samples per spectrum (length of `axis`)
- `P`: number of peaks in the chosen model

### 2) Required Keys (NPZ)
- `axis` : `float64` `(M,)` — x-axis used for fitting/reconstruction (same unit as the source map).
- `xy` : `float64` `(N, 2)` — stage coordinates of each fitted spectrum.
- `spectra_original` : `float64` `(N, M)` — the input spectra used for fitting (after any preprocessing applied upstream).
- `params_pos` : `float64` `(N, P)` — peak center positions.
- `params_width` : `float64` `(N, P)` — peak width parameter (commonly FWHM).
- `params_height` : `float64` `(N, P)` — peak amplitude (model-specific; often the profile height).
- `params_eta` : `float64` `(N, P)` — shape-mixing parameter in `[0, 1]` (e.g., pseudo-Voigt: 0=Gaussian, 1=Lorentzian).
- `params_base` : `float64` `(N,)` or `(N, K)` — baseline parameters per spectrum (scalar offset or K-term model; repository uses a scalar offset by default).

### 3) Optional Keys
- `unit` : NumPy scalar array with a string (e.g., `np.array("cm^-1")`).
- `peak_types` : `object`/`U` array `(P,)` — display labels for peaks (e.g., `["D1","D2","G","2D"]`).
- `valid_mask` : `bool` `(N,)` — `True` where the fit is considered valid (constraints satisfied, convergence achieved).
- `loss_final` : `float64` `(N,)` — final objective value per spectrum (e.g., MSE).
- `recon` : `float64` `(N, M)` — reconstructed spectra from fitted parameters on `axis`.
- `area` : `float64` `(N, P)` — integrated area per peak (if computed).
- `metadata_json` : `object` scalar — JSON string with run configuration (bounds, stage plan, optimizer, timestamps, etc.).

### 4) Semantics and Conventions
- All arrays are `float64` unless stated otherwise.
- `axis` should be **monotonic increasing**; consumers must not assume uniform spacing.
- `params_*` values are **per-spectrum, per-peak**; missing or constrained peaks should be indicated via `valid_mask=False` or sentinel values (e.g., `NaN`).
- If a multi-term baseline is used, `params_base` becomes `(N, K)` and `metadata_json` must include the baseline model description.

### 5) Minimal Save/Load Reference

```python
import numpy as np

# --- save (example) ---
np.savez_compressed(
    "fit_spectrum_map_v1.npz",
    axis=axis.astype(np.float64),
    xy=xy.astype(np.float64),
    spectra_original=spectra.astype(np.float64),
    params_pos=pos.astype(np.float64),
    params_width=width.astype(np.float64),
    params_height=height.astype(np.float64),
    params_eta=eta.astype(np.float64),
    params_base=base.astype(np.float64),
    unit=np.array(unit_str),                 # optional
    peak_types=np.array(peak_types),         # optional
    valid_mask=valid_mask,                   # optional
    loss_final=loss,                         # optional
    recon=recon.astype(np.float64),          # optional
    area=area.astype(np.float64),            # optional
    metadata_json=np.array(metadata_json),   # optional (string)
)

# --- load (example) ---
d = np.load("fit_spectrum_map_v1.npz", allow_pickle=False)
axis   = d["axis"];   xy     = d["xy"];    spectra = d["spectra_original"]
pos    = d["params_pos"];    width = d["params_width"]
height = d["params_height"]; eta   = d["params_eta"]
base   = d["params_base"]
unit   = str(d["unit"]) if "unit" in d.files else ""

peak_types = d["peak_types"] if "peak_types" in d.files else None
valid_mask = d["valid_mask"] if "valid_mask" in d.files else None
loss       = d["loss_final"] if "loss_final" in d.files else None
recon      = d["recon"]      if "recon"      in d.files else None
area       = d["area"]       if "area"       in d.files else None
metadata   = str(d["metadata_json"]) if "metadata_json" in d.files else "{}"
```

### 6) Compatibility Notes
- Keep the same `axis` length `M` for `spectra_original` and `recon` to simplify downstream comparison.
- Use `valid_mask` to filter unreliable fits before mapping parameters (`pos`, `width`, `area`) to images.
- Consumers should **not** assume that `P` (number of peaks) is identical across projects; always read `P` from the loaded arrays.
