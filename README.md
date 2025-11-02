All copyrights and usage rights for the code are held by Dong-Joon Yi. 
iver.ydj@gmail.com

# StandardSpectrumMap Toolkit

A minimal, reproducible toolkit for **mapping spectroscopy** (e.g., Raman/PL) that standardizes data into a single NPZ schema, offers a configurable preprocessing pipeline, and performs multi-peak **pseudovoigt-unified** fitting with staged optimization. This repository is designed for lab usage and collaboration—drop in your HORIBA mapping TXT, get back clean `.npz` datasets and fit results that other scripts can consume immediately.

---

## Repository Structure

```
.
├── convert_HoribaMapTxt_to_StandardSpectrumMapNpz.py  # TXT → Standard NPZ converter
├── process_module_StandardSpectrumMapNpz.py           # Core preprocessing functions + IO
├── process_main.py                                    # Batch preprocessing pipeline (driver)
├── fit_module_StandardSpectrumMapNpz.py               # pV-unified fitter & schema (FitSpectrumMap.v1)
├── fit_main.py                                        # Batch fitting pipeline (driver)
└── StandardSpectrumMap_spec.md                        # Minimal data/NPZ spec (for reuse)
```

---

## Quick Start

1) **Prepare NPZ from HORIBA TXT**

```bash
python convert_HoribaMapTxt_to_StandardSpectrumMapNpz.py
# -> ./npz_out/<basename>.npz
```

2) **Run preprocessing (despike → ROI/interp → smooth → normalize → baseline)**

```bash
python process_main.py
# -> ./npz_processed/*.npz    (processed StandardSpectrumMap)
# -> ./qa_images/*.png        (before/after QA grids)
```

3) **Run fitting (staged, pV-unified)**

```bash
python fit_main.py
# -> ./result/<run_tag>/...__FitSpectrumMap_v1.npz  (always saved)
# -> ./result/<run_tag>/plots/*.png                 (initial/final fit, loss curve)
# -> optional CSV/TXT/Origin CSV if save_extras=True
```

> Tip: Both drivers are pre-populated with example file lists and defaults. Edit at the top of each `*_main.py` to point to your data and tune options.

---

## Data Spec (Summary)

We standardize mapping data in an in-memory container and an `.npz` schema so **anyone in the lab** can load/process/fit data consistently.

```python
@dataclass
class StandardSpectrumMap:
    spectra: np.ndarray  # (N, M) float64
    xy:      np.ndarray  # (N, 2) float64
    axis:    np.ndarray  # (M,)   float64
    unit:    str = ""    # e.g., "cm^-1"
```
**NPZ keys**
- `spectra (N,M) float64`, `xy (N,2) float64`, `axis (M,) float64`, optional `unit (np.array(str))`

See the full rationale in `StandardSpectrumMap_spec.md`.

---

## Scripts & How To

### 1) `convert_HoribaMapTxt_to_StandardSpectrumMapNpz.py`
**What it does**
- Parses HORIBA mapping TXT in two common layouts:
  - **Wide-table:** header row is numeric axis; subsequent rows: `X Y I1 I2 ...`
  - **2-line pairs:** one line of spectrum, next line `X Y ...` (fallback)
- Builds a `StandardSpectrumMap` and saves to NPZ (`./npz_out/`).

**Usage**
- Edit the `txt_files` list at the bottom or import the helper in your own script:
  ```python
  from convert_HoribaMapTxt_to_StandardSpectrumMapNpz import load_HoribaMapTxt_toStandardSpectrumMap, save_StandardSpectrumMap_toNpz
  obj = load_HoribaMapTxt_toStandardSpectrumMap("my.txt", unit_hint="cm^-1")
  save_StandardSpectrumMap_toNpz(obj, "my.npz")
  ```

---

### 2) `process_module_StandardSpectrumMapNpz.py` (library)
**Key APIs**
- **IO**
  - `load_StandardSpectrumMapNpz(path) -> StandardSpectrumMap`
  - `save_StandardSpectrumMap_toNpz(obj, path, compressed=True)`
- **Despike**
  - `process_Despike(obj, mode="simple"|"wh"|"none", sf_threshold=…, wh_threshold=…, wh_m=…)`
    - `simple`: difference threshold + local linear interpolation
    - `wh`: Whitaker–Hayes (modified Z on `diff(y)`), windowed replacement
- **ROI / Interpolate**
  - `process_Roi_Interpolate(obj, xmin, xmax, xstep, interpolate=True, round_decimals=None)`
    - `interpolate=True` builds a new uniform axis; `False` slices existing samples.
    - `round_decimals`: **round intensities** (e.g., `2` → 0.01 precision).
- **Smoothing**
  - `process_SmoothSavGol(obj, window_length, polyorder=3, mode="interp", strength=1)`
- **Normalize**
  - `process_Normalize(obj, mode="global"|"instance", round_decimals=None)`
    - All rows are min-shifted to 0; then scaled either by a global max or per-row max.
    - `round_decimals` also supported here.
- **Baseline**
  - `process_Baseline(obj, mode="none"|"original"|"peakstrip", ...)`
    - `original`: min-in-window + linear connect
    - `peakstrip`: iterative peak-stripping with Savitzky–Golay

- **QA figures**
  - `save_QA_SpectrumGrid(before, after, out_png, yscale="linear"|"log")`

---

### 3) `process_main.py` (driver)
A reproducible **batch pipeline** you can tune at the top of the file:

- Switches: enable/disable despike, ROI/interp, smoothing, normalize, baseline
- Parameters:
  - Despike mode: `"simple"` / `"wh"` / `"none"`
  - ROI/interp: `xmin, xmax, xstep, interpolate`
  - **Rounding**: `roi_round_decimals`, `norm_round_decimals` (set to `2` for 0.01)
  - Smoothing: `sg_window, sg_polyorder, sg_mode, sg_strength`
  - Normalize: `"global"` or `"instance"`
  - Baseline: original/peakstrip with window/step/polynomial controls

**Outputs**
- `./npz_processed/*.npz` with tags embedded in the filename (e.g., `interp_600_3600_4_r3`, `normI_r3`, `blpeakstrip`)
- `./qa_images/*.png` before/after grids for each step

---

### 4) `fit_module_StandardSpectrumMapNpz.py` (library)
A GPU-friendly, **pseudovoigt-unified** fitter for multi-peak deconvolution:
- Models Gaussian/Lorentzian/pseudo-Voigt in a single kernel; `eta∈[0,1]`
- Stage-wise training: pass a list of `StageSpec(name, iters, lr, active)`
- Parameter limits/penalties for physically reasonable fits
- Returns fit parameters and can **reconstruct** spectra on any axis
- Saves/loads a unified **FitSpectrumMap.v1** NPZ schema (see below)

**FitSpectrumMap.v1 schema** (NPZ keys)
- `axis, unit, xy, spectra_original`
- `peak_types` (display-ready), `params_pos`, `params_width`, `params_height`, `params_eta`, `params_base`
- `valid_mask`, `metadata_json`
- Optional `recon` (N×L) and `peaks_3d` (N×P×L)

---

### 5) `fit_main.py` (driver)
Turn the processed NPZ into staged fits with clean outputs.

**Key controls**
- File selection, random subset, or explicit indices
- `x_fit_range=(xmin,xmax)` is **clamped** to data bounds internally
- Stages: freely define any number of `StageSpec`s with different active peaks
- Device control: `use_gpu` / `force_cpu`; logs the device at runtime
- Preview: `show_initial_preview=True` shows an initial guess figure;
  - **Enter** → continue; **Esc** → abort the entire run
- Saves figures for initial/final fits and **loss vs. iteration**
- Reconstructs and stores the **fit on the full original axis**

**Outputs**
- Always writes `...__FitSpectrumMap_v1.npz` into `./result/<run_tag>/`
- Optional extras (`save_extras=True`): per-index CSV of peak params (with areas≈heights), Origin-ready CSV, TXT exports

---

## Typical End-to-End Flow

1. Convert: `TXT → standard .npz`
2. Process: despike → ROI/interp (uniform axis) → smooth → normalize → baseline
3. Fit: define peak set and stage plan; run; inspect plots and loss curves
4. Use: downstream notebooks can load both **StandardSpectrumMap** and **FitSpectrumMap** NPZs

---

## Installation & Requirements

- Python 3.10+ recommended
- NumPy, SciPy, Matplotlib
- PyTorch (for GPU-accelerated fitting; CPU fallback available)

Install with:

```bash
pip install numpy scipy matplotlib torch
```

---

## FAQ

**Q. How do I round intensities to 0.01?**  
Set `round_decimals=2` in ROI or Normalize steps.

**Q. What if my axis is unsorted or has duplicates?**  
The helpers sort and merge duplicates (mean) internally before processing.

**Q. Do I have to use GPU?**  
No. If CUDA is unavailable or disabled, the fitter runs on CPU and logs the device.

---
