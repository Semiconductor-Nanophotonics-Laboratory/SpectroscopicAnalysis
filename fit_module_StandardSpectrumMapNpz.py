# fit_module_StandardSpectrumMapNpz.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# =============================================================================
# Dataclasses (hyper-params, stage plan, peak specs, fit config)
# =============================================================================

@dataclass(frozen=True)
class FitHyperParams:
    # (레거시 백워드 호환) – stages가 없을 때만 사용
    num_iter_stage1: int = 2000
    num_iter_stage2: int = 2000
    lr_stage1: float = 0.015
    lr_stage2: float = 0.002
    height_sign_reg: float = 5.0
    pos_var_reg: float = 0.0
    width_var_reg: float = 0.0
    height_var_reg: float = 0.0

@dataclass(frozen=True)
class StageSpec:
    """
    name   : 로그 표시에 쓸 스테이지 이름
    iters  : 반복 회수
    lr     : 학습률
    active : 'all' 또는 [피크 인덱스들], 예: [0,1,5]  (해당 피크만 기여/업데이트)
    """
    name: str
    iters: int
    lr: float
    active: Union[str, List[int]] = "all"  # "all" | list of ints

@dataclass(frozen=True)
class PeakSpec:
    # type: 'gaussian' | 'lorentzian'('cauchy') | 'pseudovoigt'
    type: str
    pos: float
    width: float
    height: float
    pos_limit: float
    pos_limit_reg: float
    width_limit: float
    width_limit_reg: float
    height_limit: float
    height_limit_reg: float
    extra: Optional[dict] = None  # {'eta':0~1} for pV init

@dataclass(frozen=True)
class FitConfig:
    peaks: List[PeakSpec]
    sort_peaks_by_pos: bool = True
    # (레거시) stages 미사용 시에만 참조
    stage1: Union[str, List[int]] = "all"
    stage2: Union[str, List[int]] = "all"
    # (신규) 자유 Stage 스케줄
    stages: Optional[List[StageSpec]] = None

    base_init: float = 0.0
    use_gpu: bool = True
    # x_fit_range: (xmin, xmax) — 각 항목 None 허용. 모듈이 데이터 경계로 클램프.
    x_fit_range: Optional[Tuple[Optional[float], Optional[float]]] = None
    x_label: str = "x"
    y_label: str = "Intensity"
    h: FitHyperParams = FitHyperParams()
    basic_three_only: bool = True  # True 권장


# =============================================================================
# StandardSpectrumMap loader (thin wrapper expected by main)
# =============================================================================

@dataclass(frozen=True)
class StandardSpectrumMap:
    axis: np.ndarray      # (L,)
    unit: str             # e.g., "cm^-1"
    xy: np.ndarray        # (N,2)
    spectra: np.ndarray   # (N,L)

def load_StandardSpectrumMapNpz(path: str) -> StandardSpectrumMap:
    z = np.load(path, allow_pickle=True)
    axis = z["axis"]
    unit = str(z["unit"].item()) if "unit" in z else ""
    xy = z["xy"]
    spectra = z["spectra"]
    return StandardSpectrumMap(axis=axis, unit=unit, xy=xy, spectra=spectra)


# =============================================================================
# Utilities: range crop with clamp, selector, text/csv exporters
# =============================================================================

def crop_by_range(x: np.ndarray,
                  y: np.ndarray,
                  rng: Optional[Tuple[Optional[float], Optional[float]]]):
    """
    rng = (xmin, xmax)  # 각 항목 None 가능
    규칙:
      - xmin/xmax 중 지정된 것만 적용 (개별 판단)
      - 입력 데이터 경계(x.min, x.max)로 자동 클램프
      - 지정 구간과 데이터 구간의 교집합이 비면: 전체 구간으로 폴백
    """
    if rng is None:
        xx, yy = x, y
    else:
        xmin_req = None if rng[0] is None else float(rng[0])
        xmax_req = None if rng[1] is None else float(rng[1])

        x_min = float(np.nanmin(x))
        x_max = float(np.nanmax(x))

        lb = x_min if xmin_req is None else max(xmin_req, x_min)
        ub = x_max if xmax_req is None else min(xmax_req, x_max)

        if not (lb <= ub):
            lb, ub = x_min, x_max

        m = (x >= lb) & (x <= ub) & np.isfinite(y)
        xx, yy = x[m], y[m]
        if xx.size == 0:
            xx, yy = x, y

    if xx.size and np.any(np.diff(xx) < 0):
        idx = np.argsort(xx)
        xx, yy = xx[idx], yy[idx]
    return xx, yy


def prepare_xy_from_npz(obj: StandardSpectrumMap,
                        idx: int,
                        x_fit_range: Optional[Tuple[Optional[float], Optional[float]]]):
    x_full = obj.axis.astype(float, copy=False)
    y_full = obj.spectra[idx].astype(float, copy=False)
    coord = obj.xy[idx].astype(float, copy=False)
    x, y = crop_by_range(x_full, y_full, x_fit_range)
    return x, y.reshape(1, -1), coord.reshape(1, 2)


def select_indices(n_total: int, mode: str,
                   indices: List[int] | Tuple[int, ...] = (),
                   k_random: int = 5, seed: int = 42) -> List[int]:
    if mode == "one":
        return [0] if n_total > 0 else []
    if mode == "all":
        return list(range(n_total))
    if mode == "indices":
        return [i for i in indices if 0 <= i < n_total]
    if mode == "random":
        rng = np.random.default_rng(seed)
        k = min(k_random, n_total)
        return sorted(rng.choice(n_total, size=k, replace=False).tolist())
    return []


def export_text_spectra(path: str, x: np.ndarray, y: np.ndarray, coord_xy: np.ndarray | list | tuple):
    """Origin 등에서 바로 읽을 수 있는 탭-구분 텍스트 저장."""
    coord_xy = np.asarray(coord_xy).reshape(-1)
    coord = "\t".join(f"{float(c):.6f}" for c in coord_xy[:2]) if coord_xy.size >= 2 else ""
    with open(path, "w", encoding="utf-8") as f:
        if coord:
            f.write(f"# coord_x\tcoord_y\n{coord}\n")
        f.write("# x\ty\n")
        for xi, yi in zip(x, y.squeeze()):
            f.write(f"{xi:.9f}\t{yi:.9f}\n")


def save_origin_ready_csv(outdir: str, stem: str, x: np.ndarray,
                          y: np.ndarray, peaks: np.ndarray, baseline: float) -> str:
    """
    Origin에서 바로 쓸 수 있게 멀티컬럼 CSV 생성:
    col0=x, col1=data, col2=recon(sum), col3=baseline, col4~: peak_j
    """
    out_path = f"{outdir}/{stem}_origin.csv"
    x = x.reshape(-1)
    data = y.reshape(-1)
    recon = peaks.sum(axis=1).reshape(-1)  # (L,)
    base = np.full_like(x, float(baseline))
    cols = [x, data, recon, base]
    for j in range(peaks.shape[0]):  # peaks: (P,L) 입맛에 맞게 변환
        cols.append(peaks[j, :])

    mat = np.column_stack(cols)
    header = ["x", "data", "recon", "baseline"] + [f"peak_{j+1}" for j in range(peaks.shape[0])]
    np.savetxt(out_path, mat, delimiter=",", header=",".join(header), comments="", fmt="%.9g")
    return out_path


# =============================================================================
# pV unified synthesis (Gaussian, Lorentzian normalized kernels)
# =============================================================================

def _gaussian_norm(x: torch.Tensor, mu: torch.Tensor, fwhm: torch.Tensor) -> torch.Tensor:
    sigma = torch.clamp(fwhm, min=1e-6) / 2.3548200450309493
    z = (x - mu) / sigma
    G = torch.exp(-0.5 * z * z) / (sigma * (torch.sqrt(torch.tensor(2.0 * np.pi, dtype=x.dtype, device=x.device))))
    return G  # integral 1

def _lorentz_norm(x: torch.Tensor, mu: torch.Tensor, fwhm: torch.Tensor) -> torch.Tensor:
    gamma = torch.clamp(fwhm, min=1e-6) / 2.0
    L = (1.0 / np.pi) * (gamma / ((x - mu) ** 2 + gamma ** 2))
    return L  # integral 1

def synthesize_pv(x: torch.Tensor, pos: torch.Tensor, width: torch.Tensor, height: torch.Tensor,
                  base: torch.Tensor, eta: torch.Tensor, active: List[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    x        : (B,L)
    pos      : (B,P,1)
    width    : (B,P,1)
    height   : (B,P,1)
    base     : (B,L)
    eta      : (B,P,1)   (0: pure Gaussian, 1: pure Lorentzian)
    active   : list of peak indices used in mixture
    returns:
      recon: (B,L)
      peaks: (B,P,L)
    """
    B, L = x.shape
    P = pos.shape[1]
    x_ = x[:, None, :]
    mu = pos
    w = torch.clamp(width, min=1e-6)
    h = height
    et = torch.clamp(eta, 0.0, 1.0)

    G = _gaussian_norm(x_, mu, w)   # (B,P,L)
    Lz = _lorentz_norm(x_, mu, w)   # (B,P,L)
    mixture = et * Lz + (1.0 - et) * G
    peaks = h * mixture             # (B,P,L)

    mask = torch.zeros(P, dtype=peaks.dtype, device=peaks.device)
    if len(active) > 0:
        mask[torch.tensor(active, device=mask.device, dtype=torch.long)] = 1.0
    peaks = peaks * mask.view(1, P, 1)
    recon = base + peaks.sum(dim=1)
    return recon, peaks


# =============================================================================
# Torch args builder & fitter
# =============================================================================

def build_torch_args(cfg: FitConfig, x_np: np.ndarray):
    x = torch.from_numpy(x_np.astype(np.float32, copy=False)).unsqueeze(0)  # (1,L)

    peaks = cfg.peaks
    if cfg.sort_peaks_by_pos:
        peaks = sorted(peaks, key=lambda p: p.pos)

    P = len(peaks)

    pos0    = torch.tensor([[p.pos for p in peaks]], dtype=torch.float32).unsqueeze(-1)    # (1,P,1)
    width0  = torch.tensor([[p.width for p in peaks]], dtype=torch.float32).unsqueeze(-1)  # (1,P,1)
    height0 = torch.tensor([[p.height for p in peaks]], dtype=torch.float32).unsqueeze(-1) # (1,P,1)

    eta_init = []
    display_types = []
    for p in peaks:
        t = p.type.lower()
        if t in ("gaussian",):
            e0 = 0.0; display_types.append("gaussian")
        elif t in ("lorentzian", "cauchy"):
            e0 = 1.0; display_types.append("lorentzian")
        elif t in ("pseudovoigt", "pv", "pvoigt"):
            e0 = float(p.extra.get("eta", 0.5) if p.extra else 0.5); display_types.append("pseudovoigt")
        else:
            e0 = float(p.extra.get("eta", 0.5) if p.extra else 0.5); display_types.append(f"{t}->pV")
        eta_init.append(e0)
    eta0 = torch.tensor([[e for e in eta_init]], dtype=torch.float32).unsqueeze(-1)

    base0 = torch.tensor([[cfg.base_init]], dtype=torch.float32)

    pos_lim    = torch.tensor([[p.pos_limit for p in peaks]], dtype=torch.float32).unsqueeze(-1)
    pos_lreg   = torch.tensor([[p.pos_limit_reg for p in peaks]], dtype=torch.float32).unsqueeze(-1)
    width_lim  = torch.tensor([[p.width_limit for p in peaks]], dtype=torch.float32).unsqueeze(-1)
    width_lreg = torch.tensor([[p.width_limit_reg for p in peaks]], dtype=torch.float32).unsqueeze(-1)
    height_lim = torch.tensor([[p.height_limit for p in peaks]], dtype=torch.float32).unsqueeze(-1)
    height_lreg= torch.tensor([[p.height_limit_reg for p in peaks]], dtype=torch.float32).unsqueeze(-1)

    return {
        "x": x,
        "pos0": pos0, "width0": width0, "height0": height0, "eta0": eta0, "base0": base0,
        "pos_lim": pos_lim, "pos_lreg": pos_lreg,
        "width_lim": width_lim, "width_lreg": width_lreg,
        "height_lim": height_lim, "height_lreg": height_lreg,
        "display_types": display_types,
    }


class SpectrumFitter:
    def __init__(self, targs, x_np: np.ndarray, y_np: np.ndarray, use_gpu: bool = True):
        self.device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

        self.x = targs["x"].to(self.device)                 # (1,L)
        self.y = torch.from_numpy(y_np.astype(np.float32)).to(self.device)  # (1,L)

        # trainable
        self.pos    = nn.Parameter(targs["pos0"].to(self.device))     # (1,P,1)
        self.width  = nn.Parameter(targs["width0"].to(self.device))   # (1,P,1)
        self.height = nn.Parameter(targs["height0"].to(self.device))  # (1,P,1)
        self.eta    = nn.Parameter(targs["eta0"].to(self.device))     # (1,P,1)
        self.base   = nn.Parameter(targs["base0"].to(self.device))    # (1,1)

        # limits/regs
        self.pos0 = targs["pos0"].to(self.device)
        self.width0 = targs["width0"].to(self.device)
        self.height0 = targs["height0"].to(self.device)

        self.pos_lim = targs["pos_lim"].to(self.device)
        self.pos_lreg = targs["pos_lreg"].to(self.device)
        self.width_lim = targs["width_lim"].to(self.device)
        self.width_lreg = targs["width_lreg"].to(self.device)
        self.height_lim = targs["height_lim"].to(self.device)
        self.height_lreg = targs["height_lreg"].to(self.device)

        self.display_types = targs["display_types"]
        self.num_peaks = self.pos.shape[1]

        self.loss_history: List[float] = []

    def forward(self, active: Union[str, List[int]] = "all"):
        if isinstance(active, str) and active == "all":
            active_idx = list(range(self.num_peaks))
        else:
            active_idx = list(active) if isinstance(active, (list, tuple)) else []
        recon, peaks = synthesize_pv(
            self.x, self.pos, self.width, self.height,
            self.base.expand_as(self.x), self.eta, active=active_idx
        )
        return recon, peaks

    def _loss(self, recon):
        data_loss = torch.mean((recon - self.y) ** 2)

        pos_off    = torch.relu(torch.abs(self.pos - self.pos0)    - self.pos_lim)
        width_off  = torch.relu(torch.abs(self.width - self.width0)- self.width_lim)
        height_off = torch.relu(torch.abs(self.height - self.height0)- self.height_lim)

        limit_penalty = (pos_off * self.pos_lreg).mean() + (width_off * self.width_lreg).mean() + (height_off * self.height_lreg).mean()
        height_sign_penalty = torch.relu(-self.height).mean()

        loss = data_loss + limit_penalty + height_sign_penalty * 5.0
        return loss

    def fit_stage(self, iters: int, lr: float, stage_name: str, active: Union[str, List[int]] = "all"):
        if isinstance(active, str) and active == "all":
            act_disp = list(range(self.num_peaks))
        else:
            act_disp = list(active)

        logging.info(
            "fit_module_StandardSpectrumMapNpz: %s start | lr=%.3g, iters=%d, active=%s",
            stage_name, lr, iters, act_disp
        )
        opt = optim.Adam([self.pos, self.width, self.height, self.eta, self.base], lr=lr)
        self.loss_history.clear()
        for it in range(1, iters + 1):
            opt.zero_grad()
            recon, _ = self.forward(active=active)
            loss = self._loss(recon)
            loss.backward()
            opt.step()

            with torch.no_grad():
                self.eta.clamp_(0.0, 1.0)
                self.width.clamp_(min=1e-4)
                self.height.clamp_(min=-1e3)

            self.loss_history.append(float(loss.detach().cpu().item()))
        logging.info("fit_module_StandardSpectrumMapNpz: %s done.", stage_name)

    def fit(self, cfg: FitConfig):
        """
        cfg.stages 가 있으면 그 계획대로, 없으면 (레거시) stage1/2 + HYP로 수행
        """
        if cfg.stages and len(cfg.stages) > 0:
            for s in cfg.stages:
                self.fit_stage(s.iters, s.lr, s.name, s.active)
        else:
            # legacy 2-stage
            self.fit_stage(cfg.h.num_iter_stage1, cfg.h.lr_stage1, "Stage 1", cfg.stage1)
            self.fit_stage(cfg.h.num_iter_stage2, cfg.h.lr_stage2, "Stage 2", cfg.stage2)

    @property
    def h(self) -> FitHyperParams:
        # (호출부가 필요 시 참조) – 현재는 위 fit()에서 cfg.h를 직접 사용
        return FitHyperParams()

    def params(self):
        to_np = lambda t: t.detach().cpu().numpy()
        return {
            "pos": to_np(self.pos),           # (1,P,1)
            "width": to_np(self.width),       # (1,P,1)
            "height": to_np(self.height),     # (1,P,1)
            "eta": to_np(self.eta),           # (1,P,1)
            "base": to_np(self.base),         # (1,1)
            "display_types": np.array(self.display_types, dtype=object)
        }

    def reconstructed(self):
        with torch.no_grad():
            recon, peaks = self.forward(active="all")
        return recon.detach().cpu().numpy(), peaks.detach().cpu().numpy()

    def get_loss_history(self) -> np.ndarray:
        return np.asarray(self.loss_history, dtype=float)


# =============================================================================
# Figures
# =============================================================================

def _scalar(v):
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return float(v)
        return float(np.asarray(v).reshape(-1)[0])
    try:
        return float(v)
    except Exception:
        return float(np.asarray(v).reshape(-1)[0])


def fig_initial_guess(fitter: SpectrumFitter, cfg: FitConfig) -> plt.Figure:
    x = fitter.x.detach().cpu().numpy().squeeze()
    y = fitter.y.detach().cpu().numpy().squeeze()
    with torch.no_grad():
        recon, peaks = fitter.forward(active="all")
    recon = recon.detach().cpu().numpy().squeeze()
    peaks = peaks.detach().cpu().numpy().squeeze()  # (P,L)

    fig, ax = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 1]})
    ax[0].plot(x, y, "o-", ms=2, alpha=0.7, label="data")
    ax[0].plot(x, recon, "--", label="initial mix")
    ax[0].set_title("Initial guess")
    ax[0].set_ylabel(cfg.y_label); ax[0].legend(); ax[0].grid(True, ls=":", alpha=0.5)

    ax[1].plot(x, y, alpha=0.35, label="data")
    for j in range(fitter.num_peaks):
        ax[1].plot(x, peaks[j, :], label=f"peak {j+1} ({fitter.display_types[j]})")
    ax[1].set_xlabel(cfg.x_label); ax[1].set_ylabel(cfg.y_label)
    ax[1].legend(ncol=2, fontsize=9); ax[1].grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    return fig


def compute_peak_areas(params, display_types: List[str]) -> np.ndarray:
    # pV 커널이 적분 1이므로 면적 ≈ height
    h = params["height"]  # (B,P,1)
    return h[..., 0]      # (B,P)


def fig_final_fit(fitter: SpectrumFitter, cfg: FitConfig) -> plt.Figure:
    x = fitter.x.detach().cpu().numpy().squeeze()
    y = fitter.y.detach().cpu().numpy()
    recon, peaks = fitter.reconstructed()
    params = fitter.params()
    idx = 0

    pos    = np.array([_scalar(v) for v in params["pos"][0, :, 0]])
    width  = np.array([_scalar(v) for v in params["width"][0, :, 0]])
    height = np.array([_scalar(v) for v in params["height"][0, :, 0]])
    eta    = np.array([_scalar(v) for v in params["eta"][0, :, 0]])

    areas = compute_peak_areas(params, list(params["display_types"]))
    areas_row = areas[0, :]

    fig, ax = plt.subplots(3, 1, figsize=(12, 11), gridspec_kw={"height_ratios": [1, 1, 0.7]})

    ax[0].plot(x, y[idx], "o-", ms=2, alpha=0.7, label="data")
    ax[0].plot(x, recon[idx], "--", label="fit")
    ax[0].set_title("Final fit")
    ax[0].set_ylabel(cfg.y_label); ax[0].legend(); ax[0].grid(True, ls=":", alpha=0.5)

    ax[1].plot(x, y[idx], alpha=0.3)
    for j in range(fitter.num_peaks):
        ax[1].plot(x, peaks[idx, j, :], label=f"peak {j+1} ({fitter.display_types[j]}, η={eta[j]:.3f})")
        ax[1].axvline(pos[j], ls=":", alpha=0.4)
    ax[1].set_xlabel(cfg.x_label); ax[1].set_ylabel(cfg.y_label)
    ax[1].legend(ncol=2, fontsize=8); ax[1].grid(True, ls=":", alpha=0.5)

    ax[2].axis("off")
    table = [[j + 1,
              str(fitter.display_types[j]),
              f"{pos[j]:.3f}",
              f"{width[j]:.3f}",
              f"{height[j]:.4f}",
              f"{eta[j]:.4f}",
              f"{areas_row[j]:.6g}"]
             for j in range(fitter.num_peaks)]
    tbl = ax[2].table(
        cellText=table,
        colLabels=["#", "type", "pos", "width(FWHM)", "height", "eta", "area≈height"],
        loc="center"
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.15)
    fig.tight_layout()
    return fig


# =============================================================================
# FitSpectrumMap NPZ (unified schema v1)
# =============================================================================

from dataclasses import dataclass as _dataclass
import json
from datetime import datetime

@_dataclass(frozen=True)
class FitSpectrumMap:
    axis: np.ndarray
    unit: str
    xy: np.ndarray
    spectra_original: np.ndarray

    peak_types: list
    pos: np.ndarray
    width: np.ndarray
    height: np.ndarray
    eta: np.ndarray
    base: np.ndarray

    recon: np.ndarray | None
    peaks_3d: np.ndarray | None
    valid_mask: np.ndarray
    metadata: dict

    def reconstruct(self, idx: int) -> np.ndarray:
        x = torch.from_numpy(self.axis.astype(np.float64)).float().unsqueeze(0)  # (1,L)
        pos = torch.from_numpy(self.pos[idx:idx+1, :, None]).float()
        width = torch.from_numpy(self.width[idx:idx+1, :, None]).float()
        height = torch.from_numpy(self.height[idx:idx+1, :, None]).float()
        eta = torch.from_numpy(self.eta[idx:idx+1, :, None]).float()
        base = torch.from_numpy(self.base[idx:idx+1, None]).float()
        with torch.no_grad():
            rec, _ = synthesize_pv(x, pos, width, height, base.expand(1, x.shape[1]), eta, active=list(range(pos.shape[1])))
        return rec.squeeze(0).cpu().numpy()


def save_FitSpectrumMapNpz(
    path: str,
    *,
    axis: np.ndarray,
    unit: str,
    xy: np.ndarray,
    spectra_original: np.ndarray,
    peak_types: list,
    pos: np.ndarray,
    width: np.ndarray,
    height: np.ndarray,
    eta: np.ndarray,
    base: np.ndarray,
    valid_mask: np.ndarray,
    metadata: dict | None = None,
    recon: np.ndarray | None = None,
    peaks_3d: np.ndarray | None = None,
) -> None:
    meta = dict(metadata or {})
    meta.setdefault("schema", "FitSpectrumMap.v1")
    meta.setdefault("model", "pseudovoigt-unified")
    meta.setdefault("created_utc", datetime.utcnow().isoformat(timespec="seconds") + "Z")

    np.savez_compressed(
        path,
        __schema__=np.array("FitSpectrumMap.v1", dtype=object),
        axis=axis.astype(np.float64, copy=False),
        unit=np.array(unit, dtype=object),
        xy=xy.astype(np.float64, copy=False),
        spectra_original=spectra_original.astype(np.float32, copy=False),
        peak_types=np.array(list(peak_types), dtype=object),
        params_pos=pos.astype(np.float32, copy=False),
        params_width=width.astype(np.float32, copy=False),
        params_height=height.astype(np.float32, copy=False),
        params_eta=eta.astype(np.float32, copy=False),
        params_base=base.astype(np.float32, copy=False),
        valid_mask=valid_mask.astype(bool, copy=False),
        metadata_json=np.array(json.dumps(meta), dtype=object),
        recon=(recon.astype(np.float32, copy=False) if recon is not None else np.array([], dtype=np.float32)),
        peaks_3d=(peaks_3d.astype(np.float32, copy=False) if peaks_3d is not None else np.array([], dtype=np.float32)),
    )


def load_FitSpectrumMapNpz(path: str) -> FitSpectrumMap:
    z = np.load(path, allow_pickle=True)
    schema = str(z["__schema__"].item())
    if schema != "FitSpectrumMap.v1":
        raise ValueError(f"Unsupported schema: {schema}")

    axis = z["axis"]
    unit = str(z["unit"].item())
    xy = z["xy"]
    spectra_original = z["spectra_original"]
    peak_types = list(z["peak_types"])
    pos = z["params_pos"]
    width = z["params_width"]
    height = z["params_height"]
    eta = z["params_eta"]
    base = z["params_base"]
    valid_mask = z["valid_mask"]
    metadata = json.loads(str(z["metadata_json"].item()))

    recon = z["recon"];  recon = None if recon.size == 0 else recon
    peaks_3d = z["peaks_3d"]; peaks_3d = None if peaks_3d.size == 0 else peaks_3d

    return FitSpectrumMap(
        axis=axis, unit=unit, xy=xy, spectra_original=spectra_original,
        peak_types=peak_types, pos=pos, width=width, height=height, eta=eta, base=base,
        recon=recon, peaks_3d=peaks_3d, valid_mask=valid_mask, metadata=metadata
    )
