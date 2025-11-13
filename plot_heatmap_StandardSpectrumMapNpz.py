# plot_StandardSpectrumMap_heatmap.py
# - StandardSpectrumMap NPZ 목록 -> XY heatmap 이미지 저장 (배치)
# - 모든 옵션은 아래 CONFIG 블록에서 직접 지정 (CLI 미사용)

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Tuple, Dict, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
try:
    from scipy.interpolate import griddata
except Exception:
    griddata = None  # scipy 미존재 시 불규칙 좌표 보간 불가

# =========================
# CONFIG (상단 배치)
# =========================
@dataclass
class Config:
    # 0) 입력: NPZ 파일 경로의 리스트 (예시 수정)
    npz_paths: Sequence[str] = (
        "./test1.npz",
        "./test2.npz",
    )

    # 1) x-axis ROI (None이면 전체)
    roi_xmin: Optional[float] = 460.0
    roi_xmax: Optional[float] = 490.0

    # 2) 집계 방식: 'sum'|'mean'|'max'
    aggregate: Literal["sum", "mean", "max"] = "sum"

    # 3) 컬러맵 이름 (CMRmap 기본)
    cmap_name: str = "CMRmap"

    # 4) 컬러바 스케일: auto면 vmin/vmax 자동. 수동이면 아래 값 사용
    color_auto: bool = True
    vmin: Optional[float] = None
    vmax: Optional[float] = None

    # 5) 좌표 격자 판정
    assume_grid: Optional[bool] = None      # True/False/None(auto)
    coord_round_decimals: int = 6

    # 6) 불규칙 샘플 보간용 격자 크기 (scipy 필요)
    fallback_grid_nx: Optional[int] = None  # None이면 자동(≈64)
    fallback_grid_ny: Optional[int] = None

    # 7) 출력/표시
    output_dir: str = "./result"
    figsize: Tuple[float, float] = (7.5, 6.0)
    dpi: int = 220
    show_plot: bool = False   # 배치일 때 기본 False 권장

CONFIG = Config()
# =========================


# =========================
# StandardSpectrumMap 구조
# =========================
@dataclass
class StandardSpectrumMap:
    spectra: np.ndarray   # (N, M)
    xy: np.ndarray        # (N, 2)
    axis: np.ndarray      # (M,)
    unit: str = ""

    def validate(self) -> None:
        if self.spectra.ndim != 2:
            raise ValueError("spectra must be 2D (N,M).")
        N, M = self.spectra.shape
        if self.xy.shape != (N, 2):
            raise ValueError(f"xy must be (N,2); got {self.xy.shape}, N={N}")
        if self.axis.shape != (M,):
            raise ValueError(f"axis must be (M,); got {self.axis.shape}, M={M}")
        if not np.isfinite(self.axis).all():
            raise ValueError("axis contains non-finite values.")
        if not np.isfinite(self.xy).all():
            raise ValueError("xy contains non-finite values.")

def load_StandardSpectrumMapNpz(path: str) -> StandardSpectrumMap:
    z = np.load(path, allow_pickle=False)
    spectra = z["spectra"].astype(np.float64, copy=False)
    xy = z["xy"].astype(np.float64, copy=False)
    axis = z["axis"].astype(np.float64, copy=False)
    unit = ""
    if "unit" in z.files:
        try:
            unit = str(z["unit"])
        except Exception:
            unit = ""
    obj = StandardSpectrumMap(spectra=spectra, xy=xy, axis=axis, unit=unit)
    obj.validate()
    return obj

# =========================
# 컬러맵 (수제 맵 제거, 기본: CMRmap)
# =========================
def _cmap(name: str) -> Colormap:
    try:
        return plt.get_cmap(name)
    except Exception:
        return plt.get_cmap("CMRmap")

CMAP_REGISTRY: Dict[str, Colormap] = {
    "CMRmap": _cmap("CMRmap"),
    "turbo": _cmap("turbo"),
    "viridis": _cmap("viridis"),
    "plasma": _cmap("plasma"),
    "magma": _cmap("magma"),
    "inferno": _cmap("inferno"),
    "cividis": _cmap("cividis"),
    "jet": _cmap("jet"),
}

# =========================
# 유틸
# =========================
def _roi_mask(axis: np.ndarray, xmin: Optional[float], xmax: Optional[float]) -> np.ndarray:
    if xmin is None and xmax is None:
        return np.isfinite(axis)
    a_min, a_max = float(np.nanmin(axis)), float(np.nanmax(axis))
    lo = a_min if xmin is None else max(float(xmin), a_min)
    hi = a_max if xmax is None else min(float(xmax), a_max)
    if lo > hi:
        lo, hi = hi, lo
    return (axis >= lo) & (axis <= hi) & np.isfinite(axis)

def _aggregate(y: np.ndarray, mode: str) -> float:
    if mode == "sum":
        return float(np.nansum(y))
    if mode == "mean":
        return float(np.nanmean(y))
    if mode == "max":
        return float(np.nanmax(y))
    raise ValueError("aggregate must be 'sum'|'mean'|'max'.")

def _infer_grid(xy: np.ndarray, assume_grid: Optional[bool], decimals: int):
    pts = np.round(xy, decimals=decimals)
    xs = np.unique(pts[:, 0])
    ys = np.unique(pts[:, 1])
    nx, ny = xs.size, ys.size
    grid_like = (nx * ny == pts.shape[0])
    if assume_grid is True and not grid_like:
        grid_like = False
    if assume_grid is False:
        grid_like = False
    if not grid_like:
        return None
    x2i = {float(v): i for i, v in enumerate(xs)}
    y2i = {float(v): i for i, v in enumerate(ys)}
    indexer = np.empty((pts.shape[0], 2), dtype=int)
    for k, (x, y) in enumerate(pts):
        indexer[k, 0] = x2i[float(x)]
        indexer[k, 1] = y2i[float(y)]
    return nx, ny, xs, ys, indexer

def _build_grid_from_scatter(xy: np.ndarray,
                             values: np.ndarray,
                             nx: Optional[int], ny: Optional[int]):
    if griddata is None:
        raise RuntimeError("scipy가 없어 불규칙 좌표 보간을 사용할 수 없습니다.")
    xs = np.linspace(np.nanmin(xy[:, 0]), np.nanmax(xy[:, 0]), num=nx or 64)
    ys = np.linspace(np.nanmin(xy[:, 1]), np.nanmax(xy[:, 1]), num=ny or 64)
    Xg, Yg = np.meshgrid(xs, ys)
    Vg = griddata(points=xy, values=values, xi=(Xg, Yg), method="nearest")
    return xs, ys, Vg  # xs(nx,), ys(ny,), Vg(ny,nx)

def compute_heat_value_per_point(ssm: StandardSpectrumMap,
                                 roi_mask: np.ndarray,
                                 aggregate: str) -> np.ndarray:
    Y = ssm.spectra
    vals = np.full((Y.shape[0],), np.nan, dtype=np.float64)
    for i in range(Y.shape[0]):
        yi = Y[i, roi_mask]
        if np.isfinite(yi).any():
            vals[i] = _aggregate(yi, aggregate)
    return vals

def _format_number_for_fname(x: float) -> str:
    # 파일명에 쓰기 좋게: 소수점 불필요 0 제거, '.' -> 'p'
    if np.isfinite(x):
        s = f"{x:.6g}"  # 유효자리 6
        return s.replace(".", "p")
    return "nan"

def _roi_string_for_file(ssm: StandardSpectrumMap, xmin: Optional[float], xmax: Optional[float]) -> str:
    if xmin is None and xmax is None:
        return "full"
    a_min, a_max = float(np.nanmin(ssm.axis)), float(np.nanmax(ssm.axis))
    lo = a_min if xmin is None else max(float(xmin), a_min)
    hi = a_max if xmax is None else min(float(xmax), a_max)
    if lo > hi:
        lo, hi = hi, lo
    return f"{_format_number_for_fname(lo)}-{_format_number_for_fname(hi)}"

def _derive_output_path(cfg: Config, ssm: StandardSpectrumMap, input_path: Union[str, Path]) -> Path:
    in_name = Path(input_path).name  # 파일명만
    # 확장자 제거
    base = in_name[:-4] if in_name.lower().endswith(".npz") else Path(in_name).stem
    roi_str = _roi_string_for_file(ssm, cfg.roi_xmin, cfg.roi_xmax)
    out_name = f"heatmap_{roi_str}_{base}.png"
    return Path(cfg.output_dir) / out_name

# =========================
# 메인 플로팅(단일 파일)
# =========================
def plot_heatmap_single(ssm: StandardSpectrumMap, cfg: Config, out_path: Path) -> None:
    # ROI
    m = _roi_mask(ssm.axis, cfg.roi_xmin, cfg.roi_xmax)
    if not np.any(m):
        raise ValueError("ROI에 해당하는 축 샘플이 없습니다. (axis와 ROI 범위 확인)")

    # 포인트별 집계값
    heat_vals = compute_heat_value_per_point(ssm, m, cfg.aggregate)

    # 격자/보간
    grid_info = _infer_grid(ssm.xy, cfg.assume_grid, cfg.coord_round_decimals)
    if grid_info is not None:
        nx, ny, xs, ys, indexer = grid_info
        Z = np.full((ny, nx), np.nan, dtype=np.float64)
        for (ix, iy), v in zip(indexer, heat_vals):
            Z[iy, ix] = v
        Xs, Ys = xs, ys
    else:
        Xs, Ys, Z = _build_grid_from_scatter(ssm.xy, heat_vals, cfg.fallback_grid_nx, cfg.fallback_grid_ny)

    # 컬러맵/스케일
    cmap = CMAP_REGISTRY.get(cfg.cmap_name, CMAP_REGISTRY["CMRmap"])
    if cfg.color_auto:
        finite = np.isfinite(Z)
        vmin = np.nanmin(Z[finite]) if finite.any() else None
        vmax = np.nanmax(Z[finite]) if finite.any() else None
    else:
        vmin, vmax = cfg.vmin, cfg.vmax

    # 플롯
    fig, ax = plt.subplots(figsize=cfg.figsize)
    extent = [float(np.min(Xs)), float(np.max(Xs)), float(np.min(Ys)), float(np.max(Ys))]
    im = ax.imshow(
        Z, extent=extent, origin="lower", aspect="auto",
        cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
    )

    unit = f" ({ssm.unit})" if ssm.unit else ""
    if cfg.roi_xmin is not None or cfg.roi_xmax is not None:
        lo = cfg.roi_xmin if cfg.roi_xmin is not None else ssm.axis.min()
        hi = cfg.roi_xmax if cfg.roi_xmax is not None else ssm.axis.max()
        roi_str = f" | ROI x∈[{lo:g}, {hi:g}]{unit}"
    else:
        roi_str = ""

    ax.set_xlabel("Stage X")
    ax.set_ylabel("Stage Y")
    ax.set_title(f"Integrated intensity heatmap{roi_str}")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"{cfg.aggregate} intensity (a.u.)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=cfg.dpi)

    if cfg.show_plot:
        plt.show()
    else:
        plt.close(fig)

# =========================
# 실행 (배치)
# =========================
def main(cfg: Config) -> None:
    # 문자열 단일 입력도 허용
    paths: Sequence[str]
    if isinstance(cfg.npz_paths, (str, Path)):
        paths = [str(cfg.npz_paths)]
    else:
        paths = list(cfg.npz_paths)

    for p in paths:
        ssm = load_StandardSpectrumMapNpz(p)
        out_path = _derive_output_path(cfg, ssm, p)
        plot_heatmap_single(ssm, cfg, out_path)
        print(f"[saved] {out_path}")

if __name__ == "__main__":
    main(CONFIG)
