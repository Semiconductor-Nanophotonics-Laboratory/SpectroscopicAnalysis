# plot_heatmap_AndorSpectrumMapSif.py
# Andor SIF → (in-memory) StandardSpectrumMap → Heatmap PNG
# - 중간 NPZ 저장 없음
# - 불일치/이상 시: 상세 사유 출력 후 해당 파일 건너뛰기 (임의 보간/임의 격자 금지)

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Tuple, Dict, Sequence, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    # 입력 SIF 파일(1개 또는 여러개)
    sif_paths: Sequence[str] = (
        "./test1.sif",
        "./test2.sif",
        )

    # 스캔 격자/좌표(프레임 수 = Xgrid * Ygrid와 일치해야 함)
    Xsize: float = 12.0
    Ysize: float = 12.0
    Xgrid: int = 120
    Ygrid: int = 120

    # 프레임 전개 순서(장비 저장 순서와 일치시키기)
    frame_order: Literal["YX", "XY"] = "YX"   # "YX": y-outer, "XY": x-outer
    flip_x_order: bool = False                # True면 X 반복 방향을 좌↔우 반전

    # 좌표 미러링(데이터는 그대로, 좌표만 부호 반전)
    negate_x_coord: bool = False
    negate_y_coord: bool = False

    # 스펙트럼 축 단위 힌트
    unit_hint: str = "nm"  # "nm" / "cm^-1" / ...

    # (선택) 축 강제 지정: SIF에 파장축이 없을 때 사용
    #   None               : 자동 추출 시도 → 실패하면 픽셀 인덱스 0..M-1
    #   ("linear", start, step): start + step*np.arange(M)
    axis_override: Optional[Tuple[str, float, float]] = None

    # 집계 ROI 및 방식
    roi_xmin: Optional[float] = 460
    roi_xmax: Optional[float] = 490
    aggregate: Literal["sum", "mean", "max"] = "mean"

    # 플롯 옵션
    cmap_name: str = "CMRmap"
    color_auto: bool = True
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    assume_grid: bool = True                  # SIF 매핑이면 True (불규칙이면 스킵)
    coord_round_decimals: int = 6

    # 출력
    output_dir: str = "./result"
    figsize: Tuple[float, float] = (7.5, 6.0)
    dpi: int = 220
    show_plot: bool = False

CONFIG = Config()

# =============================================================================
# 데이터 구조 및 예외
# =============================================================================

class SkipFile(RuntimeError):
    """이 파일은 처리 스킵 (사유는 메시지로 상세 전달)."""

@dataclass
class StandardSpectrumMap:
    spectra: np.ndarray   # (N, M)
    xy: np.ndarray        # (N, 2)
    axis: np.ndarray      # (M,)
    unit: str = ""

    def validate(self) -> None:
        if self.spectra.ndim != 2:
            raise SkipFile("spectra가 2D(N,M)가 아님")
        N, M = self.spectra.shape
        if self.xy.shape != (N, 2):
            raise SkipFile(f"xy가 (N,2)가 아님: xy.shape={self.xy.shape}, N={N}")
        if self.axis.shape != (M,):
            raise SkipFile(f"axis가 (M,)가 아님: axis.shape={self.axis.shape}, M={M}")
        if not np.isfinite(self.axis).any():
            raise SkipFile("axis 유효값(유한값)이 없음")
        if not np.isfinite(self.xy).any():
            raise SkipFile("xy 좌표 유효값(유한값)이 없음")
        if not np.isfinite(self.spectra).any():
            raise SkipFile("스펙트럼 유효값(유한값)이 없음")

# =============================================================================
# Andor SIF 백엔드
# =============================================================================

_BACKEND = None
try:
    import sif_parser  # pip install sif_parser  (conda: sif-parser)
    _BACKEND = "sif_parser"
except Exception:
    try:
        import sifreader  # pip install sifreader
        _BACKEND = "sifreader"
    except Exception:
        _BACKEND = None

def _try_extract_axis_from_info(info) -> Optional[np.ndarray]:
    # 1) 공식 유틸 사용 가능 시
    try:
        from sif_parser.utils import extract_calibration  # type: ignore
        cal = extract_calibration(info)
        if cal is not None:
            if isinstance(cal, dict) and "wavelength" in cal and cal["wavelength"] is not None:
                wl = np.asarray(cal["wavelength"], dtype=float).reshape(-1)
                if wl.size > 0 and np.isfinite(wl).all():
                    return wl
            else:
                wl = np.asarray(cal, dtype=float).reshape(-1)
                if wl.size > 0 and np.isfinite(wl).all():
                    return wl
    except Exception:
        pass
    # 2) info에 직접 포함된 경우들
    try:
        if isinstance(info, dict):
            for k in ("wavelength", "Wavelength", "x_wavelength", "XAxis", "x_axis", "xaxis"):
                if k in info and info[k] is not None:
                    wl = np.asarray(info[k], dtype=float).reshape(-1)
                    if wl.size > 0 and np.isfinite(wl).all():
                        return wl
    except Exception:
        pass
    # 3) 하위 메타 구조 탐색
    try:
        if isinstance(info, dict):
            for subk in ("frameData", "frames", "metadata", "header"):
                if subk in info and info[subk] is not None:
                    obj = info[subk]
                    if isinstance(obj, (list, tuple)) and len(obj) > 0:
                        cand = obj[0]
                        if isinstance(cand, dict):
                            for kk in ("wavelength", "XAxis", "xaxis"):
                                if kk in cand and cand[kk] is not None:
                                    wl = np.asarray(cand[kk], dtype=float).reshape(-1)
                                    if wl.size > 0 and np.isfinite(wl).all():
                                        return wl
    except Exception:
        pass
    return None

def _read_sif_with_sif_parser(path: str) -> Tuple[np.ndarray, np.ndarray]:
    arr, info = sif_parser.np_open(path)  # arr: (frames,pixels) or (f,h,w)
    arr = np.asarray(arr, dtype=float)
    # (frames, M) 정규화
    if arr.ndim == 2:
        frames, M = arr.shape
    elif arr.ndim == 3:
        f, a, b = arr.shape
        if a == 1 and b > 1:
            arr = arr.reshape(f, b)
        elif b == 1 and a > 1:
            arr = arr.reshape(f, a)
        else:
            arr = arr.reshape(f, a * b)
        frames, M = arr.shape
    else:
        raise SkipFile(f"sif_parser: 예상치 못한 데이터 차원(ndim={arr.ndim})")

    wl = _try_extract_axis_from_info(info)
    if wl is None:
        wl = np.arange(M, dtype=float)  # 메타 결손 → 픽셀 인덱스 축
    wl = wl.astype(float, copy=False).reshape(-1)
    if wl.size != M:
        if wl.size >= 2:
            wl = np.linspace(float(wl[0]), float(wl[-1]), num=M, dtype=float)
        else:
            wl = np.arange(M, dtype=float)
    return arr.astype(np.float64, copy=False), wl.astype(np.float64, copy=False)

def _read_sif_with_sifreader(path: str) -> Tuple[np.ndarray, np.ndarray]:
    from sifreader import SifReader
    sr = SifReader(path)
    arr = np.asarray(sr.get_data(), dtype=float)
    if arr.ndim == 3:
        f, a, b = arr.shape
        if a == 1 and b > 1:
            arr = arr.reshape(f, b)
        elif b == 1 and a > 1:
            arr = arr.reshape(f, a)
        else:
            arr = arr.reshape(f, a * b)
    frames, M = arr.shape
    wl = None
    for name in ("get_wavelengths", "get_wavelength", "wavelengths", "wavelength"):
        if hasattr(sr, name):
            try:
                obj = getattr(sr, name)()
                wl = np.asarray(obj, dtype=float).reshape(-1)
                break
            except Exception:
                wl = None
    if wl is None or wl.size != M or not np.isfinite(wl).all():
        wl = np.arange(M, dtype=float)  # 메타 결손 → 픽셀 인덱스 축
    return arr.astype(np.float64, copy=False), wl.astype(np.float64, copy=False)

def read_sif(path: str, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    if _BACKEND == "sif_parser":
        data, axis = _read_sif_with_sif_parser(path)
    elif _BACKEND == "sifreader":
        data, axis = _read_sif_with_sifreader(path)
    else:
        raise SkipFile("SIF 백엔드 미탑재. sif_parser 또는 sifreader 설치 필요")

    # 사용자 축 강제 지정(선택)
    if cfg.axis_override is not None:
        mode, a0, step = cfg.axis_override
        if mode.lower() == "linear":
            M = data.shape[1]
            axis = (a0 + step * np.arange(M, dtype=float)).astype(np.float64, copy=False)
    return data, axis

# =============================================================================
# 유틸리티
# =============================================================================

def build_axes(xsize: float, ysize: float, xgrid: int, ygrid: int,
               neg_x: bool, neg_y: bool) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(-float(xsize), +float(xsize), int(xgrid), dtype=np.float64)
    ys = np.linspace(-float(ysize), +float(ysize), int(ygrid), dtype=np.float64)
    if neg_x:
        xs = -xs
    if neg_y:
        ys = -ys
    return xs, ys

def roi_mask(axis: np.ndarray, xmin: Optional[float], xmax: Optional[float]) -> np.ndarray:
    if xmin is None and xmax is None:
        return np.isfinite(axis)
    a_min, a_max = float(np.nanmin(axis)), float(np.nanmax(axis))
    lo = a_min if xmin is None else max(float(xmin), a_min)
    hi = a_max if xmax is None else min(float(xmax), a_max)
    if lo > hi:
        lo, hi = hi, lo
    return (axis >= lo) & (axis <= hi) & np.isfinite(axis)

def aggregate_vec(y: np.ndarray, mode: str) -> float:
    if mode == "sum":
        return float(np.nansum(y))
    if mode == "mean":
        return float(np.nanmean(y))
    if mode == "max":
        return float(np.nanmax(y))
    raise SkipFile("aggregate는 'sum'|'mean'|'max' 중 하나여야 함")

def infer_grid_or_skip(xy: np.ndarray, decimals: int) -> Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    좌표가 완전한 정격자인지 검사. 아니면 SkipFile 발생.
    """
    pts = np.round(xy, decimals=decimals)
    xs = np.unique(pts[:, 0])
    ys = np.unique(pts[:, 1])
    nx, ny = xs.size, ys.size
    grid_like = (nx * ny == pts.shape[0])
    if not grid_like:
        # 누락/중복/불규칙
        raise SkipFile(
            "XY가 완전한 정격자가 아님\n"
            f"- unique X 개수 × unique Y 개수 = {nx}×{ny} = {nx*ny}\n"
            f"- 포인트 개수 N = {pts.shape[0]}\n"
            "- 동일 좌표 중복 또는 누락이 있거나, 스캔 격자와 CONFIG(Xgrid,Ygrid)가 불일치"
        )
    # 인덱서 구성
    x2i = {float(v): i for i, v in enumerate(xs)}
    y2i = {float(v): i for i, v in enumerate(ys)}
    indexer = np.empty((pts.shape[0], 2), dtype=int)
    for k, (x, y) in enumerate(pts):
        indexer[k, 0] = x2i[float(x)]
        indexer[k, 1] = y2i[float(y)]
    return nx, ny, xs, ys, indexer

def cm(name: str) -> Colormap:
    try:
        return plt.get_cmap(name)
    except Exception:
        return plt.get_cmap("CMRmap")

CMAPS: Dict[str, Colormap] = {
    "CMRmap": cm("CMRmap"),
    "turbo": cm("turbo"),
    "viridis": cm("viridis"),
    "plasma": cm("plasma"),
    "magma": cm("magma"),
    "inferno": cm("inferno"),
    "cividis": cm("cividis"),
    "jet": cm("jet"),
}

def fmt_for_fname(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    s = f"{x:.6g}"
    return s.replace(".", "p")

# =============================================================================
# Core: SIF → StandardSpectrumMap → Heatmap
# =============================================================================

def sif_to_standard_in_memory(path_sif: str, cfg: Config) -> StandardSpectrumMap:
    data, axis = read_sif(path_sif, cfg)  # (frames, M), (M,)
    frames, M = data.shape

    expected = int(cfg.Xgrid) * int(cfg.Ygrid)
    if frames != expected:
        raise SkipFile(
            "프레임 수와 CONFIG 격자 불일치\n"
            f"- frames      : {frames}\n"
            f"- Xgrid*Ygrid : {cfg.Xgrid}*{cfg.Ygrid} = {cfg.Xgrid*cfg.Ygrid}\n"
            "→ CONFIG의 Xgrid/Ygrid 또는 SIF의 스캔 설정을 확인"
        )

    # (Y, X, M) 큐브
    cube = data.reshape(int(cfg.Ygrid), int(cfg.Xgrid), M)

    xs, ys = build_axes(cfg.Xsize, cfg.Ysize, cfg.Xgrid, cfg.Ygrid,
                        cfg.negate_x_coord, cfg.negate_y_coord)

    spectra_list: List[np.ndarray] = []
    xy_list: List[Tuple[float, float]] = []

    fo = (cfg.frame_order or "YX").upper()
    x_iter = range(cfg.Xgrid - 1, -1, -1) if cfg.flip_x_order else range(cfg.Xgrid)
    if fo == "YX":
        for yi in range(cfg.Ygrid):
            for xi in x_iter:
                spectra_list.append(cube[yi, xi, :].astype(np.float64, copy=False))
                xy_list.append((float(xs[xi]), float(ys[yi])))
    elif fo == "XY":
        for xi in x_iter:
            for yi in range(cfg.Ygrid):
                spectra_list.append(cube[yi, xi, :].astype(np.float64, copy=False))
                xy_list.append((float(xs[xi]), float(ys[yi])))
    else:
        raise SkipFile("frame_order는 'YX' 또는 'XY' 여야 함")

    spectra = np.vstack(spectra_list).astype(np.float64, copy=False)
    xy = np.asarray(xy_list, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)

    ssm = StandardSpectrumMap(spectra=spectra, xy=xy, axis=axis, unit=(cfg.unit_hint or ""))
    ssm.validate()
    return ssm

def compute_heat_values(ssm: StandardSpectrumMap, roi: np.ndarray, agg: str) -> np.ndarray:
    if not np.any(roi):
        # ROI가 축과 겹치지 않음
        a_min, a_max = float(np.nanmin(ssm.axis)), float(np.nanmax(ssm.axis))
        raise SkipFile(
            "ROI가 스펙트럼 축과 겹치지 않음\n"
            f"- axis range : [{a_min:g}, {a_max:g}]\n"
            f"- ROI        : [{('None' if roi.size==0 else '')}] (cfg.roi_xmin={CONFIG.roi_xmin}, cfg.roi_xmax={CONFIG.roi_xmax})\n"
            "→ ROI 범위를 축 범위와 겹치도록 조정"
        )

    Y = ssm.spectra
    out = np.full((Y.shape[0],), np.nan, dtype=np.float64)
    for i in range(Y.shape[0]):
        yi = Y[i, roi]
        if np.isfinite(yi).any():
            out[i] = aggregate_vec(yi, agg)
    if not np.isfinite(out).any():
        raise SkipFile("ROI 내 유효한 집계값이 없음(모두 NaN)")
    return out

def to_heatmap_or_skip(ssm: StandardSpectrumMap, cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    roi = roi_mask(ssm.axis, cfg.roi_xmin, cfg.roi_xmax)
    vals = compute_heat_values(ssm, roi, cfg.aggregate)

    # 정격자 강제: 불규칙/누락 있으면 스킵
    nx, ny, xs, ys, indexer = infer_grid_or_skip(ssm.xy, cfg.coord_round_decimals)
    Z = np.full((ny, nx), np.nan, dtype=np.float64)
    for (ix, iy), v in zip(indexer, vals):
        Z[iy, ix] = v
    if not np.isfinite(Z).any():
        raise SkipFile("그리드에 매핑된 값이 없음(전부 NaN)")
    return xs, ys, Z

def derive_outpath(cfg: Config, path_sif: str, ssm: StandardSpectrumMap) -> str:
    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(path_sif).stem
    rlo = cfg.roi_xmin if cfg.roi_xmin is not None else float(np.nanmin(ssm.axis))
    rhi = cfg.roi_xmax if cfg.roi_xmax is not None else float(np.nanmax(ssm.axis))
    tag = f"{fmt_for_fname(rlo)}_{fmt_for_fname(rhi)}"
    return str(out_dir / f"{base}__heat_{cfg.aggregate}_{tag}.png")

def plot_heatmap(ssm: StandardSpectrumMap, cfg: Config, out_png: str) -> None:
    Xs, Ys, Z = to_heatmap_or_skip(ssm, cfg)

    if cfg.color_auto:
        finite = np.isfinite(Z)
        vmin = float(np.nanmin(Z[finite])) if finite.any() else None
        vmax = float(np.nanmax(Z[finite])) if finite.any() else None
    else:
        vmin, vmax = cfg.vmin, cfg.vmax

    fig, ax = plt.subplots(figsize=cfg.figsize)
    extent = [float(np.min(Xs)), float(np.max(Xs)), float(np.min(Ys)), float(np.max(Ys))]
    im = ax.imshow(
        Z, extent=extent, origin="lower", aspect="auto",
        cmap=CMAPS.get(cfg.cmap_name, CMAPS["CMRmap"]),
        vmin=vmin, vmax=vmax, interpolation="nearest"
    )

    unit = f" ({ssm.unit})" if ssm.unit else ""
    roi_txt = ""
    if cfg.roi_xmin is not None or cfg.roi_xmax is not None:
        lo = cfg.roi_xmin if cfg.roi_xmin is not None else float(np.nanmin(ssm.axis))
        hi = cfg.roi_xmax if cfg.roi_xmax is not None else float(np.nanmax(ssm.axis))
        roi_txt = f" | ROI: {lo:g}..{hi:g}{unit}"

    ax.set_title(f"Heatmap: {cfg.aggregate}{roi_txt}")
    ax.set_xlabel("X (arb. units)")
    ax.set_ylabel("Y (arb. units)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label(f"Aggregated intensity ({cfg.aggregate})")

    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=cfg.dpi, bbox_inches="tight")
    if cfg.show_plot:
        plt.show()
    plt.close(fig)

# =============================================================================
# Main
# =============================================================================

def run_one_file(path: str, cfg: Config) -> None:
    try:
        ssm = sif_to_standard_in_memory(path, cfg)
        out_png = derive_outpath(cfg, path, ssm)
        plot_heatmap(ssm, cfg, out_png)
        print(f"[saved] {out_png}")
    except SkipFile as e:
        print(
            "\n[skip] 처리 건너뜀:", Path(path).name,
            "\n- 사유:\n" + indent_lines(str(e)) +
            "\n- 조치 제안:\n" + indent_lines(suggest_actions(str(e))) + "\n"
        )
    except Exception as e:
        print(
            "\n[skip] 처리 중 예기치 못한 오류:", Path(path).name,
            f"\n- 예외형: {type(e).__name__}\n- 메시지: {e}\n"
            "- 로그/스택을 확인하고, CONFIG와 SIF 메타를 재검토\n"
        )

def indent_lines(s: str, pad: str = "  ") -> str:
    return "\n".join(pad + line for line in s.splitlines())

def suggest_actions(reason: str) -> str:
    r = reason.lower()
    tips = []
    if "frames" in r or "격자" in r:
        tips += [
            "SIF의 실제 스캔 포인트 수와 CONFIG의 Xgrid/Ygrid 값을 일치시킴",
            "스캔 순서(frame_order=YX|XY)와 flip_x_order 옵션 확인",
        ]
    if "정격자" in r or "xy" in r:
        tips += [
            "스캔이 누락 없이 완료되었는지 확인(중복 포인트/누락 포인트 점검)",
            "CONFIG.Xsize/Ysize, negate_x_coord/negate_y_coord 설정 점검",
        ]
    if "roi" in r:
        tips += [
            "cfg.roi_xmin/cfg.roi_xmax를 축 범위 안으로 조정",
            "축 단위를 재점검(unit_hint, axis_override)",
        ]
    if "axis" in r and "유효값" in r:
        tips += [
            "축 추출 실패 시 axis_override=('linear', start, step)로 명시 지정",
        ]
    if not tips:
        tips = ["CONFIG와 SIF 메타(프레임 수, 스캔 격자, 축 단위)를 재점검"]
    return "\n".join(f"- {t}" for t in tips)

def main(cfg: Config) -> None:
    paths: Sequence[str]
    if isinstance(cfg.sif_paths, (str, Path)):
        paths = [str(cfg.sif_paths)]
    else:
        paths = list(cfg.sif_paths)

    print(f"[SIF backend] {_BACKEND or 'None'}")
    for p in paths:
        run_one_file(p, cfg)

if __name__ == "__main__":
    main(CONFIG)
