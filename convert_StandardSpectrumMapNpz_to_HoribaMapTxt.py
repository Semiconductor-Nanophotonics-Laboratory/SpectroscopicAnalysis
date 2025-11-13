#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
convert_StandardSpectrumMapNpz_to_HoribaMapTxt.py

StandardSpectrumMap NPZ → Horiba 맵 TXT 변환기 (내부 옵션 지정형)

NPZ 스키마
- spectra : (N, M) float64  # N: 포인트, M: 스펙트럼 길이
- xy      : (N, 2) float64  # 각 스펙트럼의 (x, y)
- axis    : (M,)   float64  # 스펙트럼 축 (예: nm 또는 cm^-1)
- unit    : ()     str      # 선택

Horiba TXT 포맷
- 1행: "", "" , axis[0], axis[1], ..., axis[M-1]   (탭 구분, CRLF 개행)
- 2행~: X, Y, I0, I1, ..., I(M-1)

가정
- xy는 완전한 직사각 격자를 이룸(|X|×|Y|=N).
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np

# =========================
# 사용자 옵션 (여기만 수정)
# =========================
# 입력: .npz 파일 또는 폴더 지정 (폴더면 내부의 *.npz 일괄 처리)
INPUTS: List[str] = [
    "./test.npz",  # 파일 예시
]

# 출력 폴더
OUT_DIR: str = "./"

# 프레임 출력 순서/방향
FRAME_ORDER: str   = "YX"    # "YX" 또는 "XY"
FLIP_X_ORDER: bool = False   # True면 x 정렬 방향 전체 반전
NEGATE_X: bool     = False   # True면 출력 좌표 X에 -1 곱함(라벨만)
NEGATE_Y: bool     = False   # True면 출력 좌표 Y에 -1 곱함(라벨만)

# 숫자 출력 스타일
MAX_DECIMALS: int = 7        # 고정소수점 자리수 상한(뒤 0/점 제거)
DELIM: str = "\t"            # 탭
CRLF: str  = "\r\n"          # CRLF 줄바꿈

# 파일 덮어쓰기 허용
OVERWRITE: bool = True


# =========================
# 내부 유틸
# =========================
def fmt_number(x: float, max_decimals: int = 7) -> str:
    """정수는 정수로, 실수는 고정소수점으로(지수표현 금지), 말줄임 0/점 제거."""
    xf = float(x)
    if math.isfinite(xf) and xf.is_integer():
        return str(int(xf))
    s = f"{xf:.{max_decimals}f}".rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    if "e" in s.lower():  # 안전장치
        s = f"{xf:.{max_decimals}f}".rstrip("0").rstrip(".")
        if s == "-0":
            s = "0"
    return s

def load_standard_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=False)
    for k in ("spectra", "xy", "axis"):
        if k not in data:
            raise ValueError(f"{npz_path}: '{k}' key missing in NPZ.")
    spectra = np.asarray(data["spectra"], dtype=np.float64)
    xy      = np.asarray(data["xy"], dtype=np.float64)
    axis    = np.asarray(data["axis"], dtype=np.float64)
    unit    = ""
    if "unit" in data:
        u = data["unit"]
        try:
            unit = str(u.item() if hasattr(u, "item") else u)
        except Exception:
            unit = str(u)

    if spectra.ndim != 2:
        raise ValueError(f"{npz_path}: spectra must be 2D (N,M).")
    N, M = spectra.shape
    if xy.shape != (N, 2):
        raise ValueError(f"{npz_path}: xy must be (N,2); got {xy.shape}, expected N={N}.")
    if axis.shape != (M,):
        raise ValueError(f"{npz_path}: axis must be (M,); got {axis.shape}, expected M={M}.")
    if not np.isfinite(axis).all():
        raise ValueError(f"{npz_path}: axis contains non-finite values.")
    if not np.isfinite(xy).all():
        raise ValueError(f"{npz_path}: xy contains non-finite values.")
    if not np.isfinite(spectra).all():
        raise ValueError(f"{npz_path}: spectra contains non-finite values.")
    return spectra, xy, axis, unit

def unique_sorted(vals: np.ndarray) -> np.ndarray:
    u = np.unique(vals)
    u.sort()
    return u

def build_order_indices(
    xy: np.ndarray,
    frame_order: str = "YX",
    flip_x_order: bool = False,
) -> np.ndarray:
    """
    xy (N,2) → 행 출력 순서 인덱스(N,)
    - YX: y 증가 → 그 안에서 x 증가
    - XY: x 증가 → 그 안에서 y 증가
    - flip_x_order=True: x 전체 방향 반전(간단 반전)
    """
    x = xy[:, 0]
    y = xy[:, 1]
    xs = unique_sorted(x)
    ys = unique_sorted(y)

    N = xy.shape[0]
    if xs.size * ys.size != N:
        raise ValueError(
            f"XY grid incomplete: |X|*|Y|={xs.size}*{ys.size}={xs.size*ys.size} != N={N}"
        )

    # 정확 매칭 실패 대비 최근접 매칭 허용
    def nearest(arr, v):
        return int(np.argmin(np.abs(arr - v)))

    idx_pairs = np.empty((N, 2), dtype=int)
    for i, (xx, yy) in enumerate(xy):
        # float 키 딕셔너리 대신 최근접 색인으로 안전 처리
        ix = nearest(xs, float(xx))
        iy = nearest(ys, float(yy))
        idx_pairs[i] = (ix, iy)

    f = frame_order.upper()
    if f == "YX":
        order = np.lexsort((idx_pairs[:, 0], idx_pairs[:, 1]))  # (x, then y)
        if flip_x_order:
            idx_pairs2 = idx_pairs.copy()
            idx_pairs2[:, 0] = -idx_pairs2[:, 0]
            order = np.lexsort((idx_pairs2[:, 0], idx_pairs2[:, 1]))
    elif f == "XY":
        order = np.lexsort((idx_pairs[:, 1], idx_pairs[:, 0]))  # (y, then x)
        if flip_x_order:
            idx_pairs2 = idx_pairs.copy()
            idx_pairs2[:, 0] = -idx_pairs2[:, 0]
            order = np.lexsort((idx_pairs2[:, 1], idx_pairs2[:, 0]))
    else:
        raise ValueError("FRAME_ORDER must be 'YX' or 'XY'")
    return order

def write_horiba_txt(
    out_path: str,
    xy: np.ndarray,
    spectra: np.ndarray,
    axis: np.ndarray,
    *,
    frame_order: str = "YX",
    flip_x_order: bool = False,
    negate_x: bool = False,
    negate_y: bool = False,
) -> None:
    """
    Horiba TXT 작성:
    - 첫 행: "", "", axis...
    - 이후: X, Y, intensities...
    - 개행: CRLF, 구분자: 탭
    """
    N, M = spectra.shape
    order = build_order_indices(xy, frame_order=frame_order, flip_x_order=flip_x_order)

    # 헤더
    header_cells = ["", ""]
    header_cells.extend(fmt_number(v, MAX_DECIMALS) for v in axis)
    header_line = DELIM.join(header_cells) + CRLF

    # 본문
    lines: List[str] = [header_line]
    X = xy[:, 0].copy()
    Y = xy[:, 1].copy()
    if negate_x:
        X = -X
    if negate_y:
        Y = -Y

    for idx in order:
        xval = fmt_number(X[idx], MAX_DECIMALS)
        yval = fmt_number(Y[idx], MAX_DECIMALS)
        spec = spectra[idx, :]
        row = [xval, yval]
        row.extend(fmt_number(float(v), MAX_DECIMALS) for v in spec)
        lines.append(DELIM.join(row) + CRLF)

    # 파일 쓰기
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if OVERWRITE else "x"
    with open(out_path, mode, encoding="ascii", errors="strict", newline="") as f:
        for ln in lines:
            f.write(ln)

def discover_inputs(inputs: List[str]) -> List[str]:
    found: List[str] = []
    for p in inputs:
        P = Path(p)
        if P.is_file() and P.suffix.lower() == ".npz":
            found.append(str(P))
        elif P.is_dir():
            for q in P.glob("*.npz"):
                found.append(str(q))
    # 중복 제거(안정 순서 유지)
    seen = set()
    ordered: List[str] = []
    for f in found:
        if f not in seen:
            ordered.append(f)
            seen.add(f)
    return ordered

def convert_one(npz_path: str, out_dir: str) -> str:
    spectra, xy, axis, unit = load_standard_npz(npz_path)
    in_p = Path(npz_path)
    out_p = Path(out_dir) / (in_p.stem + ".txt")
    write_horiba_txt(
        str(out_p),
        xy=xy,
        spectra=spectra,
        axis=axis,
        frame_order=FRAME_ORDER,
        flip_x_order=FLIP_X_ORDER,
        negate_x=NEGATE_X,
        negate_y=NEGATE_Y,
    )
    return str(out_p)

def main():
    targets = discover_inputs(INPUTS)
    if not targets:
        print("No .npz inputs found. Edit INPUTS at top of file.")
        return
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    for p in targets:
        try:
            out = convert_one(p, OUT_DIR)
            spectra, xy, axis, unit = load_standard_npz(p)
            N, M = spectra.shape
            print(f"[OK] {Path(p).name} -> {Path(out).name}  "
                  f"(N={N}, M={M}, axis[0]={fmt_number(axis[0])}, axis[-1]={fmt_number(axis[-1])}, unit='{unit}')")
        except Exception as e:
            print(f"[ERR] {Path(p).name}: {e}")

if __name__ == "__main__":
    main()
