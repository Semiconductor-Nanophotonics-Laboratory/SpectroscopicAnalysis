# standard_spectrummap.py

from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np


# =========================
# Standard data container
# =========================
@dataclass
class StandardSpectrumMap:
    """
    표준 스펙트럼 맵 컨테이너
      - spectra: (N, M) float64  — 각 포인트의 intensity 스펙트럼
      - xy     : (N, 2) float64  — 각 포인트의 (x, y) 좌표
      - axis   : (M,)   float64  — x축(라만 쉬프트, 파장 등). 없으면 0..M-1
      - unit   : str            — 축 단위("cm^-1", "nm"...). 모르면 ""
    """
    spectra: np.ndarray
    xy: np.ndarray
    axis: np.ndarray
    unit: str = ""

    def validate(self) -> None:
        if self.spectra.ndim != 2:
            raise ValueError("spectra must be 2D (N, M).")
        if self.xy.shape != (self.spectra.shape[0], 2):
            raise ValueError(f"xy must be shape (N, 2); got {self.xy.shape}, N={self.spectra.shape[0]}")
        if self.axis.shape != (self.spectra.shape[1],):
            raise ValueError(f"axis must be shape (M,); got {self.axis.shape}, M={self.spectra.shape[1]}")
        if not (np.isfinite(self.axis).all()):
            raise ValueError("axis contains non-finite values.")
        # spectra는 NaN 허용(패딩 가능), xy는 finite 권장
        if not np.isfinite(self.xy).all():
            raise ValueError("xy contains non-finite values.")

    @property
    def shape(self) -> Tuple[int, int]:
        return self.spectra.shape  # (N, M)


# =========================
# Internal helpers
# =========================
_num_re = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")

def _tokenize(line: str) -> List[str]:
    # BOM 제거 + 쉼표/세미콜론을 공백으로 치환하여 혼용 구분 허용
    line = line.replace("\ufeff", "").replace(",", " ").replace(";", " ")
    return [t for t in line.strip().split() if t]

def _is_float(tok: str) -> bool:
    try:
        float(tok)
        return True
    except Exception:
        return False

def _line_to_floats(line: str) -> List[float]:
    toks = _tokenize(line)
    return [float(t) for t in toks if _is_float(t)]


# =========================
# Public API 1:
# TXT -> StandardSpectrumMap
# =========================
def load_HoribaMapTxt_toStandardSpectrumMap(path: str, *,
                                            min_spectrum_len: int = 10,
                                            unit_hint: str = "") -> StandardSpectrumMap:
    """
    HORIBA mapping TXT를 로드하여 StandardSpectrumMap으로 반환.
    형식 자동 감지:
      A) 와이드 테이블: 1행(헤더)에 숫자 축이 주어지고, 이후 행은 'X Y I1 I2 ...'
      B) (백업) 2-줄 묶음: '스펙트럼 1행' + 'X Y ... 1행' 반복 (축 없으면 0..M-1)
    """
    lines_raw = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    # 공백/주석 제거(단, 첫 줄은 포맷 감지를 위해 남김)
    lines = []
    for i, ln in enumerate(lines_raw):
        s = ln.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("//"):
            continue
        lines.append(s)
    if not lines:
        raise ValueError("No valid content found in TXT.")

    # ---- 형식 A: 와이드 테이블 감지 ----
    header_tokens = _tokenize(lines[0])
    num_mask = [_is_float(t) for t in header_tokens]
    header_numeric_count = sum(num_mask)

    if header_numeric_count >= 10 and header_numeric_count >= int(0.8 * max(1, len(header_tokens))):
        # 헤더 대부분이 숫자 → 축으로 사용
        header_axis = [float(t) for t, m in zip(header_tokens, num_mask) if m]

        spectra_list: List[np.ndarray] = []
        xy_list: List[np.ndarray] = []

        for ln in lines[1:]:
            nums = _line_to_floats(ln)
            if len(nums) < 3:
                continue
            x, y = nums[0], nums[1]
            vals = np.asarray(nums[2:], dtype=np.float64)
            xy_list.append(np.array([x, y], dtype=np.float64))
            spectra_list.append(vals)

        if not spectra_list:
            raise ValueError("Wide-table detected, but no data rows parsed.")

        # 패딩 정렬
        M = max(len(v) for v in spectra_list)
        N = len(spectra_list)
        spectra = np.full((N, M), np.nan, dtype=np.float64)
        for i, v in enumerate(spectra_list):
            spectra[i, :len(v)] = v
        xy = np.vstack(xy_list).astype(np.float64)

        # 축 길이 불일치 시 인덱스 축으로 대체
        axis = np.asarray(header_axis, dtype=np.float64)
        if axis.shape != (M,):
            axis = np.arange(M, dtype=np.float64)

        obj = StandardSpectrumMap(spectra=spectra, xy=xy, axis=axis, unit=unit_hint)
        obj.validate()
        return obj

    # ---- 형식 B: 2-줄 묶음(백업) ----
    spectra_list: List[np.ndarray] = []
    xy_list: List[np.ndarray] = []
    i = 0
    while i < len(lines):
        spec_nums = _line_to_floats(lines[i])
        if len(spec_nums) >= min_spectrum_len:
            if i + 1 < len(lines):
                coord_nums = _line_to_floats(lines[i + 1])
                if len(coord_nums) >= 2:
                    spectra_list.append(np.asarray(spec_nums, dtype=np.float64))
                    xy_list.append(np.array([coord_nums[0], coord_nums[1]], dtype=np.float64))
                    i += 2
                    continue
        i += 1

    if not spectra_list:
        raise ValueError("Could not parse TXT as wide-table or 2-line-paired format.")

    M = max(len(v) for v in spectra_list)
    N = len(spectra_list)
    spectra = np.full((N, M), np.nan, dtype=np.float64)
    for r, a in enumerate(spectra_list):
        spectra[r, :len(a)] = a
    xy = np.vstack(xy_list).astype(np.float64)
    axis = np.arange(M, dtype=np.float64)

    obj = StandardSpectrumMap(spectra=spectra, xy=xy, axis=axis, unit=unit_hint)
    obj.validate()
    return obj


# =========================
# Public API 2:
# StandardSpectrumMap -> NPZ
# =========================
def save_StandardSpectrumMap_toNpz(obj: StandardSpectrumMap,
                                   npz_path: str,
                                   *,
                                   compressed: bool = True) -> None:
    """
    표준 스펙트럼 맵을 NPZ로 저장.
      keys: spectra(N,M), xy(N,2), axis(M,), unit(str)
    """
    obj.validate()
    saver = np.savez_compressed if compressed else np.savez
    saver(npz_path,
          spectra=obj.spectra.astype(np.float64, copy=False),
          xy=obj.xy.astype(np.float64, copy=False),
          axis=obj.axis.astype(np.float64, copy=False),
          unit=np.array(obj.unit))


# =========================
# Example main (batch)
# =========================
if __name__ == "__main__":
    # 처리할 TXT 파일들을 직접 나열
    txt_files = [
        "7thR_0p_p1.txt",
        "7thR_10p_p1.txt",
        "7thR_20p_p1.txt",
        "7thR_30p_p1.txt",
        "7thR_40p_p1.txt",
        "7thR_50p_p1.txt",
        "7thR_60p_p1.txt",
        "7thR_70p_p1.txt",
        "7thR_80p_p1.txt",
        "7thR_90p_p1.txt",
        "7thR_100p_p1.txt"
    ]
    out_dir = Path("./npz_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    for fp in txt_files:
        try:
            obj = load_HoribaMapTxt_toStandardSpectrumMap(fp, unit_hint="cm^-1")  # 필요시 단위 지정
            npz_path = out_dir / (Path(fp).stem + ".npz")
            save_StandardSpectrumMap_toNpz(obj, str(npz_path))
            # 간단 검증
            data = np.load(npz_path, allow_pickle=False)
            N, M = data["spectra"].shape
            print(f"[OK] {Path(fp).name} -> {npz_path.name} (N={N}, M={M}, axis[0]={data['axis'][0]:.4g}, axis[-1]={data['axis'][-1]:.4g}, unit='{str(data['unit'])}')")
        except Exception as e:
            print(f"[ERR] {fp}: {e}")