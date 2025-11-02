# fit_main.py
# -----------------------------------------------------------------------------
# Run:  python fit_main.py
# 입력: StandardSpectrumMapNpz
# 출력: FitSpectrumMap.v1 (.npz는 항상 저장)
# - save_plots  : 플롯 PNG 저장 여부 (plots/ 폴더)
# - save_extras : CSV/TXT/Origin CSV 보조 산출물 일괄 저장 여부
# - stages      : StageSpec 스케줄(스테이지 개수/학습률/반복수/활성 피크 인덱스 지정)
# 기타 기능: 범위 자동 클램프, pV 통일 모델, 원본축 재합성 저장, Enter/Esc 프리뷰, 손실곡선 저장
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import time
import numpy as np
import matplotlib.pyplot as plt
import torch  # device 확인/재합성용

from process_module_StandardSpectrumMapNpz import load_StandardSpectrumMapNpz
from fit_module_StandardSpectrumMapNpz import (
    FitConfig, FitHyperParams, StageSpec, PeakSpec,
    build_torch_args, SpectrumFitter,
    prepare_xy_from_npz, select_indices,
    fig_initial_guess, fig_final_fit,
    compute_peak_areas, save_origin_ready_csv, export_text_spectra,
    save_FitSpectrumMapNpz, synthesize_pv,
)

# ============================= CONFIG ========================================

@dataclass(frozen=True)
class AppConfig:
    # -------------------- 입출력 --------------------
    input_file: str = "7thR_80p_p1_whT4_m5_interp_600_3600_4_sgW7_P3_S2_normI_r3_blpeakstrip.npz"
    output_root: str = "./result"
    run_prefix: str = ""
    run_tag: str | None = None

    # -------------------- 스펙트럼 선택 --------------------
    select_mode: str = "all"          # 'one' | 'all' | 'indices' | 'random'
    indices: List[int] = (0, 10, 20)
    k_random: int = 5
    seed: int = 42

    # -------------------- X축 범위 (개별 None 허용; 모듈에서 데이터 경계로 클램프) --------------------
    xmin: Optional[float] = 550.0
    xmax: Optional[float] = 3500.0

    # -------------------- 장치/로그 --------------------
    force_cpu: bool = False
    show_initial_preview: bool = False   # True면 초기 플롯에서 Enter=계속, Esc=중단
    verbosity: int = 1                   # 0: WARNING, 1: INFO, 2+: DEBUG

    # -------------------- 저장 옵션 --------------------
    save_plots: bool = True    # plots 폴더와 PNG 저장
    save_extras: bool = False   # CSV(피크파라미터), TXT(베이스 포함/제거), Origin CSV 저장

# -------------------- 옵션: 세 가지 기본함수로 제한 --------------------
BASIC_THREE_ONLY: bool = True

# -------------------- PeakSpec 주석 --------------------
# PeakSpec(
#   type,        # "gaussian" | "lorentzian" | "pseudovoigt"
#   pos,         # 초기 중심(cm^-1)
#   width,       # 초기 FWHM
#   height,      # 초기 세기(면적≈height, 커널 정규화)
#   pos_limit,   # 중심 허용범위(±) 초과 시 패널티
#   pos_limit_reg,    # 중심 초과 패널티 가중치
#   width_limit,      # FWHM 허용범위(±) 초과 시 패널티
#   width_limit_reg,  # FWHM 초과 패널티 가중치
#   height_limit,     # 높이 허용범위(±) 초과 시 패널티
#   height_limit_reg, # 높이 초과 패널티 가중치
#   extra={"eta":0~1} # pV 초기 혼합율(가우시안0↔로렌츠1), g/l은 고정, pV는 자유
# )

# --------- Peaks (D4, D1, D3, G, D′, 2D, D+D2, 2D2) ----------
DEMO_PEAKS: List[PeakSpec] = [
    PeakSpec(type="pseudovoigt", pos=1181.0, width=157.0, height=13, pos_limit=4,  pos_limit_reg=6, width_limit=20, width_limit_reg=6, height_limit=0.3, height_limit_reg=1.0, extra={"eta":0.6}),  # D4
    PeakSpec(type="lorentzian",  pos=1329.0, width=82.0,  height=120, pos_limit=7,  pos_limit_reg=7, width_limit=20, width_limit_reg=4, height_limit=0.3, height_limit_reg=1.0),                  # D1
    PeakSpec(type="pseudovoigt", pos=1475.0, width=139.0, height=12, pos_limit=8,  pos_limit_reg=7, width_limit=20, width_limit_reg=5, height_limit=0.3, height_limit_reg=1.0, extra={"eta":0.5}), # D3
    
    PeakSpec(type="lorentzian",  pos=1581.0, width=58.0,  height=40, pos_limit=4,  pos_limit_reg=3, width_limit=20, width_limit_reg=6, height_limit=0.3, height_limit_reg=10),                  # G
    PeakSpec(type="lorentzian",  pos=1613.0, width=37.0,  height=22, pos_limit=4,  pos_limit_reg=3, width_limit=20, width_limit_reg=6, height_limit=0.3, height_limit_reg=10),                  # D′
    
    PeakSpec(type="pseudovoigt", pos=2474.0, width=190.0, height=4,  pos_limit=4,  pos_limit_reg=3, width_limit=20, width_limit_reg=3, height_limit=0.3, height_limit_reg=1.0, extra={"eta":0.5}), # 2D
    PeakSpec(type="lorentzian",  pos=2654.0, width=119.0, height=23, pos_limit=4,  pos_limit_reg=3, width_limit=20, width_limit_reg=3, height_limit=0.3, height_limit_reg=1.0),                  # D+D2
    PeakSpec(type="lorentzian",  pos=2911.0, width=138.0, height=8,  pos_limit=4, pos_limit_reg=3, width_limit=8,  width_limit_reg=3, height_limit=0.3, height_limit_reg=1.0),                  # 2D2
]

# -------------------- 하이퍼파라미터 (레거시 백업용) --------------------
HYP = FitHyperParams(
    num_iter_stage1=2000,
    num_iter_stage2=2000,
    lr_stage1=0.015,
    lr_stage2=0.002,
    height_sign_reg=5.0,
    pos_var_reg=0.0,
    width_var_reg=0.0,
    height_var_reg=0.0,
)

# -------------------- 피팅 구성 (스테이지 스케줄) --------------------
FITCFG = FitConfig(
    peaks=DEMO_PEAKS,
    sort_peaks_by_pos=True,
    # 레거시(stage1/2) 대신 자유 스테이지 스케줄을 사용
    stages=[
        StageSpec(name="main", iters=2000, lr=0.003, active=[0, 1, 5, 6, 7]),     # D4, D1, D3 먼저
        StageSpec(name="defect", iters=8000, lr=0.003, active="all"),# +G, D'
    ],
    base_init=0.0,
    use_gpu=True,
    x_fit_range=None,  # (xmin, xmax) 튜플로 주입(각 항목 None 허용)
    x_label="x",
    y_label="Intensity",
    h=HYP,
    basic_three_only=BASIC_THREE_ONLY,
)

CONFIG = AppConfig()

# ============================== HELPERS ======================================

def setup_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(level=level, format="[%(levelname)s] %(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")

def plt_close(fig) -> None:
    try:
        plt.close(fig)
    except Exception:
        pass

def preview_wait_for_key(fig: plt.Figure) -> str:
    """
    초기 미리보기에서 키 입력 대기:
    - Enter: 계속 진행
    - Esc  : 전체 프로세스 중단
    기타 키: 무시(대기 지속)
    """
    state = {"action": None}

    def on_key(event):
        if event.key == "enter":
            state["action"] = "continue"
            plt.close(fig)
        elif event.key == "escape":
            state["action"] = "abort"
            plt.close(fig)

    cid = fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    return state["action"] or "continue"

def save_peak_csv(outdir: Path, stem: str, idx: int, params, areas, coord) -> None:
    import csv
    path = outdir / f"fitted_peak_parameters_{stem}_idx{idx}.csv"
    disp_types = list(params["display_types"])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["coord_x", "coord_y", "baseline"]
        for j in range(len(disp_types)):
            header += [f"peak_{j+1}_type", f"peak_{j+1}_pos", f"peak_{j+1}_width", f"peak_{j+1}_height", f"peak_{j+1}_eta", f"peak_{j+1}_area"]
        w.writerow(header)
        row = [f"{coord[0,0]:.6f}", f"{coord[0,1]:.6f}", f"{params['base'][0,0]:.8f}"]
        for j, t in enumerate(disp_types):
            row += [t,
                    f"{params['pos'][0,j,0]:.8f}",
                    f"{params['width'][0,j,0]:.8f}",
                    f"{params['height'][0,j,0]:.8f}",
                    f"{params['eta'][0,j,0]:.6f}",
                    f"{areas[j]:.10g}"]
        w.writerow(row)

def export_txt_pair(outdir: Path, stem: str, idx: int, x: np.ndarray, fitted_with_base: np.ndarray, baseline_scalar: float, coord: np.ndarray) -> None:
    nobase = fitted_with_base - baseline_scalar
    export_text_spectra(str(outdir / f"fitted_spectra_with_base_{stem}_idx{idx}.txt"), x, fitted_with_base, coord)
    export_text_spectra(str(outdir / f"fitted_spectra_nobase_{stem}_idx{idx}.txt"), x, nobase, coord)

def reconstruct_on_full_axis(params, x_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    피팅 파라미터(params)로 원본 축(x_full) 전체에서 재합성.
    반환: (recon_full:(1,L), peaks_full:(1,P,L)) — numpy
    """
    L = int(x_full.shape[0])
    P = int(params["pos"].shape[1])

    x_t = torch.from_numpy(x_full.astype(np.float32, copy=False)).unsqueeze(0)  # (1,L)
    pos   = torch.from_numpy(params["pos"][0, :, 0].astype(np.float32, copy=False)).view(1, P, 1)
    width = torch.from_numpy(params["width"][0, :, 0].astype(np.float32, copy=False)).view(1, P, 1)
    height= torch.from_numpy(params["height"][0, :, 0].astype(np.float32, copy=False)).view(1, P, 1)
    eta   = torch.from_numpy(params["eta"][0, :, 0].astype(np.float32, copy=False)).view(1, P, 1)
    base  = torch.tensor([[float(params["base"][0, 0])]], dtype=torch.float32)  # (1,1)
    with torch.no_grad():
        recon_t, peaks_t = synthesize_pv(x_t, pos, width, height, base.expand(1, L), eta, active=list(range(P)))
    return recon_t.cpu().numpy(), peaks_t.cpu().numpy()

def save_loss_plot(plots_dir: Path, stem: str, idx: int, loss_hist: List[float]) -> None:
    if not loss_hist:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(loss_hist)), loss_hist, lw=1.5)
    ax.set_title("Fitting error vs. iteration")
    ax.set_xlabel("iteration")
    ax.set_ylabel("error (loss)")
    ax.set_yscale("log")
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{stem}_idx{idx}_loss.png", dpi=200)
    plt_close(fig)

def _format_elapsed(sec: float) -> str:
    m, s = divmod(sec, 60.0)
    h, m = divmod(m, 60.0)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"

# ============================== RUNTIME ======================================

def run_with_config(app: AppConfig, fitcfg: FitConfig) -> None:
    setup_logging(app.verbosity)
    t0 = time.perf_counter()

    # ----- Device 정보 출력 -----
    use_gpu_flag = (False if app.force_cpu else fitcfg.use_gpu)
    if use_gpu_flag and torch.cuda.is_available():
        dev_index = torch.cuda.current_device()
        dev_name = torch.cuda.get_device_name(dev_index)
        logging.info("Device: CUDA (index=%d, name=%s)", dev_index, dev_name)
    else:
        if use_gpu_flag and not torch.cuda.is_available():
            logging.info("Device: CPU (CUDA not available; falling back to CPU)")
        else:
            logging.info("Device: CPU (forced or GPU disabled)")
    # --------------------------------

    # 입력 로드
    in_path = Path(app.input_file)
    if not in_path.exists():
        raise FileNotFoundError(f"Not found: {in_path}")
    ssm = load_StandardSpectrumMapNpz(str(in_path))
    N_total, L = ssm.spectra.shape
    logging.info("Loaded NPZ | spectra=%s, axis=%s, unit=%s", ssm.spectra.shape, ssm.axis.shape, ssm.unit)

    # 선택
    idx_list = select_indices(N_total, app.select_mode, indices=app.indices, k_random=app.k_random, seed=app.seed)
    if not idx_list:
        logging.warning("No indices selected. Exit.")
        logging.info("Total execution time: %s", _format_elapsed(time.perf_counter() - t0))
        return

    # 요청 범위를 그대로 튜플로 전달(각 항목 None 허용) → 모듈이 데이터 경계로 클램프
    rng: Tuple[Optional[float], Optional[float]] = (app.xmin, app.xmax)

    fitcfg = FitConfig(
        peaks=fitcfg.peaks,
        sort_peaks_by_pos=fitcfg.sort_peaks_by_pos,
        # 자유 stages: 그대로 전달
        stages=fitcfg.stages,
        # 레거시 필드 유지(미사용)
        stage1=fitcfg.stage1,
        stage2=fitcfg.stage2,
        base_init=fitcfg.base_init,
        use_gpu=(False if app.force_cpu else fitcfg.use_gpu),
        x_fit_range=rng,
        x_label=f"x ({ssm.unit})" if ssm.unit else fitcfg.x_label,
        y_label=fitcfg.y_label,
        h=fitcfg.h,
        basic_three_only=fitcfg.basic_three_only,
    )

    # 출력 폴더
    run_tag = app.run_tag or in_path.stem
    outdir = Path(app.output_root) / f"{app.run_prefix}{run_tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir / "plots"
    if app.save_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    # ==== FitSpectrumMap 수집 버퍼 (항상 원본 축 길이 L로 할당) ====
    P = len(fitcfg.peaks)
    pos_buf    = np.full((N_total, P), np.nan, dtype=np.float32)
    width_buf  = np.full((N_total, P), np.nan, dtype=np.float32)
    height_buf = np.full((N_total, P), np.nan, dtype=np.float32)
    eta_buf    = np.full((N_total, P), np.nan, dtype=np.float32)
    base_buf   = np.full((N_total,),   np.nan, dtype=np.float32)
    valid_mask = np.zeros((N_total,), dtype=bool)

    # 선택 저장(용량)
    SAVE_RECON = True           # fitted 전체 스펙트럼 저장(N×L) — NPZ 내부 (권장)
    SAVE_PEAKS_3D = False       # 피크 분해 성분(N×P×L). 매우 큼(기본 False).
    recon_buf = np.full((N_total, L), np.nan, dtype=np.float32) if SAVE_RECON else None
    peaks_buf = np.full((N_total, P, L), np.nan, dtype=np.float32) if SAVE_PEAKS_3D else None
    peak_types_display: List[str] | None = None

    # 피팅 루프
    for k, idx in enumerate(idx_list, start=1):
        logging.info("Task %d/%d | idx=%d", k, len(idx_list), idx)
        # 1) 크롭된 축에서 피팅
        x_crop, y_crop, coord = prepare_xy_from_npz(ssm, idx, fitcfg.x_fit_range)
        targs = build_torch_args(fitcfg, x_crop)

        fitter = SpectrumFitter(targs, x_crop, y_crop, use_gpu=fitcfg.use_gpu)

        # 초기 플롯: 저장/미리보기 옵션에 따라 생성
        if app.save_plots or app.show_initial_preview:
            fig0 = fig_initial_guess(fitter, fitcfg)
            if app.show_initial_preview:
                action = preview_wait_for_key(fig0)
                if action == "abort":
                    logging.info("Aborted by user (Esc).")
                    logging.info("Total execution time: %s", _format_elapsed(time.perf_counter() - t0))
                    raise SystemExit(0)
            else:
                if app.save_plots:
                    fig0.savefig(plots_dir / f"{in_path.stem}_idx{idx}_initial.png", dpi=300)
                plt_close(fig0)

        # 2) 피팅 (스테이지 스케줄 전달)
        fitter.fit(fitcfg)

        # 결과 플롯: 저장 옵션에 따라 생성
        if app.save_plots:
            figf = fig_final_fit(fitter, fitcfg)
            figf.savefig(plots_dir / f"{in_path.stem}_idx{idx}.png", dpi=300)
            plt_close(figf)

            # loss/error 곡선 저장
            try:
                loss_hist = getattr(fitter, "loss_history", None)
                if loss_hist:
                    save_loss_plot(plots_dir, in_path.stem, idx, loss_hist)
            except Exception as e:
                logging.warning("loss plot save skipped: %s", e)

        params = fitter.params()
        recon_crop, peaks_crop = fitter.reconstructed()   # 크롭 축 기준
        baseline = float(params["base"][0, 0])

        # 3) 저장은 원본 축 전체에서 재합성 → 길이 불일치 방지
        recon_full, peaks_full = reconstruct_on_full_axis(params, ssm.axis.astype(np.float64, copy=False))

        # 버퍼 채움
        pos_buf[idx, :]    = params["pos"][0, :, 0]
        width_buf[idx, :]  = params["width"][0, :, 0]
        height_buf[idx, :] = params["height"][0, :, 0]
        eta_buf[idx, :]    = params["eta"][0, :, 0]
        base_buf[idx]      = baseline
        valid_mask[idx]    = True

        if SAVE_RECON:
            recon_buf[idx, :] = recon_full[0, :].astype(np.float32, copy=False)
        if SAVE_PEAKS_3D:
            peaks_buf[idx, :, :] = peaks_full[0, :, :].astype(np.float32, copy=False)

        if peak_types_display is None:
            peak_types_display = [str(t) for t in list(params["display_types"])]

        # ---- 보조 산출물 저장(옵션 일괄 제어) ----
        if app.save_extras:
            areas = compute_peak_areas(params, list(params["display_types"]))[0, :]
            save_peak_csv(outdir, in_path.stem, idx, params, areas, coord)
            export_txt_pair(outdir, in_path.stem, idx, x_crop, recon_crop, baseline, coord)
            csv_path = save_origin_ready_csv(str(outdir), f"{in_path.stem}_idx{idx}", x_crop, y_crop, peaks_crop, baseline)
            logging.info("Saved (extras): %s", csv_path)

    # ==== FitSpectrumMap 표준으로 저장 (항상 저장) ====
    meta = {
        "input_file": str(in_path),
        "x_fit_range": [CONFIG.xmin, CONFIG.xmax],  # 사용자가 요청한 값(내부 클램프는 모듈에서 수행)
        "basic_three_only": bool(fitcfg.basic_three_only),
        "notes": "StageSpec schedule; pV unified; recon on full axis; preview Enter/Esc; loss plot.",
        "stages": [
            {"name": s.name, "iters": s.iters, "lr": s.lr, "active": ("all" if isinstance(s.active, str) else list(s.active))}
            for s in (fitcfg.stages or [])
        ],
    }
    out_npz = outdir / f"{in_path.stem}__FitSpectrumMap_v1.npz"
    save_FitSpectrumMapNpz(
        str(out_npz),
        axis=ssm.axis,
        unit=ssm.unit or "",
        xy=ssm.xy,
        spectra_original=ssm.spectra,
        peak_types=peak_types_display or [p.type for p in fitcfg.peaks],
        pos=pos_buf, width=width_buf, height=height_buf, eta=eta_buf, base=base_buf,
        valid_mask=valid_mask, metadata=meta,
        recon=(recon_buf if SAVE_RECON else None),
        peaks_3d=(peaks_buf if SAVE_PEAKS_3D else None),
    )
    logging.info("Saved FitSpectrumMap NPZ: %s", out_npz)

    # ----- 총 실행 시간 출력 -----
    elapsed = time.perf_counter() - t0
    logging.info("Total execution time: %s", _format_elapsed(elapsed))
    logging.info("All done. Output: %s", outdir)

def main():
    run_with_config(CONFIG, FITCFG)

if __name__ == "__main__":
    main()
