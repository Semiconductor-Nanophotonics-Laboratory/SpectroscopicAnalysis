# fit_main.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import time
import numpy as np
import matplotlib.pyplot as plt
import torch

from process_module_StandardSpectrumMapNpz import load_StandardSpectrumMapNpz
from fit_module_StandardSpectrumMapNpz import (
    FitConfig, StageSpec, PeakSpec,
    build_torch_args, SpectrumFitter,
    prepare_xy_from_npz, select_indices,
    fig_initial_guess, fig_final_fit,
    compute_peak_areas, save_origin_ready_csv, export_text_spectra,
    save_FitSpectrumMapNpz, synthesize_pv,
    save_demo_peaks_block, format_demo_peaks_block_text,  # ← 초기 저장용 텍스트 생성 사용
)

# ============================= CONFIG ========================================

@dataclass(frozen=True)
class AppConfig:
    input_file: str = "7thR_60p_p1_whT4_m5_interp_600_3600_4_sgW7_P3_S2_normI_r3_blpeakstrip.npz"
    output_root: str = "./result"
    run_tag: str | None = None

    select_mode: str = "all"          # 'one' | 'all' | 'indices' | 'random'
    indices: List[int] = (0, 10, 20)
    k_random: int = 5
    seed: int = 42

    # ---- Initial guess (avg spectrum pre-fit) ----
    initial_guess: bool = False
    init_select_mode: str = "random"          # 'indices' | 'random'
    init_indices: List[int] = (0, 10, 20)
    init_k_random: int = 20
    init_seed: int = 123

    xmin: Optional[float] = 550.0
    xmax: Optional[float] = 3500.0

    force_cpu: bool = False
    verbosity: int = 1

    save_plots: bool = True
    save_extras: bool = False

BASIC_THREE_ONLY: bool = True

# --------- Peaks (D4, D1, D3, G, D′, 2D, D+D2, 2D2) ----------
DEMO_PEAKS: List[PeakSpec] = [
    PeakSpec(type="pseudovoigt", pos=1176.2, width=157.1, height=13.7092, pos_limit=4,  pos_limit_reg=6, width_limit=20, width_limit_reg=6, height_limit=0.3, height_limit_reg=1, extra={"eta":0.000}),
    PeakSpec(type="lorentzian", pos=1331.1, width=91.7, height=128.171, pos_limit=7,  pos_limit_reg=7, width_limit=20, width_limit_reg=4, height_limit=0.3, height_limit_reg=1),
    PeakSpec(type="pseudovoigt", pos=1456.4, width=125.9, height=14.1072, pos_limit=8,  pos_limit_reg=7, width_limit=20, width_limit_reg=5, height_limit=0.3, height_limit_reg=1, extra={"eta":0.000}),
    PeakSpec(type="lorentzian", pos=1578.0, width=105.2, height=40.3363, pos_limit=4,  pos_limit_reg=3, width_limit=20, width_limit_reg=6, height_limit=0.3, height_limit_reg=10),
    PeakSpec(type="lorentzian", pos=1608.9, width=62.7, height=22.349, pos_limit=4,  pos_limit_reg=3, width_limit=20, width_limit_reg=6, height_limit=0.3, height_limit_reg=10),
    PeakSpec(type="pseudovoigt", pos=2477.8, width=195.3, height=3.30412, pos_limit=4,  pos_limit_reg=3, width_limit=20, width_limit_reg=3, height_limit=0.3, height_limit_reg=1, extra={"eta":0.710}),
    PeakSpec(type="lorentzian", pos=2658.5, width=119.8, height=22.3022, pos_limit=4,  pos_limit_reg=3, width_limit=20, width_limit_reg=3, height_limit=0.3, height_limit_reg=1),
    PeakSpec(type="lorentzian", pos=2912.8, width=156.3, height=7.27899, pos_limit=4,  pos_limit_reg=3, width_limit=8, width_limit_reg=3, height_limit=0.3, height_limit_reg=1),
]

FITCFG = FitConfig(
    peaks=DEMO_PEAKS,
    sort_peaks_by_pos=True,
    stages=[
        StageSpec(name="main",   iters=2000, lr=0.003, active=[0, 1, 5, 6, 7]),
        StageSpec(name="defect", iters=8000, lr=0.003, active="all"),
    ],
    base_init=0.0,
    use_gpu=True,
    x_fit_range=None,
    x_label="x",
    y_label="Intensity",
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
    L = int(x_full.shape[0])
    P = int(params["pos"].shape[1])
    x_t = torch.from_numpy(x_full.astype(np.float32, copy=False)).unsqueeze(0)  # (1,L)
    pos   = torch.from_numpy(params["pos"][0, :, 0].astype(np.float32, copy=False)).view(1, P, 1)
    width = torch.from_numpy(params["width"][0, :, 0].astype(np.float32, copy=False)).view(1, P, 1)
    height= torch.from_numpy(params["height"][0, :, 0].astype(np.float32, copy=False)).view(1, P, 1)
    eta   = torch.from_numpy(params["eta"][0, :, 0].astype(np.float32, copy=False)).view(1, P, 1)
    base  = torch.tensor([[float(params["base"][0, 0])]], dtype=torch.float32)
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

# ============================== INITIAL GUESS ================================

def run_initial_guess(app: AppConfig, base_cfg: FitConfig, ssm, outdir: Path, plots_dir: Path, in_stem: str) -> List[PeakSpec]:
    """
    평균 스펙트럼 1회 피팅으로 초기 파라미터 추정.
    - DEMO_PEAKS 텍스트 파일과 프리뷰 PNG를 **먼저 저장**한 뒤 Enter/Esc 대기.
    - Esc로 중단해도 파일은 남음.
    """
    logging.info("Initial-guess pass: building averaged spectrum ...")

    # 선택 대상
    if app.init_select_mode == "indices":
        init_idx_list = [i for i in app.init_indices if 0 <= i < ssm.spectra.shape[0]]
    elif app.init_select_mode == "random":
        rng = np.random.default_rng(app.init_seed)
        k = min(app.init_k_random, ssm.spectra.shape[0])
        init_idx_list = sorted(rng.choice(ssm.spectra.shape[0], size=k, replace=False).tolist())
    else:
        raise ValueError("init_select_mode must be 'indices' or 'random'.")

    if not init_idx_list:
        logging.warning("Initial-guess skipped (no indices).")
        return list(base_cfg.peaks)

    rng_tuple: Tuple[Optional[float], Optional[float]] = (CONFIG.xmin, CONFIG.xmax)

    # 공통 x_crop 확보 및 평균
    x_crop, y0, _ = prepare_xy_from_npz(ssm, init_idx_list[0], rng_tuple)
    ys = [y0.reshape(-1)]
    for i in init_idx_list[1:]:
        xi, yi, _ = prepare_xy_from_npz(ssm, i, rng_tuple)
        if xi.shape != x_crop.shape or not np.allclose(xi, x_crop, rtol=0, atol=1e-9):
            raise RuntimeError("Initial-guess: x-crop mismatch among selected spectra.")
        ys.append(yi.reshape(-1))
    y_avg = np.mean(np.vstack(ys), axis=0, dtype=np.float64).astype(np.float32, copy=False).reshape(1, -1)

    # 피팅
    targs = build_torch_args(base_cfg, x_crop)
    fitter = SpectrumFitter(targs, x_crop, y_avg, use_gpu=base_cfg.use_gpu)
    logging.info("Initial-guess fitting start (stages=%d) ...", len(base_cfg.stages or []))
    fitter.fit(base_cfg)
    logging.info("Initial-guess fitting done.")

    # ---- 결과 저장을 먼저 수행 (Esc여도 남도록) ----
    params = fitter.params()
    # DEMO_PEAKS 텍스트 저장
    txt = format_demo_peaks_block_text(params, base_cfg.peaks, base_cfg.sort_peaks_by_pos)
    (outdir / f"DEMO_PEAKS_{in_stem}_INITIAL.txt").write_text(txt, encoding="utf-8")
    logging.info("Saved initial DEMO_PEAKS: %s", outdir / f"DEMO_PEAKS_{in_stem}_INITIAL.txt")

    # 프리뷰 PNG 저장
    figf = fig_final_fit(fitter, base_cfg)
    if app.save_plots:
        figf.savefig(plots_dir / f"{in_stem}_INITIAL_guess_preview.png", dpi=250)
        logging.info("Saved initial preview: %s", plots_dir / f"{in_stem}_INITIAL_guess_preview.png")

    # 사용자 확인 (Enter=계속, Esc=중단)
    action = preview_wait_for_key(figf)
    if action == "abort":
        logging.info("Aborted by user (Esc) at initial-guess stage.")
        raise SystemExit(0)

    # ---- 새 PeakSpec 리스트 생성 (limit류 유지, pos/width/height/eta 반영) ----
    peaks_sorted = sorted(base_cfg.peaks, key=lambda p: p.pos) if base_cfg.sort_peaks_by_pos else list(base_cfg.peaks)
    disp_types = list(params["display_types"])
    pos = params["pos"][0, :, 0]; width = params["width"][0, :, 0]
    height = params["height"][0, :, 0]; eta = params["eta"][0, :, 0]

    new_peaks: List[PeakSpec] = []
    for j, (p0, tdisp) in enumerate(zip(peaks_sorted, disp_types)):
        t = ("pseudovoigt" if tdisp.lower().startswith("pseudo")
             else "lorentzian" if "lorentz" in tdisp.lower()
             else "gaussian")
        extra = ({"eta": float(eta[j])} if t == "pseudovoigt" else None)
        new_peaks.append(PeakSpec(
            type=t, pos=float(pos[j]), width=float(width[j]), height=float(height[j]),
            pos_limit=p0.pos_limit, pos_limit_reg=p0.pos_limit_reg,
            width_limit=p0.width_limit, width_limit_reg=p0.width_limit_reg,
            height_limit=p0.height_limit, height_limit_reg=p0.height_limit_reg,
            extra=extra,
        ))
    return new_peaks

# ============================== RUNTIME ======================================

def run_with_config(app: AppConfig, fitcfg: FitConfig) -> None:
    setup_logging(app.verbosity)
    t0 = time.perf_counter()

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

    in_path = Path(app.input_file)
    if not in_path.exists():
        raise FileNotFoundError(f"Not found: {in_path}")
    ssm = load_StandardSpectrumMapNpz(str(in_path))
    N_total, L = ssm.spectra.shape
    logging.info("Loaded NPZ | spectra=%s, axis=%s, unit=%s", ssm.spectra.shape, ssm.axis.shape, ssm.unit)

    # 출력 폴더
    in_stem = in_path.stem
    tag_prefix = (app.run_tag + "_") if app.run_tag else ""
    outdir = Path(app.output_root) / f"{tag_prefix}{in_stem}"
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir / "plots"
    if app.save_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    # ---- Initial guess (필요 시) ----
    updated_peaks = list(fitcfg.peaks)
    if app.initial_guess:
        try:
            updated_peaks = run_initial_guess(app, fitcfg, ssm, outdir, plots_dir, in_stem)
            logging.info("Initial-guess: peaks updated from averaged-spectrum.")
        except SystemExit:
            logging.info("Total execution time: %s", _format_elapsed(time.perf_counter() - t0))
            return
        except Exception as e:
            logging.warning("Initial-guess failed (%s). Continue with original peaks.", e)

    # 선택
    idx_list = select_indices(N_total, app.select_mode, indices=app.indices, k_random=app.k_random, seed=app.seed)
    if not idx_list:
        logging.warning("No indices selected. Exit.")
        logging.info("Total execution time: %s", _format_elapsed(time.perf_counter() - t0))
        return

    rng: Tuple[Optional[float], Optional[float]] = (app.xmin, app.xmax)

    fitcfg = FitConfig(
        peaks=updated_peaks,
        sort_peaks_by_pos=fitcfg.sort_peaks_by_pos,
        stages=fitcfg.stages,
        stage1=fitcfg.stage1, stage2=fitcfg.stage2,
        base_init=fitcfg.base_init,
        use_gpu=(False if app.force_cpu else fitcfg.use_gpu),
        x_fit_range=rng,
        x_label=f"x ({ssm.unit})" if ssm.unit else fitcfg.x_label,
        y_label=fitcfg.y_label,
        basic_three_only=fitcfg.basic_three_only,
    )

    # 버퍼
    P = len(fitcfg.peaks)
    pos_buf    = np.full((N_total, P), np.nan, dtype=np.float32)
    width_buf  = np.full((N_total, P), np.nan, dtype=np.float32)
    height_buf = np.full((N_total, P), np.nan, dtype=np.float32)
    eta_buf    = np.full((N_total, P), np.nan, dtype=np.float32)
    base_buf   = np.full((N_total,),   np.nan, dtype=np.float32)
    valid_mask = np.zeros((N_total,), dtype=bool)

    SAVE_RECON = True
    SAVE_PEAKS_3D = False
    recon_buf = np.full((N_total, L), np.nan, dtype=np.float32) if SAVE_RECON else None
    peaks_buf = np.full((N_total, P, L), np.nan, dtype=np.float32) if SAVE_PEAKS_3D else None
    peak_types_display: List[str] | None = None

    for k, idx in enumerate(idx_list, start=1):
        logging.info("Task %d/%d | idx=%d", k, len(idx_list), idx)
        x_crop, y_crop, coord = prepare_xy_from_npz(ssm, idx, fitcfg.x_fit_range)
        targs = build_torch_args(fitcfg, x_crop)
        fitter = SpectrumFitter(targs, x_crop, y_crop, use_gpu=fitcfg.use_gpu)

        fitter.fit(fitcfg)

        if app.save_plots:
            figf = fig_final_fit(fitter, fitcfg)
            figf.savefig(plots_dir / f"{in_stem}_idx{idx}.png", dpi=300)
            plt_close(figf)
            try:
                loss_hist = getattr(fitter, "loss_history", None)
                if loss_hist:
                    save_loss_plot(plots_dir, in_stem, idx, loss_hist)
            except Exception as e:
                logging.warning("loss plot save skipped: %s", e)

        params = fitter.params()
        recon_crop, peaks_crop = fitter.reconstructed()
        baseline = float(params["base"][0, 0])

        recon_full, peaks_full = reconstruct_on_full_axis(params, ssm.axis.astype(np.float64, copy=False))

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

        if app.save_extras:
            areas = compute_peak_areas(params, list(params["display_types"]))[0, :]
            save_peak_csv(outdir, in_stem, idx, params, areas, coord)
            export_txt_pair(outdir, in_stem, idx, x_crop, recon_crop, baseline, coord)
            csv_path = save_origin_ready_csv(str(outdir), f"{in_stem}_idx{idx}", x_crop, y_crop, peaks_crop, baseline)
            _ = save_demo_peaks_block(outdir, in_stem, idx, params, fitcfg.peaks, fitcfg.sort_peaks_by_pos)
            logging.info("Saved (extras): %s", csv_path)

    meta = {
        "input_file": str(in_path),
        "x_fit_range": [CONFIG.xmin, CONFIG.xmax],
        "basic_three_only": bool(fitcfg.basic_three_only),
        "notes": "Initial-guess avg fit → DEMO_PEAKS saved before Enter/Esc; pV unified; recon on full axis; loss plot",
        "stages": [
            {"name": s.name, "iters": s.iters, "lr": s.lr, "active": ("all" if isinstance(s.active, str) else list(s.active))}
            for s in (fitcfg.stages or [])
        ],
        "initial_guess": {
            "enabled": bool(CONFIG.initial_guess),
            "mode": CONFIG.init_select_mode,
            "k_random": CONFIG.init_k_random,
            "indices": list(CONFIG.init_indices),
            "seed": CONFIG.init_seed,
        }
    }
    out_npz = outdir / f"{in_stem}__FitSpectrumMap_v1.npz"
    save_FitSpectrumMapNpz(
        str(out_npz),
        axis=ssm.axis, unit=ssm.unit or "", xy=ssm.xy, spectra_original=ssm.spectra,
        peak_types=peak_types_display or [p.type for p in fitcfg.peaks],
        pos=pos_buf, width=width_buf, height=height_buf, eta=eta_buf, base=base_buf,
        valid_mask=valid_mask, metadata=meta,
        recon=(recon_buf if SAVE_RECON else None),
        peaks_3d=(peaks_buf if SAVE_PEAKS_3D else None),
    )
    logging.info("Saved FitSpectrumMap NPZ: %s", out_npz)

    elapsed = time.perf_counter() - t0
    logging.info("Total execution time: %s", _format_elapsed(elapsed))
    logging.info("All done. Output: %s", outdir)

def main():
    run_with_config(CONFIG, FITCFG)

if __name__ == "__main__":
    main()
