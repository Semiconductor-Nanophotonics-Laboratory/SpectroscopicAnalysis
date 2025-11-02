from pathlib import Path
import numpy as np
from process_module_StandardSpectrumMapNpz import (
    load_StandardSpectrumMapNpz,
    save_StandardSpectrumMap_toNpz,
    process_Despike,                 # << 통합 디스패처
    process_Roi_Interpolate,
    process_SmoothSavGol,
    process_Normalize,
    process_Baseline,
    save_QA_SpectrumGrid,
)

def _fmt_token(x):
    if x is None:
        return "NA"
    try:
        xf = float(x)
        return str(int(xf)) if xf.is_integer() else f"{xf:g}"
    except Exception:
        return str(x)

def main():
    in_files = [
        "7thR_0p_p1.npz",
        "7thR_10p_p1.npz",
        "7thR_20p_p1.npz",
        "7thR_30p_p1.npz",
        "7thR_40p_p1.npz",
        "7thR_50p_p1.npz",
        "7thR_60p_p1.npz",
        "7thR_70p_p1.npz",
        "7thR_80p_p1.npz",
        "7thR_90p_p1.npz",
        "7thR_100p_p1.npz",
    ]

    out_dir_npz = Path("./npz_processed"); out_dir_npz.mkdir(parents=True, exist_ok=True)
    out_dir_img = Path("./qa_images");     out_dir_img.mkdir(parents=True, exist_ok=True)

    # ===== 스위치 =====
    USE_SPIKE      = True
    USE_ROI        = True
    USE_SMOOTH     = True
    USE_NORMALIZE  = True
    BASELINE_MODE  = "peakstrip"      # 'none' | 'original' | 'peakstrip'

    # ===== Spike 선택 =====
    SPIKE_MODE     = "wh"             # 'simple' | 'wh' | 'none'
    # simple params
    sf_threshold   = 40
    # Whitaker–Hayes params
    wh_threshold   = 4.0
    wh_m           = 5

    # ===== ROI / Interpolate =====
    xmin, xmax, xstep = 600, 3600, 4.0
    use_interp = True
    roi_round_decimals = None      # << 추가: ROI 후 intensity 반올림 자리수 (None이면 미적용)
    
    # ===== Smoothing =====
    sg_window, sg_polyorder, sg_mode, sg_strength = 7, 3, "interp", 2

    # ===== Normalize =====
    norm_mode = "instance"      # "global" or "instance"
    norm_round_decimals = 3     # << 추가: Normalize 후 intensity 반올림 자리수 (None이면 미적용)


    # ===== Normalize =====
    norm_mode = "instance"  # "global" or "instance"

    # ===== Baseline =====
    bl_orig_window, bl_orig_step = 500, 150
    bl_ps_wmin, bl_ps_wmax, bl_ps_poly = 121, 1111, 0

    for f in in_files:
        src = load_StandardSpectrumMapNpz(f)

        # A) Raw
        save_QA_SpectrumGrid(src, src, str(out_dir_img / f"{Path(f).stem}_A_raw.png"),
                             suptitle="A) Raw snapshot", yscale="linear")

        curr = src
        tags = []

        # 1) Spike (선택)
        if USE_SPIKE and SPIKE_MODE != "none":
            nxt = process_Despike(curr, mode=SPIKE_MODE,
                                  sf_threshold=sf_threshold,
                                  wh_threshold=wh_threshold,
                                  wh_m=wh_m)
            save_QA_SpectrumGrid(curr, nxt, str(out_dir_img / f"{Path(f).stem}_B_raw_vs_spike_{SPIKE_MODE}.png"),
                                 suptitle=f"B) Raw vs Despike ({SPIKE_MODE})", yscale="log")
            curr = nxt
            if SPIKE_MODE == "simple":
                tags.append(f"sfT{sf_threshold}")
            else:
                tags.append(f"whT{wh_threshold:g}_m{wh_m}")
        else:
            tags.append("sfOFF")

        # 2) ROI / Interpolate
        if USE_ROI:
            nxt = process_Roi_Interpolate(
                curr, xmin=xmin, xmax=xmax, xstep=xstep, interpolate=use_interp,
                round_decimals=roi_round_decimals  # << 전달
            )
            mode  = "interp" if use_interp and xstep and xstep > 0 else "slice"
            save_QA_SpectrumGrid(curr, nxt, str(out_dir_img / f"{Path(f).stem}_C_prev_vs_{mode}.png"),
                                suptitle=f"C) Prev vs {mode}", yscale="linear")
            curr = nxt
            tag_r = f"_r{roi_round_decimals}" if roi_round_decimals is not None else ""
            tags.append(f"{mode}_{_fmt_token(xmin)}_{_fmt_token(xmax)}_{_fmt_token(xstep)}{tag_r}")
        else:
            tags.append("roiOFF")

        # 3) Smooth
        if USE_SMOOTH:
            nxt = process_SmoothSavGol(curr, window_length=sg_window, polyorder=sg_polyorder,
                                       mode=sg_mode, strength=sg_strength)
            save_QA_SpectrumGrid(curr, nxt, str(out_dir_img / f"{Path(f).stem}_D_prev_vs_smooth.png"),
                                 suptitle=f"D) Prev vs Smooth (win={sg_window}, poly={sg_polyorder}, str={sg_strength})",
                                 yscale="linear")
            curr = nxt
            tags.append(f"sgW{sg_window}_P{sg_polyorder}_S{sg_strength}")
        else:
            tags.append("sgOFF")

        # 4) Normalize
        if USE_NORMALIZE:
            nxt = process_Normalize(curr, mode=norm_mode, round_decimals=norm_round_decimals)  # << 전달
            save_QA_SpectrumGrid(curr, nxt, str(out_dir_img / f"{Path(f).stem}_E_prev_vs_norm_{norm_mode}.png"),
                                suptitle=f"E) Prev vs Normalize ({norm_mode})", yscale="log")
            curr = nxt
            tag_nr = f"_r{norm_round_decimals}" if norm_round_decimals is not None else ""
            tags.append(f"norm{norm_mode[0].upper()}{tag_nr}")
        else:
            tags.append("normOFF")

        # 5) Baseline
        if BASELINE_MODE.lower().strip() != "none":
            nxt = process_Baseline(
                curr,
                mode=BASELINE_MODE,
                orig_window_size=bl_orig_window,
                orig_step_size=bl_orig_step,
                ps_window_min=bl_ps_wmin,
                ps_window_max=bl_ps_wmax,
                ps_polyorder=bl_ps_poly,
            )
            save_QA_SpectrumGrid(curr, nxt, str(out_dir_img / f"{Path(f).stem}_F_prev_vs_baseline_{BASELINE_MODE}.png"),
                                 suptitle=f"F) Prev vs Baseline ({BASELINE_MODE})", yscale="linear")
            curr = nxt
            tags.append(f"bl{BASELINE_MODE}")
        else:
            tags.append("blOFF")

        # 저장
        out_npz = out_dir_npz / f"{Path(f).stem}_{'_'.join(tags)}.npz"
        save_StandardSpectrumMap_toNpz(curr, str(out_npz), compressed=True)

        step_est = np.median(np.diff(curr.axis)) if curr.axis.size > 1 else np.nan
        print(f"[OK] {Path(f).name} -> {out_npz.name}  "
              f"(N={curr.shape[0]}, M={curr.shape[1]}, axis[{curr.axis[0]:.3g}..{curr.axis[-1]:.3g}], "
              f"step≈{step_est:.3g}, unit='{curr.unit}')")

if __name__ == "__main__":
    main()
