from __future__ import annotations

import argparse
import os
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(_THIS_DIR / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

from cislunar_hohmann_numba_solver import (
    CislunarConfig,
    build_animation_data,
)

BASE_DIR = Path(__file__).resolve().parent


def make_animation(
    output: Path,
    cfg: CislunarConfig,
    fps: int = 20,
    n_frames: int = 720,
    trail: int = 120,
):
    data = build_animation_data(cfg)

    t = data["t"]
    sc = data["sc_xy_km"]
    moon = data["moon_xy_km"]
    hoh_analytic = data["hohmann_analytic_km"]
    eps_arr = data["eps_m"]
    phase_arr = data["phase"]
    sc_visible = data.get("sc_visible", np.ones(len(t), dtype=bool))

    r_soi = data["r_soi_km"]
    r_earth = data["r_earth_km"]
    r_moon = data["r_moon_km"]

    hoh_idxs = np.where(phase_arr == "Hohmann Transfer")[0]
    coast_idxs = np.where(phase_arr == "Coast")[0]
    i_hoh_start = int(hoh_idxs[0]) if hoh_idxs.size > 0 else 0
    i_coast_start = int(coast_idxs[0]) if coast_idxs.size > 0 else len(t) - 1

    # Time-warp sampling:
    # - Slow down pre-coast (launch -> big circular orbit end)
    # - Smoothly blend to normal speed around coast start to avoid abrupt jumps.
    t_coast = t[i_coast_start]
    blend_window = 4.0 * 3600.0
    w_pre = 5.2
    w_post = 1.0

    z = (t - (t_coast - blend_window)) / max(2.0 * blend_window, 1.0)
    z = np.clip(z, 0.0, 1.0)
    smooth = z * z * (3.0 - 2.0 * z)  # smoothstep
    w = w_pre * (1.0 - smooth) + w_post * smooth

    # Allocate denser frames to final lunar operations so orbiting/descent is visible.
    phase_mult = np.ones_like(w)
    phase_mult[np.isin(phase_arr, ["Lunar Orbiting"])] = 4.0
    phase_mult[np.isin(phase_arr, ["Deorbit & Coast"])] = 5.0
    phase_mult[np.isin(phase_arr, ["Powered Descent"])] = 8.0
    # Keep post-landing playback speed consistent with powered descent.
    phase_mult[np.isin(phase_arr, ["Landing Complete"])] = 8.0
    w = w * phase_mult

    s = np.zeros_like(t)
    ds = 0.5 * (w[:-1] + w[1:]) * np.diff(t)
    s[1:] = np.cumsum(ds)

    s_sample = np.linspace(s[0], s[-1], n_frames)
    idx = np.searchsorted(s, s_sample, side="left")
    idx = np.clip(idx, 0, len(t) - 1)
    idx = np.unique(idx)

    fig, ax = plt.subplots(figsize=(9, 9))

    xy_all = np.vstack([sc, moon])
    span = np.max(np.abs(xy_all))
    lim = 1.05 * max(span, r_earth * 1.3)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.set_xlabel(r"$x\,[\mathrm{km}]$")
    ax.set_ylabel(r"$y\,[\mathrm{km}]$")
    ax.set_title("Launch + Hohmann + Coast + Lunar Capture")

    ax.plot(moon[:, 0], moon[:, 1], "--", lw=0.7, color="gray", alpha=0.32)
    ax.plot(sc[:, 0], sc[:, 1], lw=0.8, color="gray", alpha=0.55)
    ax.plot(hoh_analytic[:, 0], hoh_analytic[:, 1], "--", lw=1.0, color="tab:purple", alpha=0.95)

    # Smaller points to reduce overlap.
    earth_dot, = ax.plot([0.0], [0.0], marker="o", ms=2.6, color="tab:blue", ls="None")
    sc_dot, = ax.plot([], [], marker="o", ms=1.8, color="tab:red", ls="None")
    moon_dot, = ax.plot([], [], marker="o", ms=2.4, color="tab:orange", ls="None")

    trail_line, = ax.plot([], [], lw=1.2, color="tab:red", alpha=0.95)
    soi_patch = plt.Circle((0.0, 0.0), r_soi, fill=False, ls=":", lw=1.0, color="tab:green", alpha=0.85)
    moon_patch = plt.Circle((0.0, 0.0), r_moon, fill=False, lw=0.8, color="tab:orange", alpha=0.90)
    ax.add_patch(soi_patch)
    ax.add_patch(moon_patch)

    phase_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top")
    info_text = ax.text(0.02, 0.90, "", transform=ax.transAxes, va="top")

    legend_handles = [
        Line2D([0], [0], marker="o", color="tab:blue", ls="None", ms=2.6, label="Earth"),
        Line2D([0], [0], marker="o", color="tab:orange", ls="None", ms=2.4, label="Moon"),
        Line2D([0], [0], marker="o", color="tab:red", ls="None", ms=1.8, label="Spacecraft"),
        Line2D([0], [0], color="tab:green", ls=":", lw=1.0, label="Moon SOI"),
        Line2D([0], [0], color="gray", ls="-", lw=0.8, label="Spacecraft trajectory"),
        Line2D([0], [0], color="tab:purple", ls="--", lw=1.0, label="Hohmann analytic"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    lim_global = lim
    lim_close = min(lim_global, max(25000.0, 1.8 * (r_earth + data["summary"]["closest_earth_distance_km"] * 0.05)))
    # Use one fixed lunar-focused view window to avoid apparent zoom jumps
    # during deorbit/descent/landing stages.
    lim_lunar_fixed = max(9000.0, 5.0 * r_moon)
    zoom_transition_frames = max(25, n_frames // 12)

    def init():
        trail_line.set_data([], [])
        sc_dot.set_data([], [])
        moon_dot.set_data([], [])
        phase_text.set_text("")
        info_text.set_text("")
        return trail_line, sc_dot, moon_dot, soi_patch, moon_patch, phase_text, info_text

    def update(k: int):
        i = idx[k]
        if sc_visible[i]:
            i_end = i
        else:
            vis_prev = np.where(sc_visible[: i + 1])[0]
            i_end = int(vis_prev[-1]) if vis_prev.size > 0 else i
        i0 = max(0, i_end - trail)

        x_sc, y_sc = sc[i_end]
        x_m, y_m = moon[i]

        lunar_focus = phase_arr[i] in ("Lunar Orbiting", "Deorbit & Coast", "Powered Descent", "Landing Complete")
        seg_sc = sc[i0 : i_end + 1]
        seg_m = moon[i0 : i_end + 1]
        if phase_arr[i] == "Landing Complete":
            # After landing: remove trail and keep only landed marker on the surface.
            trail_line.set_data([], [])
        elif lunar_focus:
            # Display trail in a moon-translating frame to avoid apparent straight-line artifacts.
            trail_x = seg_sc[:, 0] - seg_m[:, 0] + x_m
            trail_y = seg_sc[:, 1] - seg_m[:, 1] + y_m
            trail_line.set_data(trail_x, trail_y)
        else:
            trail_line.set_data(seg_sc[:, 0], seg_sc[:, 1])
        if sc_visible[i]:
            sc_dot.set_data([x_sc], [y_sc])
        elif phase_arr[i] == "Landing Complete":
            # Keep spacecraft attached on the lunar surface after touchdown.
            sc_dot.set_data([sc[i, 0]], [sc[i, 1]])
        else:
            sc_dot.set_data([], [])
        if lunar_focus:
            moon_dot.set_data([], [])
            moon_patch.set_fill(True)
            moon_patch.set_alpha(0.30)
        else:
            moon_dot.set_data([x_m], [y_m])
            moon_patch.set_fill(False)
            moon_patch.set_alpha(0.90)

        soi_patch.center = (x_m, y_m)
        moon_patch.center = (x_m, y_m)

        # Camera effect:
        # - Earth-centered for early stages.
        # - Moon-centered zoom for multi-orbit lunar phase + descent/landing.
        if phase_arr[i] in ("Lunar Orbiting", "Deorbit & Coast", "Powered Descent", "Landing Complete"):
            current_lim = lim_lunar_fixed
            ax.set_xlim(x_m - current_lim, x_m + current_lim)
            ax.set_ylim(y_m - current_lim, y_m + current_lim)
        else:
            if i < i_coast_start:
                current_lim = lim_close
            elif i < i_coast_start + zoom_transition_frames:
                z = (i - i_coast_start) / max(zoom_transition_frames, 1)
                current_lim = lim_close * (1.0 - z) + lim_global * z
            else:
                current_lim = lim_global
            ax.set_xlim(-current_lim, current_lim)
            ax.set_ylim(-current_lim, current_lim)

        day = t[i] / 86400.0
        phase_text.set_text(f"Stage: {phase_arr[i]}")

        eps = eps_arr[i]
        eps_text = r"$\epsilon_M = \mathrm{N/A}$"
        if np.isfinite(eps):
            eps_text = rf"$\epsilon_M = {eps/1e6:.3f}\,\mathrm{{km^2/s^2}}$"

        if sc_visible[i]:
            info_text.set_text(
                "\n".join(
                    [
                        rf"$t = {day:.3f}\,\mathrm{{day}}$",
                        rf"$r_{{SM}} = {np.linalg.norm(sc[i]-moon[i]):.1f}\,\mathrm{{km}}$",
                        eps_text,
                    ]
                )
            )
        else:
            info_text.set_text(
                "\n".join(
                    [
                        rf"$t = {day:.3f}\,\mathrm{{day}}$",
                        r"Status: Landing complete",
                        r"$r_{SM} = \mathrm{N/A}$",
                    ]
                )
            )

        return trail_line, sc_dot, moon_dot, soi_patch, moon_patch, phase_text, info_text

    playback_speed = 1.10  # Moderate overall playback speed.
    ani = FuncAnimation(
        fig,
        update,
        frames=len(idx),
        init_func=init,
        interval=1000 / (fps * playback_speed),
        blit=False,
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix.lower() == ".gif":
        # [TEMP DISABLED] GIF saving is intentionally commented out.
        #ani.save(output, writer=PillowWriter(fps=fps))
        print(f"[TEMP DISABLED] GIF saving commented out. Intended path: {output}")
    elif output.suffix.lower() == ".mp4":
        ani.save(output, writer=FFMpegWriter(fps=fps, bitrate=2200))
    else:
        raise ValueError("Output suffix must be .gif or .mp4")

    plt.show()
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Create cislunar Hohmann-to-capture animation.")
    p.add_argument(
        "--output",
        type=str,
        default=str(BASE_DIR / "cislunar_capture_animation.gif"),
        help="Output .gif or .mp4 (default: script directory)",
    )
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--frames", type=int, default=720)
    p.add_argument("--dt", type=float, default=20.0, help="Integrator step [s]")
    p.add_argument("--phase-bias-deg", type=float, default=10.0)
    p.add_argument("--disable-loi", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = CislunarConfig(
        dt=args.dt,
        moon_phase_bias_deg=args.phase_bias_deg,
        enable_loi=not args.disable_loi,
    )

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = BASE_DIR / output_path

    make_animation(output=output_path, cfg=cfg, fps=args.fps, n_frames=args.frames)
    print(f"Animation target path: {output_path}")


if __name__ == "__main__":
    main()
