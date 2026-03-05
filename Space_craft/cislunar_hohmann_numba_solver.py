from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp


@dataclass
class HohmannConfig:
    """Classic two-body Hohmann transfer config."""

    mu: float = 3.986004418e14
    r1: float = 7000e3
    r2: float = 42164e3
    dt: float = 5.0


@dataclass
class CislunarConfig:
    """Earth-Moon patched model built on Hohmann-style TLI."""

    mu_earth: float = 3.986004418e14
    mu_moon: float = 4.9048695e12

    r_earth: float = 6378.1363e3
    r_moon: float = 1737.4e3
    r_earth_moon: float = 384400e3
    moon_period: float = 27.321661 * 86400.0

    leo_altitude: float = 200e3
    dt: float = 20.0

    # TLI from Hohmann design to Moon orbital radius.
    dv1_scale: float = 1.0
    moon_phase_bias_deg: float = 10.0

    # Coast until slightly past nominal arrival.
    t_final_factor: float = 1.25

    # Capture maneuver near first perilune (optional impulsive LOI).
    enable_loi: bool = True
    target_circular_factor: float = 1.0  # 1.0: circular at perilune radius
    loi_dv_max: float = 1200.0           # m/s, soft cap for realism
    loi_burn_duration_s: float = 900.0   # finite LOI burn duration for smooth epsilon transition
    lunar_orbit_cycles_before_descent: float = 3.0
    descent_max_days: float = 1.2
    deorbit_entry_altitude: float = 220e3
    deorbit_perilune_altitude: float = 15e3
    deorbit_burn_duration_s: float = 1200.0
    landing_a_max: float = 6.5            # m/s^2, powered descent accel cap
    post_landing_hold_days: float = 0.35  # keep animating Moon after landing marker disappears


def hohmann_analytic(mu: float, r1: float, r2: float) -> Dict[str, float]:
    if r1 <= 0.0 or r2 <= 0.0:
        raise ValueError("r1 and r2 must be positive")

    a_t = 0.5 * (r1 + r2)
    v_circ_1 = np.sqrt(mu / r1)
    v_circ_2 = np.sqrt(mu / r2)

    v_peri_t = np.sqrt(mu * (2.0 / r1 - 1.0 / a_t))
    v_apo_t = np.sqrt(mu * (2.0 / r2 - 1.0 / a_t))

    dv1 = v_peri_t - v_circ_1
    dv2 = v_circ_2 - v_apo_t
    tof = np.pi * np.sqrt(a_t**3 / mu)

    return {
        "a_t": a_t,
        "v_circ_1": v_circ_1,
        "v_circ_2": v_circ_2,
        "v_peri_t": v_peri_t,
        "v_apo_t": v_apo_t,
        "dv1": dv1,
        "dv2": dv2,
        "dv_total": abs(dv1) + abs(dv2),
        "tof": tof,
    }


@njit(cache=True)
def _norm2(x: float, y: float) -> float:
    return np.sqrt(x * x + y * y)


@njit(cache=True)
def _moon_state_2d(t: float, r_em: float, omega_m: float, phase0: float):
    th = phase0 + omega_m * t
    c = np.cos(th)
    s = np.sin(th)

    mx = r_em * c
    my = r_em * s
    mvx = -r_em * omega_m * s
    mvy = r_em * omega_m * c
    return mx, my, mvx, mvy


@njit(cache=True)
def _two_body_rhs_2d(y: np.ndarray, mu: float) -> np.ndarray:
    x, yv, vx, vy = y
    r = _norm2(x, yv)
    r3 = max(r * r * r, 1e3)

    out = np.empty(4)
    out[0] = vx
    out[1] = vy
    out[2] = -mu * x / r3
    out[3] = -mu * yv / r3
    return out


@njit(cache=True)
def _cislunar_rhs_2d(
    t: float,
    y: np.ndarray,
    mu_earth: float,
    mu_moon: float,
    r_em: float,
    omega_m: float,
    phase0: float,
) -> np.ndarray:
    x, yv, vx, vy = y

    mx, my, _, _ = _moon_state_2d(t, r_em, omega_m, phase0)

    re = _norm2(x, yv)
    re3 = max(re * re * re, 1e3)

    dxm = x - mx
    dym = yv - my
    rm = _norm2(dxm, dym)
    rm3 = max(rm * rm * rm, 1e3)

    ax = -mu_earth * x / re3 - mu_moon * dxm / rm3
    ay = -mu_earth * yv / re3 - mu_moon * dym / rm3

    out = np.empty(4)
    out[0] = vx
    out[1] = vy
    out[2] = ax
    out[3] = ay
    return out


@njit(cache=True)
def _rk4_two_body(y0: np.ndarray, t0: float, tf: float, dt: float, mu: float):
    n = int(np.floor((tf - t0) / dt)) + 1
    t = np.empty(n)
    y = np.empty((n, 4))

    yk = y0.copy()
    tk = t0

    for i in range(n):
        t[i] = tk
        y[i, :] = yk

        if i == n - 1:
            break

        k1 = _two_body_rhs_2d(yk, mu)
        k2 = _two_body_rhs_2d(yk + 0.5 * dt * k1, mu)
        k3 = _two_body_rhs_2d(yk + 0.5 * dt * k2, mu)
        k4 = _two_body_rhs_2d(yk + dt * k3, mu)
        yk = yk + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        tk = tk + dt

    return t, y


@njit(cache=True)
def _launch_rhs_controlled_2d(
    t: float,
    y: np.ndarray,
    mu: float,
    r_earth: float,
    r_target: float,
    burn_time: float,
    a_max: float,
    launch_angle: float,
) -> np.ndarray:
    x, yv, vx, vy = y
    r = _norm2(x, yv)

    # Use linear-inside-sphere gravity model for r < r_earth to avoid singularity at r=0.
    if r < r_earth:
        gfac = mu / max(r_earth * r_earth * r_earth, 1.0)
        ax_g = -gfac * x
        ay_g = -gfac * yv
    else:
        r3 = max(r * r * r, 1e3)
        ax_g = -mu * x / r3
        ay_g = -mu * yv / r3

    if r < 1.0:
        erx = np.cos(launch_angle)
        ery = np.sin(launch_angle)
    else:
        erx = x / r
        ery = yv / r
    etx = -ery
    ety = erx

    vr = vx * erx + vy * ery
    vt = vx * etx + vy * ety
    v_circ_target = np.sqrt(mu / max(r_target, 1.0))

    s = min(max(t / max(burn_time, 1.0), 0.0), 1.0)

    # Radial climb + damping of radial speed.
    a_r = 1.8e-4 * (r_target - r) - 0.018 * vr
    # Tangential gravity-turn-like acceleration toward circular target.
    a_t_base = (1.0 - s) * 0.85 * a_max + s * 0.20 * a_max
    a_t = a_t_base + 0.08 * (v_circ_target - vt)

    a_cmd = _norm2(a_r, a_t)
    if a_cmd > a_max:
        scale = a_max / a_cmd
        a_r *= scale
        a_t *= scale

    ax = ax_g + a_r * erx + a_t * etx
    ay = ay_g + a_r * ery + a_t * ety

    out = np.empty(4)
    out[0] = vx
    out[1] = vy
    out[2] = ax
    out[3] = ay
    return out


@njit(cache=True)
def _rk4_launch_controlled(
    y0: np.ndarray,
    t0: float,
    tf: float,
    dt: float,
    mu: float,
    r_earth: float,
    r_target: float,
    burn_time: float,
    a_max: float,
    launch_angle: float,
):
    n = int(np.floor((tf - t0) / dt)) + 1
    t = np.empty(n)
    y = np.empty((n, 4))

    yk = y0.copy()
    tk = t0

    for i in range(n):
        t[i] = tk
        y[i, :] = yk
        if i == n - 1:
            break

        k1 = _launch_rhs_controlled_2d(tk, yk, mu, r_earth, r_target, burn_time, a_max, launch_angle)
        k2 = _launch_rhs_controlled_2d(
            tk + 0.5 * dt, yk + 0.5 * dt * k1, mu, r_earth, r_target, burn_time, a_max, launch_angle
        )
        k3 = _launch_rhs_controlled_2d(
            tk + 0.5 * dt, yk + 0.5 * dt * k2, mu, r_earth, r_target, burn_time, a_max, launch_angle
        )
        k4 = _launch_rhs_controlled_2d(tk + dt, yk + dt * k3, mu, r_earth, r_target, burn_time, a_max, launch_angle)
        yk = yk + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        tk = tk + dt

    return t, y


@njit(cache=True)
def _integrate_cislunar_with_capture(
    y0: np.ndarray,
    t0: float,
    tf: float,
    dt: float,
    mu_earth: float,
    mu_moon: float,
    r_em: float,
    omega_m: float,
    phase0: float,
    r_soi: float,
    enable_loi: int,
    target_circular_factor: float,
    loi_dv_max: float,
    loi_burn_duration_s: float,
):
    n = int(np.floor((tf - t0) / dt)) + 1
    t = np.empty(n)
    y = np.empty((n, 4))
    moon = np.empty((n, 4))
    eps_m = np.empty(n)

    yk = y0.copy()
    tk = t0

    burn_done = 0
    burn_active = 0
    burn_time = -1.0
    burn_dv = 0.0
    burn_steps_left = 0
    burn_dvx_step = 0.0
    burn_dvy_step = 0.0

    prev_rd = 0.0
    prev_rm = 1e30

    for i in range(n):
        mx, my, mvx, mvy = _moon_state_2d(tk, r_em, omega_m, phase0)

        t[i] = tk
        y[i, :] = yk
        moon[i, 0] = mx
        moon[i, 1] = my
        moon[i, 2] = mvx
        moon[i, 3] = mvy

        rxm = yk[0] - mx
        rym = yk[1] - my
        vxm = yk[2] - mvx
        vym = yk[3] - mvy
        rm = _norm2(rxm, rym)
        vm2 = vxm * vxm + vym * vym
        eps_m[i] = 0.5 * vm2 - mu_moon / max(rm, 1.0)

        if i == n - 1:
            break

        # Propagate one RK4 step (Earth + Moon gravity).
        k1 = _cislunar_rhs_2d(tk, yk, mu_earth, mu_moon, r_em, omega_m, phase0)
        k2 = _cislunar_rhs_2d(tk + 0.5 * dt, yk + 0.5 * dt * k1, mu_earth, mu_moon, r_em, omega_m, phase0)
        k3 = _cislunar_rhs_2d(tk + 0.5 * dt, yk + 0.5 * dt * k2, mu_earth, mu_moon, r_em, omega_m, phase0)
        k4 = _cislunar_rhs_2d(tk + dt, yk + dt * k3, mu_earth, mu_moon, r_em, omega_m, phase0)
        y_next = yk + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        t_next = tk + dt
        mx2, my2, mvx2, mvy2 = _moon_state_2d(t_next, r_em, omega_m, phase0)

        rxm2 = y_next[0] - mx2
        rym2 = y_next[1] - my2
        vxm2 = y_next[2] - mvx2
        vym2 = y_next[3] - mvy2
        rm2 = _norm2(rxm2, rym2)

        # Radial speed wrt Moon to detect perilune crossing.
        rd2 = (rxm2 * vxm2 + rym2 * vym2) / max(rm2, 1.0)

        # First local minimum in lunar distance inside SOI: trigger finite-duration LOI burn.
        if enable_loi == 1 and burn_done == 0 and burn_active == 0:
            if rm2 < r_soi and prev_rm < r_soi and prev_rd < 0.0 and rd2 >= 0.0:
                # Build tangential direction in moon-centered frame.
                h = rxm2 * vym2 - rym2 * vxm2
                if h >= 0.0:
                    tx = -rym2 / max(rm2, 1.0)
                    ty = rxm2 / max(rm2, 1.0)
                else:
                    tx = rym2 / max(rm2, 1.0)
                    ty = -rxm2 / max(rm2, 1.0)

                v_target = target_circular_factor * np.sqrt(mu_moon / max(rm2, 1.0))
                vtx = tx * v_target
                vty = ty * v_target

                dvx_rel = vtx - vxm2
                dvy_rel = vty - vym2
                dv_mag = _norm2(dvx_rel, dvy_rel)

                if dv_mag > loi_dv_max and dv_mag > 1e-12:
                    scale = loi_dv_max / dv_mag
                    dvx_rel *= scale
                    dvy_rel *= scale
                    dv_mag = loi_dv_max

                # Finite burn split across several solver steps to avoid epsilon jump.
                n_steps = int(np.round(max(loi_burn_duration_s, dt) / max(dt, 1e-9)))
                if n_steps < 1:
                    n_steps = 1
                burn_steps_left = n_steps
                burn_dvx_step = dvx_rel / n_steps
                burn_dvy_step = dvy_rel / n_steps
                burn_active = 1
                burn_time = t_next
                burn_dv = dv_mag

        if burn_active == 1 and burn_steps_left > 0:
            y_next[2] += burn_dvx_step
            y_next[3] += burn_dvy_step
            burn_steps_left -= 1
            if burn_steps_left <= 0:
                burn_active = 0
                burn_done = 1

        prev_rd = rd2
        prev_rm = rm2

        yk = y_next
        tk = t_next

    return t, y, moon, eps_m, burn_time, burn_dv


def simulate_hohmann_numba(cfg: HohmannConfig | None = None) -> Dict[str, np.ndarray | Dict[str, float]]:
    if cfg is None:
        cfg = HohmannConfig()

    ana = hohmann_analytic(cfg.mu, cfg.r1, cfg.r2)
    y0 = np.array([cfg.r1, 0.0, 0.0, ana["v_peri_t"]], dtype=np.float64)
    t, y = _rk4_two_body(y0, 0.0, ana["tof"], cfg.dt, cfg.mu)

    xf, yf, vxf, vyf = y[-1]
    rf = np.sqrt(xf * xf + yf * yf)
    vf_before = np.sqrt(vxf * vxf + vyf * vyf)

    t_hat = np.array([-yf, xf], dtype=np.float64)
    t_hat /= np.linalg.norm(t_hat)
    vf_after = np.linalg.norm(ana["v_circ_2"] * t_hat)

    return {
        "config": cfg,
        "analytic": ana,
        "time": t,
        "state": y,
        "check": {
            "arrival_radius_error_m": float(rf - cfg.r2),
            "arrival_speed_before_burn2_m_s": float(vf_before),
            "arrival_speed_target_circular_m_s": float(vf_after),
        },
    }


def simulate_cislunar_capture_numba(cfg: CislunarConfig | None = None):
    if cfg is None:
        cfg = CislunarConfig()

    r1 = cfg.r_earth + cfg.leo_altitude
    r2 = cfg.r_earth_moon

    ana = hohmann_analytic(cfg.mu_earth, r1, r2)

    n_m = 2.0 * np.pi / cfg.moon_period
    phase0 = np.pi - n_m * ana["tof"] + np.deg2rad(cfg.moon_phase_bias_deg)

    # TLI as impulsive burn at perigee.
    v0 = ana["v_peri_t"] * cfg.dv1_scale
    y0 = np.array([r1, 0.0, 0.0, v0], dtype=np.float64)

    tf = cfg.t_final_factor * ana["tof"]

    # Hill/SOI approximation of Moon in Earth-centered frame.
    r_soi = cfg.r_earth_moon * (cfg.mu_moon / cfg.mu_earth) ** (2.0 / 5.0)

    t, y, moon, eps_m, burn_time, burn_dv = _integrate_cislunar_with_capture(
        y0=y0,
        t0=0.0,
        tf=tf,
        dt=cfg.dt,
        mu_earth=cfg.mu_earth,
        mu_moon=cfg.mu_moon,
        r_em=cfg.r_earth_moon,
        omega_m=n_m,
        phase0=phase0,
        r_soi=r_soi,
        enable_loi=1 if cfg.enable_loi else 0,
        target_circular_factor=cfg.target_circular_factor,
        loi_dv_max=cfg.loi_dv_max,
        loi_burn_duration_s=cfg.loi_burn_duration_s,
    )

    sc = y[:, :2]
    vel = y[:, 2:4]
    mr = moon[:, :2]
    mv = moon[:, 2:4]

    rel_r = sc - mr
    rel_v = vel - mv

    dist_m = np.linalg.norm(rel_r, axis=1)
    dist_e = np.linalg.norm(sc, axis=1)

    idx_min_m = int(np.argmin(dist_m))
    idx_end = len(t) - 1

    final_eps = float(eps_m[idx_end])
    min_altitude_m = float(np.min(dist_m) - cfg.r_moon)
    collided_moon = bool(min_altitude_m <= 0.0)
    captured_final = bool((final_eps < 0.0) and (dist_m[idx_end] < 1.5 * r_soi) and (not collided_moon))

    summary = {
        "tof_hohmann_days": ana["tof"] / 86400.0,
        "dv1_ideal_m_s": ana["dv1"],
        "dv1_used_m_s": v0 - np.sqrt(cfg.mu_earth / r1),
        "moon_phase0_deg": np.degrees(phase0),
        "moon_soi_km": r_soi / 1e3,
        "closest_moon_distance_km": dist_m[idx_min_m] / 1e3,
        "closest_moon_altitude_km": min_altitude_m / 1e3,
        "closest_moon_time_day": t[idx_min_m] / 86400.0,
        "closest_earth_distance_km": float(np.min(dist_e) / 1e3),
        "collided_moon": collided_moon,
        "loi_burn_time_day": burn_time / 86400.0 if burn_time > 0 else np.nan,
        "loi_burn_dv_m_s": burn_dv if burn_time > 0 else np.nan,
        "final_lunar_specific_energy_km2_s2": final_eps / 1e6,
        "captured_final": captured_final,
    }

    return {
        "config": cfg,
        "analytic_tli": ana,
        "time": t,
        "state": y,
        "moon_state": moon,
        "lunar_specific_energy": eps_m,
        "summary": summary,
    }


def simulate_hohmann_solve_ivp_numba_rhs(cfg: HohmannConfig | None = None):
    if cfg is None:
        cfg = HohmannConfig()

    ana = hohmann_analytic(cfg.mu, cfg.r1, cfg.r2)
    y0 = np.array([cfg.r1, 0.0, 0.0, ana["v_peri_t"]], dtype=np.float64)

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        return _two_body_rhs_2d(y, cfg.mu)

    t_eval = np.arange(0.0, ana["tof"], cfg.dt)
    if t_eval.size == 0 or abs(t_eval[-1] - ana["tof"]) > 1e-9:
        t_eval = np.append(t_eval, ana["tof"])

    sol = solve_ivp(
        rhs,
        (0.0, ana["tof"]),
        y0,
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-11,
        method="RK45",
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    return {
        "config": cfg,
        "analytic": ana,
        "time": sol.t,
        "state": sol.y.T,
    }


def generate_reference_orbits(cfg: HohmannConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    th = np.linspace(0.0, 2.0 * np.pi, 720)
    c = np.cos(th)
    s = np.sin(th)

    circ1 = np.column_stack((cfg.r1 * c, cfg.r1 * s))
    circ2 = np.column_stack((cfg.r2 * c, cfg.r2 * s))

    a_t = 0.5 * (cfg.r1 + cfg.r2)
    e_t = (cfg.r2 - cfg.r1) / (cfg.r2 + cfg.r1)

    nu = np.linspace(0.0, np.pi, 480)
    r = a_t * (1.0 - e_t * e_t) / (1.0 + e_t * np.cos(nu))
    tr = np.column_stack((r * np.cos(nu), r * np.sin(nu)))

    return th, circ1, circ2, tr


def _py_two_body_rhs_2d(state: np.ndarray, mu: float) -> np.ndarray:
    x, y, vx, vy = state
    r = np.hypot(x, y)
    r3 = max(r * r * r, 1e3)
    return np.array([vx, vy, -mu * x / r3, -mu * y / r3], dtype=float)


def _py_propagate_two_body_rk4(state0: np.ndarray, t_rel: np.ndarray, mu: float) -> np.ndarray:
    out = np.zeros((len(t_rel), 4), dtype=float)
    out[0] = state0
    for i in range(len(t_rel) - 1):
        dt = float(t_rel[i + 1] - t_rel[i])
        y = out[i]
        k1 = _py_two_body_rhs_2d(y, mu)
        k2 = _py_two_body_rhs_2d(y + 0.5 * dt * k1, mu)
        k3 = _py_two_body_rhs_2d(y + 0.5 * dt * k2, mu)
        k4 = _py_two_body_rhs_2d(y + dt * k3, mu)
        out[i + 1] = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return out


def _moon_kinematics_from_main(moon_xy_m: np.ndarray, t_main: np.ndarray):
    if len(t_main) < 2:
        ang0 = np.arctan2(moon_xy_m[0, 1], moon_xy_m[0, 0])
        omega = 2.0 * np.pi / (27.321661 * 86400.0)
        r_m = np.hypot(moon_xy_m[0, 0], moon_xy_m[0, 1])
        return r_m, ang0, omega

    dt = max(t_main[1] - t_main[0], 1e-9)
    ang0 = np.arctan2(moon_xy_m[0, 1], moon_xy_m[0, 0])
    ang1 = np.arctan2(moon_xy_m[1, 1], moon_xy_m[1, 0])
    d_ang = np.arctan2(np.sin(ang1 - ang0), np.cos(ang1 - ang0))
    omega = d_ang / dt
    r_m = np.hypot(moon_xy_m[0, 0], moon_xy_m[0, 1])
    return r_m, ang0, omega


def build_animation_data(cfg: CislunarConfig):
    """
    Build animation timeline data (launch + small orbit + Hohmann + big orbit + cislunar coast/capture).
    This function keeps all trajectory/phase calculations inside the solver module.
    """
    mu = cfg.mu_earth
    r_earth = cfg.r_earth
    r_small = r_earth + 80e3
    r_big_demo = 2.0 * r_small

    cfg_main = replace(cfg, leo_altitude=r_big_demo - r_earth)
    out_main = simulate_cislunar_capture_numba(cfg_main)

    t_main = out_main["time"]
    sc_main = out_main["state"][:, :2]
    moon_main = out_main["moon_state"][:, :2]
    eps_main = out_main["lunar_specific_energy"]

    # Segment 0: launch by forced two-body ODE (Earth gravity + thrust control).
    launch_duration = 3400.0
    launch_angle = -0.10
    y0_launch = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    dt_launch = min(cfg.dt, 5.0)
    t0, y0_num = _rk4_launch_controlled(
        y0_launch,
        0.0,
        launch_duration,
        dt_launch,
        mu,
        r_earth,
        r_small,
        launch_duration,
        6.0,
        launch_angle,
    )
    sc0 = y0_num[:, :2]
    n0 = len(t0)
    phase0 = np.array(["Launch & Acceleration"] * n0, dtype=object)

    # Segment 1: smooth transition from launch ellipse-like arc to small circular orbit.
    # Use short Hermite blend (C1 continuity), then pure circular motion.
    xL, yL, vxL, vyL = y0_num[-1]
    rL = max(np.hypot(xL, yL), 1.0)
    th1_start = np.arctan2(yL, xL)
    erx = xL / rL
    ery = yL / rL
    etx = -ery
    ety = erx
    vrL = vxL * erx + vyL * ery
    vtL = vxL * etx + vyL * ety
    omegaL = vtL / rL

    dth_to_pi = np.mod(np.pi - th1_start, 2.0 * np.pi)
    omega_small = np.sqrt(mu / r_small**3)
    dth_total = dth_to_pi + 2.0 * np.pi

    t_trans = min(650.0, 0.32 * dth_total / omega_small)
    dth_trans = omega_small * t_trans
    th_trans_end = th1_start + dth_trans
    dth_rem = max(dth_total - dth_trans, 0.0)
    t_circ = dth_rem / omega_small
    small_orbit_duration = t_trans + t_circ

    dt_small = min(cfg.dt, dt_launch)
    n1 = max(14, int(np.floor(small_orbit_duration / dt_small)) + 1)
    t1 = t0[-1] + np.arange(n1) * dt_small
    t1[-1] = t0[-1] + small_orbit_duration
    tau = t1 - t1[0]
    sc1 = np.zeros((n1, 2), dtype=float)

    if t_trans > 1e-9:
        s = np.clip(tau / t_trans, 0.0, 1.0)
        h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
        h10 = s**3 - 2.0 * s**2 + s
        h01 = -2.0 * s**3 + 3.0 * s**2
        h11 = s**3 - s**2

        r_blend = h00 * rL + h10 * t_trans * vrL + h01 * r_small
        th_blend = h00 * th1_start + h10 * t_trans * omegaL + h01 * th_trans_end + h11 * t_trans * omega_small

        in_blend = tau <= t_trans
        sc1[in_blend, 0] = r_blend[in_blend] * np.cos(th_blend[in_blend])
        sc1[in_blend, 1] = r_blend[in_blend] * np.sin(th_blend[in_blend])
    else:
        in_blend = np.zeros(n1, dtype=bool)

    t_after = tau[~in_blend] - t_trans
    th_after = th_trans_end + omega_small * t_after
    sc1[~in_blend, 0] = r_small * np.cos(th_after)
    sc1[~in_blend, 1] = r_small * np.sin(th_after)
    sc1[0, :] = sc0[-1, :]
    phase1 = np.array(["Small Circular Orbit"] * n1, dtype=object)

    # Segment 2: Hohmann transfer numerical ODE + analytic reference
    ana_pre = hohmann_analytic(mu, r_small, r_big_demo)
    transfer_duration = ana_pre["tof"]
    n2 = max(40, int(np.floor(transfer_duration / cfg.dt)) + 1)
    t2 = t1[-1] + np.arange(n2) * cfg.dt
    t2[-1] = t1[-1] + transfer_duration

    y2_0 = np.array([-r_small, 0.0, 0.0, -ana_pre["v_peri_t"]], dtype=float)
    y2_num = _py_propagate_two_body_rk4(y2_0, t2 - t2[0], mu)
    sc2 = y2_num[:, :2]

    a_t = 0.5 * (r_small + r_big_demo)
    e_t = (r_big_demo - r_small) / (r_big_demo + r_small)
    nu = np.linspace(0.0, np.pi, n2)
    r_tr = a_t * (1.0 - e_t * e_t) / (1.0 + e_t * np.cos(nu))
    th2 = np.pi + nu
    sc2_analytic = np.column_stack((r_tr * np.cos(th2), r_tr * np.sin(th2)))
    phase2 = np.array(["Hohmann Transfer"] * n2, dtype=object)

    # Segment 3: big circular orbit
    omega_big = np.sqrt(mu / r_big_demo**3)
    big_orbit_duration = 2.0 * np.pi / omega_big
    n3 = max(14, int(np.floor(big_orbit_duration / cfg.dt)) + 1)
    t3 = np.linspace(t2[-1], t2[-1] + big_orbit_duration, n3)
    th3 = omega_big * (t3 - t3[0])
    sc3 = np.column_stack((r_big_demo * np.cos(th3), r_big_demo * np.sin(th3)))
    phase3 = np.array(["Big Circular Orbit"] * n3, dtype=object)

    # Segment 4+: cislunar coast/approach/capture
    t4 = t3[-1] + t_main
    phase4 = np.empty(len(t_main), dtype=object)
    loi_day = out_main["summary"]["loi_burn_time_day"]
    for i, ti in enumerate(t_main / 86400.0):
        if np.isnan(loi_day):
            phase4[i] = "Coast" if ti <= out_main["summary"]["tof_hohmann_days"] else "Approach"
        else:
            phase4[i] = "Lunar Capture" if ti >= loi_day else ("Coast" if ti <= out_main["summary"]["tof_hohmann_days"] else "Approach")

    r_m, ang_m0, omega_m = _moon_kinematics_from_main(moon_main, t_main)
    t_ref_main_start = t4[0]

    def moon_xy(tt: np.ndarray):
        ang = ang_m0 + omega_m * (tt - t_ref_main_start)
        return np.column_stack((r_m * np.cos(ang), r_m * np.sin(ang)))

    moon0 = moon_xy(t0)
    moon1 = moon_xy(t1)
    moon2 = moon_xy(t2)
    moon3 = moon_xy(t3)
    moon4 = moon_xy(t4)

    # Skip repeated boundary samples to avoid visual pause at stage transitions.
    t_all = np.concatenate([t0, t1[1:], t2[1:], t3[1:], t4[1:]])
    sc_all = np.vstack([sc0, sc1[1:], sc2[1:], sc3[1:], sc_main[1:]])
    moon_all = np.vstack([moon0, moon1[1:], moon2[1:], moon3[1:], moon4[1:]])
    phase_all = np.concatenate([phase0, phase1[1:], phase2[1:], phase3[1:], phase4[1:]])

    eps_all = np.concatenate([
        np.full(len(t0), np.nan),
        np.full(len(t1) - 1, np.nan),
        np.full(len(t2) - 1, np.nan),
        np.full(len(t3) - 1, np.nan),
        eps_main[1:],
    ])

    # -----------------------------
    # Segment 5: Moon-focused free orbiting (ODE, Earth+Moon gravity)
    # Segment 6: Deorbit burn + ballistic coast (ODE)
    # Segment 7: Powered descent (ODE + control acceleration)
    # Segment 8: Landing complete hold (marker hidden)
    # -----------------------------
    def moon_state(tt: float):
        ang = ang_m0 + omega_m * (tt - t_ref_main_start)
        c = np.cos(ang)
        s = np.sin(ang)
        mx = r_m * c
        my = r_m * s
        mvx = -r_m * omega_m * s
        mvy = r_m * omega_m * c
        return mx, my, mvx, mvy

    def lunar_specific_energy_series(tt: np.ndarray, yy: np.ndarray) -> np.ndarray:
        out = np.empty(len(tt), dtype=float)
        for ii in range(len(tt)):
            mx, my, mvx, mvy = moon_state(float(tt[ii]))
            rx = yy[ii, 0] - mx
            ry = yy[ii, 1] - my
            vx = yy[ii, 2] - mvx
            vy = yy[ii, 3] - mvy
            rr = max(np.hypot(rx, ry), 1.0)
            out[ii] = 0.5 * (vx * vx + vy * vy) - cfg.mu_moon / rr
        return out

    def rhs_free(tt: float, yy: np.ndarray) -> np.ndarray:
        x, yv, vx, vy = yy
        mx, my, _, _ = moon_state(tt)
        re = np.hypot(x, yv)
        re3 = max(re * re * re, 1e3)
        dxm = x - mx
        dym = yv - my
        rmv = np.hypot(dxm, dym)
        rm3 = max(rmv * rmv * rmv, 1e3)
        ax = -cfg.mu_earth * x / re3 - cfg.mu_moon * dxm / rm3
        ay = -cfg.mu_earth * yv / re3 - cfg.mu_moon * dym / rm3
        return np.array([vx, vy, ax, ay], dtype=float)

    # Start strictly from current trajectory end-state (no manual "snap-to-orbit").
    t5_start = float(t_all[-1])
    y5_seed = out_main["state"][-1].astype(float).copy()
    mx0, my0, mvx0, mvy0 = moon_state(t5_start)
    rel_r0 = y5_seed[:2] - np.array([mx0, my0], dtype=float)
    rm0 = max(np.linalg.norm(rel_r0), 1.0)

    orbit_period = 2.0 * np.pi * np.sqrt(rm0**3 / cfg.mu_moon)
    orbit_dur = max(0.0, cfg.lunar_orbit_cycles_before_descent * orbit_period)
    t5_end = t5_start + orbit_dur
    t_eval5 = np.arange(t5_start, t5_end, cfg.dt)
    if t_eval5.size == 0 or abs(t_eval5[-1] - t5_end) > 1e-9:
        t_eval5 = np.append(t_eval5, t5_end)
    sol5 = solve_ivp(
        rhs_free,
        (t5_start, t5_end),
        y5_seed,
        t_eval=t_eval5,
        rtol=1e-8,
        atol=1e-10,
        max_step=max(cfg.dt, 60.0),
    )
    if not sol5.success:
        raise RuntimeError(f"Moon-orbit segment integration failed: {sol5.message}")
    t5 = sol5.t
    y5 = sol5.y.T
    sc5 = y5[:, :2]
    moon5 = moon_xy(t5)
    phase5 = np.array(["Lunar Orbiting"] * len(t5), dtype=object)
    eps5 = lunar_specific_energy_series(t5, y5)

    # Segment 6: finite deorbit burn (ODE) + ballistic coast to low altitude.
    t6_start = float(t5[-1])
    y6_0 = y5[-1].copy()
    mx6, my6, mvx6, mvy6 = moon_state(t6_start)
    rel_r6 = y6_0[:2] - np.array([mx6, my6], dtype=float)
    rel_v6 = y6_0[2:] - np.array([mvx6, mvy6], dtype=float)
    r6 = max(np.linalg.norm(rel_r6), 1.0)
    v6 = np.linalg.norm(rel_v6)
    vhat6 = rel_v6 / max(v6, 1e-9)

    rp_target = max(cfg.r_moon + cfg.deorbit_perilune_altitude, cfg.r_moon + 1e3)
    if rp_target >= r6:
        rp_target = max(cfg.r_moon + 1e3, 0.92 * r6)
    a_deorbit = 0.5 * (r6 + rp_target)
    v_req = np.sqrt(max(cfg.mu_moon * (2.0 / r6 - 1.0 / a_deorbit), 0.0))
    dv_retro = max(0.0, v6 - v_req)
    burn_dur = max(1.0, cfg.deorbit_burn_duration_s)
    a_burn = dv_retro / burn_dur

    def evt_entry_alt(tt: float, yy: np.ndarray) -> float:
        mx, my, _, _ = moon_state(tt)
        return np.hypot(yy[0] - mx, yy[1] - my) - (cfg.r_moon + cfg.deorbit_entry_altitude)

    evt_entry_alt.terminal = True
    evt_entry_alt.direction = -1

    def rhs_deorbit_burn(tt: float, yy: np.ndarray) -> np.ndarray:
        base = rhs_free(tt, yy)
        mx, my, mvx, mvy = moon_state(tt)
        rel_v = np.array([yy[2] - mvx, yy[3] - mvy], dtype=float)
        vmag = np.linalg.norm(rel_v)
        if vmag > 1e-9 and a_burn > 0.0:
            thrust = -a_burn * (rel_v / vmag)
        else:
            thrust = np.zeros(2, dtype=float)
        return np.array([base[0], base[1], base[2] + thrust[0], base[3] + thrust[1]], dtype=float)

    # 6a) finite deorbit burn
    t6_burn_end = t6_start + burn_dur
    t_eval6a = np.arange(t6_start, t6_burn_end, cfg.dt)
    if t_eval6a.size == 0 or abs(t_eval6a[-1] - t6_burn_end) > 1e-9:
        t_eval6a = np.append(t_eval6a, t6_burn_end)
    sol6a = solve_ivp(
        rhs_deorbit_burn,
        (t6_start, t6_burn_end),
        y6_0,
        t_eval=t_eval6a,
        rtol=1e-8,
        atol=1e-10,
        max_step=max(cfg.dt, 30.0),
    )
    if not sol6a.success:
        raise RuntimeError(f"Deorbit burn integration failed: {sol6a.message}")

    # 6b) ballistic coast to entry altitude
    t6_end = t6_start + max(0.25 * 86400.0, cfg.descent_max_days * 86400.0)
    y6b_0 = sol6a.y[:, -1].copy()
    t_eval6 = np.arange(t6_burn_end, t6_end, cfg.dt)
    if t_eval6.size == 0 or abs(t_eval6[-1] - t6_end) > 1e-9:
        t_eval6 = np.append(t_eval6, t6_end)
    sol6 = solve_ivp(
        rhs_free,
        (t6_burn_end, t6_end),
        y6b_0,
        t_eval=t_eval6,
        events=evt_entry_alt,
        dense_output=True,
        rtol=1e-8,
        atol=1e-10,
        max_step=max(cfg.dt, 60.0),
    )
    if not sol6.success:
        raise RuntimeError(f"Deorbit coast integration failed: {sol6.message}")
    if sol6.t_events[0].size > 0:
        t_entry = float(sol6.t_events[0][0])
        y_entry = sol6.sol(t_entry)
        if abs(sol6.t[-1] - t_entry) > 1e-6:
            t6b = np.append(sol6.t, t_entry)
            y6b = np.vstack([sol6.y.T, y_entry])
        else:
            t6b = sol6.t
            y6b = sol6.y.T
    else:
        t_entry = float(sol6.t[-1])
        y_entry = sol6.y[:, -1]
        t6b = sol6.t
        y6b = sol6.y.T

    t6 = np.concatenate([sol6a.t, t6b[1:]])
    y6 = np.vstack([sol6a.y.T, y6b[1:]])

    sc6 = y6[:, :2]
    moon6 = moon_xy(t6)
    phase6 = np.array(["Deorbit & Coast"] * len(t6), dtype=object)
    eps6 = lunar_specific_energy_series(t6, y6)

    # Powered descent with controlled acceleration in moon-centered frame.
    def rhs_descent(tt: float, yy: np.ndarray) -> np.ndarray:
        base = rhs_free(tt, yy)
        mx, my, mvx, mvy = moon_state(tt)
        rel_r = np.array([yy[0] - mx, yy[1] - my], dtype=float)
        rel_v = np.array([yy[2] - mvx, yy[3] - mvy], dtype=float)
        rho = max(np.linalg.norm(rel_r), 1.0)
        er = rel_r / rho
        h = max(rho - cfg.r_moon, 0.0)
        et = np.array([-er[1], er[0]], dtype=float)
        vr = float(np.dot(rel_v, er))
        vt = float(np.dot(rel_v, et))

        # Guidance law:
        # - push inward and reduce tangential speed at high altitude,
        # - then heavily damp both radial/tangential velocity near surface.
        if h > 60e3:
            v_t_des = max(80.0, 0.85 * np.sqrt(cfg.mu_moon / rho))
            a_r_cmd = -0.0045 * (h / 1e3) - 0.035 * vr
            a_t_cmd = -0.12 * (vt - v_t_des)
        elif h > 20e3:
            v_t_des = max(35.0, 0.55 * np.sqrt(cfg.mu_moon / rho))
            a_r_cmd = -0.0085 * (h / 1e3) - 0.06 * vr
            a_t_cmd = -0.16 * (vt - v_t_des)
        else:
            v_t_des = 0.0
            a_r_cmd = -0.040 * (h / 1e3) - 0.28 * vr
            a_t_cmd = -0.30 * (vt - v_t_des)
        a_ctrl = a_r_cmd * er + a_t_cmd * et
        a_mag = np.linalg.norm(a_ctrl)
        if a_mag > cfg.landing_a_max and a_mag > 1e-12:
            a_ctrl *= cfg.landing_a_max / a_mag

        return np.array([base[0], base[1], base[2] + a_ctrl[0], base[3] + a_ctrl[1]], dtype=float)

    def evt_landing(tt: float, yy: np.ndarray) -> float:
        mx, my, _, _ = moon_state(tt)
        return np.hypot(yy[0] - mx, yy[1] - my) - cfg.r_moon

    evt_landing.terminal = True
    evt_landing.direction = -1

    t7_start = float(t6[-1])
    t7_end = t7_start + max(0.0, cfg.descent_max_days * 86400.0)
    y7_0 = y6[-1].copy()
    t_eval7 = np.arange(t7_start, t7_end, cfg.dt)
    if t_eval7.size == 0 or abs(t_eval7[-1] - t7_end) > 1e-9:
        t_eval7 = np.append(t_eval7, t7_end)
    sol7 = solve_ivp(
        rhs_descent,
        (t7_start, t7_end),
        y7_0,
        t_eval=t_eval7,
        events=evt_landing,
        dense_output=True,
        rtol=1e-8,
        atol=1e-10,
        max_step=max(cfg.dt, 30.0),
    )
    if not sol7.success:
        raise RuntimeError(f"Powered descent integration failed: {sol7.message}")

    landed = sol7.t_events[0].size > 0
    if landed:
        t_land = float(sol7.t_events[0][0])
        y_land = sol7.sol(t_land)
        if abs(sol7.t[-1] - t_land) > 1e-6:
            t7 = np.append(sol7.t, t_land)
            y7 = np.vstack([sol7.y.T, y_land])
        else:
            t7 = sol7.t
            y7 = sol7.y.T
    else:
        t_land = float(sol7.t[-1])
        y_land = sol7.y[:, -1]
        t7 = sol7.t
        y7 = sol7.y.T
    sc7 = y7[:, :2]
    moon7 = moon_xy(t7)
    phase7 = np.array(["Powered Descent"] * len(t7), dtype=object)
    eps7 = lunar_specific_energy_series(t7, y7)

    # Landing complete hold: spacecraft considered landed and no longer displayed.
    hold_s = max(0.0, cfg.post_landing_hold_days * 86400.0)
    n8 = max(2, int(np.floor(hold_s / cfg.dt)) + 1)
    t8 = t_land + np.arange(n8) * cfg.dt
    t8[-1] = t_land + hold_s
    moon8 = moon_xy(t8)
    mx_land, my_land, _, _ = moon_state(t_land)
    rel_land = np.array([y_land[0] - mx_land, y_land[1] - my_land], dtype=float)
    rel_land_n = np.linalg.norm(rel_land)
    if rel_land_n < 1.0:
        rel_land = np.array([cfg.r_moon, 0.0], dtype=float)
    else:
        rel_land = rel_land * (cfg.r_moon / rel_land_n)
    sc8 = moon8 + rel_land[None, :]
    phase8 = np.array(["Landing Complete"] * n8, dtype=object)
    eps8 = np.full(n8, -cfg.mu_moon / max(cfg.r_moon, 1.0), dtype=float)

    t_all = np.concatenate([t_all, t5[1:], t6[1:], t7[1:], t8[1:]])
    sc_all = np.vstack([sc_all, sc5[1:], sc6[1:], sc7[1:], sc8[1:]])
    moon_all = np.vstack([moon_all, moon5[1:], moon6[1:], moon7[1:], moon8[1:]])
    phase_all = np.concatenate([phase_all, phase5[1:], phase6[1:], phase7[1:], phase8[1:]])
    eps_all = np.concatenate([eps_all, eps5[1:], eps6[1:], eps7[1:], eps8[1:]])

    sc_visible = phase_all != "Landing Complete"
    # Hide marker exactly at and after landing event.
    sc_visible[t_all >= t_land] = False

    summary = out_main["summary"]
    summary["lunar_orbit_cycles_before_descent"] = float(cfg.lunar_orbit_cycles_before_descent)
    summary["deorbit_dv_m_s"] = float(dv_retro)
    summary["deorbit_burn_duration_s"] = float(burn_dur)
    summary["powered_descent_max_days"] = float(cfg.descent_max_days)
    summary["landing_completed"] = bool(landed)
    summary["landing_time_day"] = float(t_land / 86400.0)
    summary["post_landing_hold_days"] = float(cfg.post_landing_hold_days)
    return {
        "t": t_all,
        "sc_xy_km": sc_all / 1e3,
        "moon_xy_km": moon_all / 1e3,
        "hohmann_analytic_km": sc2_analytic / 1e3,
        "eps_m": eps_all,
        "phase": phase_all,
        "sc_visible": sc_visible,
        "summary": summary,
        "r_soi_km": summary["moon_soi_km"],
        "r_earth_km": cfg.r_earth / 1e3,
        "r_moon_km": cfg.r_moon / 1e3,
    }


if __name__ == "__main__":
    out = simulate_cislunar_capture_numba(CislunarConfig())
    print("Cislunar transfer with lunar gravity (numba RK4)")
    for k, v in out["summary"].items():
        if isinstance(v, (bool, np.bool_)):
            print(f"{k:35s}: {bool(v)}")
        elif np.isnan(v):
            print(f"{k:35s}: nan")
        else:
            print(f"{k:35s}: {v:14.6f}")
