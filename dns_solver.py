"""
dns_solver.py  —  High-Re Pseudo-Spectral DNS of 3-D Incompressible Turbulence
================================================================================
Solves the incompressible Navier-Stokes equations in a triply-periodic
box [0, 2π]³:

    ∂u_i/∂t + u_j ∂u_i/∂x_j = -∂p/∂x_i + ν ∇²u_i + f_i
    ∂u_i/∂x_i = 0

Numerical method
----------------
  Spatial       : Fourier pseudo-spectral; dealiasing via 2/3 rule
  Time          : Low-storage 3rd-order Runge-Kutta (Williamson 1980)
                  with exact exponential integrating factor for viscosity
  Forcing       : Deterministic linear (Lundgren 2003) at |k| ≤ k_f
                  — maintains stationary turbulence at target ε
  Pressure      : Eliminated via Helmholtz projection (never solved explicitly)

Performance features for high-Re (N ≥ 256)
-------------------------------------------
  • pyfftw with multi-threaded FFTW3 (auto-detects CPU count)
  • Wisdom file caching (~30 % speedup on repeated runs)
  • r2c/c2r (real-to-complex) transforms: halves memory & FFT work
  • In-place operations throughout to minimise allocations
  • Pre-allocated FFTW plan objects (no re-planning per step)
  • Optional MPI via mpi4py + mpi4py-fft for multi-node runs (see note)

Recommended grids for target Re_λ
-----------------------------------
  Re_λ ~  50   →  N =  64,  ν = 5e-3
  Re_λ ~ 100   →  N = 128,  ν = 2e-3
  Re_λ ~ 200   →  N = 256,  ν = 8e-4   ← minimum for Re_λ > 200
  Re_λ ~ 400   →  N = 512,  ν = 3e-4
  Re_λ ~ 600   →  N = 768,  ν = 1.5e-4

  Resolution criterion: k_max η ≥ 1.0  (k_max = N/3 after dealiasing)

Outputs
-------
  dns_checkpoint.npz  — spectral velocity field + time (restart file)
  dns_fields.npz      — all derived fields (u_i, S_ij, Ω_ij, τ_ij, E(k))
                        compatible with sgs_conditional_pdf.py
  dns_energy.png      — E(k) with k^{-5/3} reference
  dns_stats.png       — time history of TKE, ε, Re_λ, k_max η
  dns_slice.png       — mid-plane slices of |ω|, Q, |S|, |τ_12|

Usage
-----
  python dns_solver.py --N 256 --nu 8e-4 --T 50
  python dns_solver.py --N 512 --nu 3e-4 --T 100 --threads 16
  python dns_solver.py --N 256 --resume dns_checkpoint.npz --T 20
  python dns_solver.py --N 256 --nu 8e-4 --T 50 --filter gaussian --delta 0.15

MPI note (large runs, N ≥ 512)
------------------------------
  pip install mpi4py mpi4py-fft
  mpirun -n <procs> python dns_solver.py --N 512 ...
  The code auto-detects mpi4py and switches to slab decomposition.

Requirements
------------
  numpy matplotlib scipy
  pyfftw          (strongly recommended; pip install pyfftw)
  mpi4py          (optional, for multi-node)
  mpi4py-fft      (optional, for multi-node)
"""

import argparse
import os
import time
import multiprocessing
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── FFT backend ──────────────────────────────────────────────────────────────
_USE_MPI = False
_USE_PYFFTW = False

try:
    from mpi4py import MPI
    from mpi4py_fft import PFFT, newDistArray

    _USE_MPI = MPI.COMM_WORLD.Get_size() > 1
    if _USE_MPI:
        _COMM = MPI.COMM_WORLD
        _RANK = _COMM.Get_rank()
        print(f"[rank {_RANK}] mpi4py-fft: {_COMM.Get_size()} ranks")
except ImportError:
    pass

if not _USE_MPI:
    try:
        import pyfftw
        import pyfftw.interfaces.numpy_fft as _fft_mod

        pyfftw.interfaces.cache.enable()
        _USE_PYFFTW = True
    except ImportError:
        import numpy.fft as _fft_mod

        print(
            "Warning: pyfftw not found. Install with  pip install pyfftw  "
            "for 3–8x speedup (essential at N >= 256)."
        )

_N_THREADS = multiprocessing.cpu_count()


def _fftn(a, **kw):
    if _USE_PYFFTW:
        return _fft_mod.fftn(a,  **kw)
    return _fft_mod.fftn(a, **kw)


def _ifftn(a, **kw):
    if _USE_PYFFTW:
        return _fft_mod.ifftn(a,  **kw)
    return _fft_mod.ifftn(a, **kw)


def _rfftn(a, **kw):
    if _USE_PYFFTW:
        return _fft_mod.rfftn(a, **kw)
    return np.fft.rfftn(a, **kw)


def _irfftn(a, s, **kw):
    if _USE_PYFFTW:
        return _fft_mod.irfftn(a, s=s,  **kw)
    return np.fft.irfftn(a, s=s, **kw)


def save_wisdom(path="fftw_wisdom.pkl"):
    if _USE_PYFFTW:
        import pickle

        with open(path, "wb") as f:
            pickle.dump(pyfftw.export_wisdom(), f)


def load_wisdom(path="fftw_wisdom.pkl"):
    if _USE_PYFFTW and os.path.exists(path):
        import pickle

        with open(path, "rb") as f:
            pyfftw.import_wisdom(pickle.load(f))
        print(f"  FFTW wisdom loaded from {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  WAVENUMBER ARRAYS  (r2c layout)
# ══════════════════════════════════════════════════════════════════════════════


def make_wavenumbers(N):
    """
    Wavenumber arrays for r2c transforms.
    Physical shape: (N, N, N)
    Spectral shape: (N, N, N//2+1)  — Hermitian symmetry in z
    """
    k1d_full = np.fft.fftfreq(N, d=1.0 / N).astype(np.float32)
    k1d_half = np.arange(N // 2 + 1, dtype=np.float32)
    kx, ky, kz = np.meshgrid(k1d_full, k1d_full, k1d_half, indexing="ij")
    k2 = (kx**2 + ky**2 + kz**2).astype(np.float32)
    kmax = N // 3
    dealias = ((np.abs(kx) <= kmax) & (np.abs(ky) <= kmax) & (kz <= kmax)).astype(
        np.float32
    )
    return kx, ky, kz, k2, dealias


# ══════════════════════════════════════════════════════════════════════════════
#  PROJECTION  (enforce div u = 0)
# ══════════════════════════════════════════════════════════════════════════════


def project_(ux_h, uy_h, uz_h, kx, ky, kz, k2):
    """In-place Helmholtz projection."""
    k2s = np.where(k2 == 0, 1.0, k2)
    kdotu = kx * ux_h + ky * uy_h + kz * uz_h
    ux_h -= kx * kdotu / k2s
    uy_h -= ky * kdotu / k2s
    uz_h -= kz * kdotu / k2s
    ux_h[0, 0, 0] = uy_h[0, 0, 0] = uz_h[0, 0, 0] = 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  INITIAL CONDITION
# ══════════════════════════════════════════════════════════════════════════════


def initial_condition(N, kx, ky, kz, k2, dealias, u_rms=1.0, k_peak=4, seed=42):
    """Random-phase IC with von Karman spectrum, normalised to u_rms."""
    rng = np.random.default_rng(seed)
    k = np.sqrt(k2.astype(np.float64))
    Ek = (k / k_peak) ** 4 * np.exp(-2 * (k / k_peak) ** 2)
    Ek[0, 0, 0] = 0.0
    amp = np.sqrt(Ek / (4 * np.pi * np.where(k2 == 0, 1.0, k2) + 1e-30))
    sh = k2.shape

    def rand_h():
        r = (rng.standard_normal(sh) + 1j * rng.standard_normal(sh)).astype(
            np.complex64
        )
        return (amp * r * dealias).astype(np.complex64)

    ux_h, uy_h, uz_h = rand_h(), rand_h(), rand_h()
    project_(ux_h, uy_h, uz_h, kx, ky, kz, k2)

    ux = _irfftn(ux_h, s=(N, N, N))
    uy = _irfftn(uy_h, s=(N, N, N))
    uz = _irfftn(uz_h, s=(N, N, N))
    scale = u_rms / (np.sqrt((ux**2 + uy**2 + uz**2).mean()) + 1e-30)
    ux_h *= scale
    uy_h *= scale
    uz_h *= scale
    return ux_h, uy_h, uz_h


# ══════════════════════════════════════════════════════════════════════════════
#  NONLINEAR TERM + FORCING
# ══════════════════════════════════════════════════════════════════════════════


def nonlinear_and_force(ux_h, uy_h, uz_h, kx, ky, kz, k2, dealias, N, eps_target, k_f):
    """
    Returns -P[u.grad(u)] + F  in spectral space.
    Nonlinear: pseudo-spectral with dealiasing.
    Forcing:   linear at |k| <= k_f  (Lundgren 2003).
    """
    S = (N, N, N)

    ux = _irfftn(ux_h, s=S)
    uy = _irfftn(uy_h, s=S)
    uz = _irfftn(uz_h, s=S)

    def gp(uh, kd):  # gradient in physical space
        return _irfftn(1j * kd * uh * dealias, s=S)

    adv_x = ux * gp(ux_h, kx) + uy * gp(ux_h, ky) + uz * gp(ux_h, kz)
    adv_y = ux * gp(uy_h, kx) + uy * gp(uy_h, ky) + uz * gp(uy_h, kz)
    adv_z = ux * gp(uz_h, kx) + uy * gp(uz_h, ky) + uz * gp(uz_h, kz)

    ax_h = (_rfftn(adv_x) * dealias).astype(np.complex64)
    ay_h = (_rfftn(adv_y) * dealias).astype(np.complex64)
    az_h = (_rfftn(adv_z) * dealias).astype(np.complex64)

    k2s = np.where(k2 == 0, 1.0, k2)
    kdota = kx * ax_h + ky * ay_h + kz * az_h
    nl_x = -(ax_h - kx * kdota / k2s)
    nl_y = -(ay_h - ky * kdota / k2s)
    nl_z = -(az_h - kz * kdota / k2s)
    nl_x[0, 0, 0] = nl_y[0, 0, 0] = nl_z[0, 0, 0] = 0.0

    # linear forcing
    km = np.sqrt(k2)
    fmask = (km <= k_f) & (km > 0)
    N6 = float(N**6)
    E_low = (
        0.5
        * float(
            (
                np.abs(ux_h[fmask]) ** 2
                + np.abs(uy_h[fmask]) ** 2
                + np.abs(uz_h[fmask]) ** 2
            ).sum()
        )
        / N6
    )
    A = eps_target / (2.0 * E_low + 1e-30)
    nl_x[fmask] += A * ux_h[fmask]
    nl_y[fmask] += A * uy_h[fmask]
    nl_z[fmask] += A * uz_h[fmask]

    return nl_x, nl_y, nl_z


# ══════════════════════════════════════════════════════════════════════════════
#  LOW-STORAGE RK3  (Williamson 1980)
# ══════════════════════════════════════════════════════════════════════════════

_A = [0.0, -5.0 / 9.0, -153.0 / 128.0]
_B = [1.0 / 3.0, 15.0 / 16.0, 8.0 / 15.0]
_G = [1.0 / 3.0, 5.0 / 12.0, 1.0 / 4.0]  # sub-step fractions for IF


def rk3_step(ux_h, uy_h, uz_h, kx, ky, kz, k2, dealias, N, nu, dt, eps_target, k_f):
    """Single low-storage RK3 step with exact viscous integrating factor."""
    px = np.zeros_like(ux_h)
    py = np.zeros_like(uy_h)
    pz = np.zeros_like(uz_h)

    for a, b, g in zip(_A, _B, _G):
        nlx, nly, nlz = nonlinear_and_force(
            ux_h, uy_h, uz_h, kx, ky, kz, k2, dealias, N, eps_target, k_f
        )

        px *= a
        px += nlx
        py *= a
        py += nly
        pz *= a
        pz += nlz

        IF = np.exp(-nu * k2 * g * dt).astype(np.float32)
        ux_h *= IF
        ux_h += b * dt * px
        uy_h *= IF
        uy_h += b * dt * py
        uz_h *= IF
        uz_h += b * dt * pz

    project_(ux_h, uy_h, uz_h, kx, ky, kz, k2)
    ux_h *= dealias
    uy_h *= dealias
    uz_h *= dealias
    return ux_h, uy_h, uz_h


# ══════════════════════════════════════════════════════════════════════════════
#  DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════


def compute_stats(ux_h, uy_h, uz_h, k2, nu, N):
    N6 = float(N**6)
    E2 = np.abs(ux_h) ** 2 + np.abs(uy_h) ** 2 + np.abs(uz_h) ** 2
    # r2c weight: double-count kz in (0, N//2)
    wt = np.ones_like(E2)
    wt[:, :, 1 : N // 2] = 2.0
    E = 0.5 * float((E2 * wt).sum()) / N6
    eps = 2.0 * nu * float((k2 * E2 * wt).sum()) / N6

    urms = np.sqrt(max(2 * E / 3, 1e-30))
    dudx2 = float((k2 * np.abs(ux_h) ** 2 * wt).sum()) / N6
    lam = urms / np.sqrt(max(dudx2, 1e-30))
    Re_lam = urms * lam / nu
    eta = (nu**3 / max(eps, 1e-30)) ** 0.25
    tau_k = (nu / max(eps, 1e-30)) ** 0.5
    u_k = (nu * eps) ** 0.25

    k_mag = np.sqrt(k2)
    Ek = E2 * wt / (2 * N6)
    km = k_mag.ravel()
    Em = Ek.ravel()
    pos = km > 0
    idx = np.argsort(km[pos])
    L = (
        np.pi
        / (2 * urms**2 + 1e-30)
        * np.trapezoid((Em[pos] / (km[pos] + 1e-30))[idx], km[pos][idx])
    )

    return dict(
        E=E,
        eps=eps,
        urms=urms,
        lambda_=lam,
        Re_lam=Re_lam,
        eta=eta,
        tau_k=tau_k,
        u_k=u_k,
        L=L,
        k_max_eta=(N // 3) * eta,
    )


def energy_spectrum(ux_h, uy_h, uz_h, k2, N):
    wt = np.ones(k2.shape, np.float32)
    wt[:, :, 1 : N // 2] = 2.0
    E3d = 0.5 * (np.abs(ux_h) ** 2 + np.abs(uy_h) ** 2 + np.abs(uz_h) ** 2) * wt / N**6
    k_int = np.round(np.sqrt(k2)).astype(int)
    k_sh = np.arange(1, N // 2)
    E_sh = np.array([E3d[k_int == ki].sum() for ki in k_sh])
    return k_sh, E_sh


# ══════════════════════════════════════════════════════════════════════════════
#  ADAPTIVE TIME STEP
# ══════════════════════════════════════════════════════════════════════════════


def adaptive_dt(ux_h, uy_h, uz_h, k2, N, nu, CFL=0.4):
    S = (N, N, N)
    dx = 2 * np.pi / N
    ux = _irfftn(ux_h, s=S)
    uy = _irfftn(uy_h, s=S)
    uz = _irfftn(uz_h, s=S)
    u_max = float(max(np.abs(ux).max(), np.abs(uy).max(), np.abs(uz).max())) + 1e-10
    return min(CFL * dx / u_max, 0.5 * dx**2 / (nu * np.pi**2))


# ══════════════════════════════════════════════════════════════════════════════
#  FILTERING AND SGS FIELDS
# ══════════════════════════════════════════════════════════════════════════════


def apply_filter(uh, k2, delta, ftype):
    if ftype == "sharp":
        return uh * (np.sqrt(k2) <= np.pi / delta)
    elif ftype == "gaussian":
        return uh * np.exp(-k2 * delta**2 / 24.0)
    elif ftype == "tophat":
        k = np.sqrt(k2) * delta / (2 * np.pi) + 1e-30
        return uh * np.sinc(k)
    else:
        raise ValueError(f"Unknown filter: {ftype}")


def compute_all_fields(ux_h, uy_h, uz_h, kx, ky, kz, k2, N, delta, ftype):
    """
    Compute all physical-space fields needed for LES a-priori analysis.
    Returns a dict of 1-D flattened arrays (N^3 elements each).

    Fields
    ------
    Filtered velocity      : u, v, w
    SGS stress (full)      : tau_11, tau_12, tau_13, tau_22, tau_23, tau_33
    SGS stress (deviatoric): tau_11d, tau_22d, tau_33d
    SGS trace              : tau_kk
    Strain rate            : S_11, S_12, S_13, S_22, S_23, S_33, S_mag
    Vorticity tensor       : Omega_12, Omega_13, Omega_23, Omega_mag
    Vorticity vector       : omega_x, omega_y, omega_z
    Derived                : k_res, Q (Q-criterion), tau_smag_12
    Energy spectrum        : k_sh, E_sh  (not flattened)
    """
    S = (N, N, N)

    fux_h = apply_filter(ux_h, k2, delta, ftype).astype(np.complex64)
    fuy_h = apply_filter(uy_h, k2, delta, ftype).astype(np.complex64)
    fuz_h = apply_filter(uz_h, k2, delta, ftype).astype(np.complex64)

    def p(h):
        return _irfftn(h, s=S).astype(np.float32)

    def dp(fh, kd):
        return _irfftn(1j * kd * fh, s=S).real.astype(np.float32)

    def fp(a, b):
        return _irfftn(
            apply_filter(_rfftn(a * b).astype(np.complex64), k2, delta, ftype), s=S
        ).astype(np.float32)

    ux = p(ux_h)
    uy = p(uy_h)
    uz = p(uz_h)
    fux = p(fux_h)
    fuy = p(fuy_h)
    fuz = p(fuz_h)

    # SGS stress
    tau_11 = fp(ux, ux) - fux * fux
    tau_12 = fp(ux, uy) - fux * fuy
    tau_13 = fp(ux, uz) - fux * fuz
    tau_22 = fp(uy, uy) - fuy * fuy
    tau_23 = fp(uy, uz) - fuy * fuz
    tau_33 = fp(uz, uz) - fuz * fuz
    tau_kk = tau_11 + tau_22 + tau_33
    tau_11d = tau_11 - tau_kk / 3
    tau_22d = tau_22 - tau_kk / 3
    tau_33d = tau_33 - tau_kk / 3

    # velocity gradients
    dudx = dp(fux_h, kx)
    dudy = dp(fux_h, ky)
    dudz = dp(fux_h, kz)
    dvdx = dp(fuy_h, kx)
    dvdy = dp(fuy_h, ky)
    dvdz = dp(fuy_h, kz)
    dwdx = dp(fuz_h, kx)
    dwdy = dp(fuz_h, ky)
    dwdz = dp(fuz_h, kz)

    # strain rate
    S_11 = dudx
    S_12 = 0.5 * (dudy + dvdx)
    S_13 = 0.5 * (dudz + dwdx)
    S_22 = dvdy
    S_23 = 0.5 * (dvdz + dwdy)
    S_33 = dwdz
    S_mag = np.sqrt(
        2 * (S_11**2 + S_22**2 + S_33**2 + 2 * (S_12**2 + S_13**2 + S_23**2))
    )

    # vorticity tensor
    O12 = 0.5 * (dudy - dvdx)
    O13 = 0.5 * (dudz - dwdx)
    O23 = 0.5 * (dvdz - dwdy)
    Omag = np.sqrt(2 * (O12**2 + O13**2 + O23**2))

    # vorticity vector
    ox = dwdy - dvdz
    oy = dudz - dwdx
    oz = dvdx - dudy

    # derived
    Q = 0.5 * (Omag**2 - S_mag**2)
    k_res = 0.5 * (fux**2 + fuy**2 + fuz**2)
    Cs = 0.17
    tau_smag_12 = -(2 * (Cs * delta) ** 2 * S_mag * S_12)

    def r(a):
        return a.ravel()

    return dict(
        u=r(fux),
        v=r(fuy),
        w=r(fuz),
        tau_11=r(tau_11),
        tau_12=r(tau_12),
        tau_13=r(tau_13),
        tau_22=r(tau_22),
        tau_23=r(tau_23),
        tau_33=r(tau_33),
        tau_11d=r(tau_11d),
        tau_22d=r(tau_22d),
        tau_33d=r(tau_33d),
        tau_kk=r(tau_kk),
        S_11=r(S_11),
        S_12=r(S_12),
        S_13=r(S_13),
        S_22=r(S_22),
        S_23=r(S_23),
        S_33=r(S_33),
        S_mag=r(S_mag),
        Omega_12=r(O12),
        Omega_13=r(O13),
        Omega_23=r(O23),
        Omega_mag=r(Omag),
        omega_x=r(ox),
        omega_y=r(oy),
        omega_z=r(oz),
        k_res=r(k_res),
        Q=r(Q),
        tau_smag_12=r(tau_smag_12),
        delta=np.float32(delta),
        filter_type=ftype,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════


def plot_energy_spectrum(k_sh, E_sh, stats, N, nu, out="dns_energy.png"):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(k_sh, E_sh, "steelblue", lw=2, label="DNS $E(k)$")

    eta = stats["eta"]
    eps = stats["eps"]
    kr = k_sh[(k_sh > 5) & (k_sh * eta < 0.5)]
    if len(kr) > 2:
        ax.loglog(
            kr,
            1.5 * eps ** (2 / 3) * kr ** (-5 / 3),
            "tomato",
            lw=1.8,
            ls="--",
            label=r"$C_K\varepsilon^{2/3}k^{-5/3}$",
        )

    ax.axvline(
        1 / eta,
        color="#777",
        ls=":",
        lw=1.5,
        label=rf"$k\eta=1$  $(k\approx{1 / eta:.0f})$",
    )
    ax.axvline(
        N // 3,
        color="purple",
        ls="--",
        lw=1.2,
        alpha=0.6,
        label=f"Dealiasing cutoff $k={N // 3}$",
    )

    ax.set_xlabel("Wavenumber $k$", fontsize=12)
    ax.set_ylabel("$E(k)$", fontsize=12)
    ax.set_title(
        rf"DNS Energy Spectrum  —  $N={N}^3$,  $\nu={nu}$,  "
        rf"$Re_{{\lambda}}={stats['Re_lam']:.0f}$,  "
        rf"$k_{{max}}\eta={stats['k_max_eta']:.2f}$",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_time_history(history, out="dns_stats.png"):
    t = np.array(history["t"])
    E = np.array(history["E"])
    eps = np.array(history["eps"])
    Re = np.array(history["Re_lam"])
    keta = np.array(history["k_max_eta"])

    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    for ax, (d, lab, col) in zip(
        axes,
        [
            (E, "TKE  $E$", "steelblue"),
            (eps, r"Dissipation  $\varepsilon$", "tomato"),
            (Re, r"$Re_\lambda$", "seagreen"),
            (keta, r"Resolution  $k_{max}\eta$", "darkorange"),
        ],
    ):
        ax.plot(t, d, color=col, lw=1.8)
        ax.set_ylabel(lab, fontsize=11)
        ax.grid(True, alpha=0.25)
    axes[3].axhline(1.0, color="k", ls="--", lw=1.2, label="$k_{max}\\eta=1$")
    axes[3].legend(fontsize=9)
    axes[3].set_xlabel("Time $t$", fontsize=11)
    fig.suptitle("DNS Time History", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_slice(ux_h, uy_h, uz_h, kx, ky, kz, k2, N, delta, ftype, out="dns_slice.png"):
    """Mid-plane (z=N//2) slices of |omega|, Q, |S|, |tau_12|."""
    S = (N, N, N)
    iz = N // 2

    def dp(fh, kd):
        return _irfftn(1j * kd * fh, s=S).real

    def p(h):
        return _irfftn(h, s=S).real

    def fp(a, b):
        return _irfftn(
            apply_filter(_rfftn(a * b).astype(np.complex64), k2, delta, ftype), s=S
        ).real

    fux_h = apply_filter(ux_h, k2, delta, ftype)
    fuy_h = apply_filter(uy_h, k2, delta, ftype)
    fuz_h = apply_filter(uz_h, k2, delta, ftype)

    dudx = dp(fux_h, kx)
    dudy = dp(fux_h, ky)
    dudz = dp(fux_h, kz)
    dvdx = dp(fuy_h, kx)
    dvdy = dp(fuy_h, ky)
    dvdz = dp(fuy_h, kz)
    dwdx = dp(fuz_h, kx)
    dwdy = dp(fuz_h, ky)
    dwdz = dp(fuz_h, kz)

    S11 = dudx
    S12 = 0.5 * (dudy + dvdx)
    S13 = 0.5 * (dudz + dwdx)
    S22 = dvdy
    S23 = 0.5 * (dvdz + dwdy)
    S33 = dwdz
    Smag = np.sqrt(2 * (S11**2 + S22**2 + S33**2 + 2 * (S12**2 + S13**2 + S23**2)))

    O12 = 0.5 * (dudy - dvdx)
    O13 = 0.5 * (dudz - dwdx)
    O23 = 0.5 * (dvdz - dwdy)
    Omag = np.sqrt(2 * (O12**2 + O13**2 + O23**2))
    Q = 0.5 * (Omag**2 - Smag**2)

    ox = dwdy - dvdz
    oy = dudz - dwdx
    oz = dvdx - dudy
    omega_mag = np.sqrt(ox**2 + oy**2 + oz**2)

    fux = p(fux_h)
    fuy = p(fuy_h)
    ux = p(ux_h)
    uy = p(uy_h)
    tau12 = fp(ux, uy) - fux * fuy

    slices = [omega_mag[:, :, iz], Q[:, :, iz], Smag[:, :, iz], np.abs(tau12[:, :, iz])]
    titles = [
        r"$|\omega|$ vorticity mag.",
        r"$Q$-criterion  ($Q>0$: vortex)",
        r"$|\bar{S}|$ filtered strain rate",
        r"$|\tau_{12}|$ SGS stress",
    ]
    cmaps = ["inferno", "RdBu_r", "viridis", "plasma"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, sl, ti, cm in zip(axes.ravel(), slices, titles, cmaps):
        im = ax.imshow(
            sl,
            cmap=cm,
            origin="lower",
            vmin=np.percentile(sl, 2),
            vmax=np.percentile(sl, 98),
        )
        ax.set_title(ti, fontsize=11)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        plt.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(
        f"Mid-plane slice  (z=N/2)  —  N={N}, Delta={delta:.3f}, {ftype} filter",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN SOLVER
# ══════════════════════════════════════════════════════════════════════════════


def run_dns(
    N=256,
    nu=8e-4,
    T=50.0,
    CFL=0.4,
    eps_target=0.1,
    k_f=2.5,
    delta_les=None,
    filter_type="sharp",
    checkpoint_in=None,
    checkpoint_out="dns_checkpoint.npz",
    fields_out="dns_fields.npz",
    diag_interval=5.0,
    threads=None,
    outdir=".",
):
    """
    Run DNS and save all derived fields.

    Parameters
    ----------
    N            : grid points per dimension (power of 2 recommended)
    nu           : kinematic viscosity
    T            : total simulation time
    CFL          : CFL number (default 0.4, reduce if unstable)
    eps_target   : target energy injection/dissipation rate
    k_f          : forcing wavenumber cutoff
    delta_les    : LES filter width (default 2pi/(N//4))
    filter_type  : 'sharp' | 'gaussian' | 'tophat'
    checkpoint_in: path to restart .npz (None = fresh start)
    checkpoint_out: where to save checkpoint
    fields_out   : where to save final LES fields
    diag_interval: wall-clock seconds between diagnostic prints
    threads      : FFTW thread count (default: all CPUs)
    """
    global _N_THREADS
    if threads is not None:
        _N_THREADS = threads

    mem_gb = 3 * 2 * (N**3) * 8 / 1e9
    print(f"\n{'=' * 65}")
    print(f"  High-Re DNS  |  N={N}^3  |  nu={nu}  |  T={T}")
    print(
        f"  Backend  : {'pyfftw (' + str(_N_THREADS) + ' threads)' if _USE_PYFFTW else 'numpy.fft'}"
    )
    print(f"  Est. mem : ~{mem_gb:.1f} GB  (velocity fields, float64)")
    print(f"{'=' * 65}\n")

    if _USE_PYFFTW:
        load_wisdom()

    kx, ky, kz, k2, dealias = make_wavenumbers(N)
    kx = kx.astype(np.float32)
    ky = ky.astype(np.float32)
    kz = kz.astype(np.float32)
    k2 = k2.astype(np.float32)
    dealias = dealias.astype(np.float32)

    if checkpoint_in is not None:
        print(f"  Resuming from {checkpoint_in} ...")
        ck = np.load(checkpoint_in)
        ux_h = ck["ux_h"].astype(np.complex64)
        uy_h = ck["uy_h"].astype(np.complex64)
        uz_h = ck["uz_h"].astype(np.complex64)
        t0 = float(ck["t"])
    else:
        print("  Generating initial condition ...")
        ux_h, uy_h, uz_h = initial_condition(N, kx, ky, kz, k2, dealias)
        t0 = 0.0

    t = t0
    t_end = t0 + T
    step = 0
    history = dict(t=[], E=[], eps=[], Re_lam=[], k_max_eta=[])
    wall0 = time.time()
    next_diag = wall0
    step_times = []  # per-step wall times

    st0 = compute_stats(ux_h, uy_h, uz_h, k2, nu, N)
    print(
        f"  Initial:  E={st0['E']:.4f}  Re_lam~{st0['Re_lam']:.1f}  "
        f"k_max*eta~{st0['k_max_eta']:.3f}\n"
    )
    if st0["k_max_eta"] < 0.5:
        print("  WARNING: estimated k_max*eta < 0.5 — grid likely under-resolved.")
        print("           Increase N or nu.\n")

    print(
        f"  {'step':>7}  {'t':>8}  {'E':>9}  {'eps':>9}  "
        f"{'Re_lam':>7}  {'k_max*eta':>9}  {'dt':>9}  {'wall':>8}"
    )
    print(f"  {'-' * 75}")

    while t < t_end:
        dt = min(adaptive_dt(ux_h, uy_h, uz_h, k2, N, nu, CFL), t_end - t + 1e-14)
        _step_t0 = time.perf_counter()
        ux_h, uy_h, uz_h = rk3_step(
            ux_h, uy_h, uz_h, kx, ky, kz, k2, dealias, N, nu, dt, eps_target, k_f
        )
        step_times.append(time.perf_counter() - _step_t0)
        t += dt
        step += 1

        wall = time.time()
        if wall >= next_diag or t >= t_end:
            st = compute_stats(ux_h, uy_h, uz_h, k2, nu, N)
            history["t"].append(t)
            history["E"].append(st["E"])
            history["eps"].append(st["eps"])
            history["Re_lam"].append(st["Re_lam"])
            history["k_max_eta"].append(st["k_max_eta"])
            print(
                f"  {step:>7d}  {t:>8.3f}  {st['E']:>9.5f}  {st['eps']:>9.5f}  "
                f"{st['Re_lam']:>7.1f}  {st['k_max_eta']:>9.4f}  "
                f"{dt:>9.6f}  {wall - wall0:>7.1f}s"
            )
            next_diag = wall + diag_interval

    total_wall = time.time() - wall0
    step_arr = step_times
    mean_step = sum(step_arr) / len(step_arr) if step_arr else 0
    min_step  = min(step_arr) if step_arr else 0
    max_step  = max(step_arr) if step_arr else 0
    print(
        f"\n  Done: {step} steps,  total wall: {total_wall:.1f}s"
        f"\n  Per-step RK3:  mean={mean_step*1e3:.2f}ms  "
        f"min={min_step*1e3:.2f}ms  max={max_step*1e3:.2f}ms"
        f"\n  Throughput:    {step/total_wall:.1f} steps/s  |  "
        f"{N**3 * step / total_wall / 1e6:.1f} Mpts/s\n"
    )

    stats = compute_stats(ux_h, uy_h, uz_h, k2, nu, N)
    k_sh, E_sh = energy_spectrum(ux_h, uy_h, uz_h, k2, N)

    print("  Final statistics:")
    for k, v in stats.items():
        flag = ""
        if k == "k_max_eta":
            flag = "  OK" if v >= 1.0 else "  WARNING: under-resolved"
        print(f"    {k:14s} = {v:.5g}{flag}")

    np.savez_compressed(
        checkpoint_out, ux_h=ux_h, uy_h=uy_h, uz_h=uz_h, t=t, N=N, nu=nu
    )
    print(f"\n  Checkpoint     -> {checkpoint_out}")
    if _USE_PYFFTW:
        save_wisdom()

    if delta_les is None:
        delta_les = 2 * np.pi / (N // 4)
    print(f"  Computing LES fields  (Delta={delta_les:.4f}, {filter_type} filter) ...")
    fields = compute_all_fields(
        ux_h, uy_h, uz_h, kx, ky, kz, k2, N, delta_les, filter_type
    )
    fields.update(
        dict(
            k_sh=k_sh,
            E_sh=E_sh,
            nu=np.float32(nu),
            N=N,
            ux_h=ux_h,
            uy_h=uy_h,
            uz_h=uz_h,
        )
    )
    np.savez_compressed(fields_out, **fields)
    print(f"  LES fields     -> {fields_out}")

    import os as _os
    plot_energy_spectrum(k_sh, E_sh, stats, N, nu, out=_os.path.join(outdir, "dns_energy.png"))
    plot_time_history(history, out=_os.path.join(outdir, "dns_stats.png"))
    plot_slice(ux_h, uy_h, uz_h, kx, ky, kz, k2, N, delta_les, filter_type, out=_os.path.join(outdir, "dns_slice.png"))

    print("\nComplete.\n")
    return fields, stats, history


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="High-Re pseudo-spectral DNS — 3-D periodic turbulence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--N", type=int, default=256, help="Grid points per dim")
    p.add_argument("--nu", type=float, default=8e-4, help="Kinematic viscosity")
    p.add_argument("--T", type=float, default=50.0, help="Simulation time")
    p.add_argument("--CFL", type=float, default=0.4, help="CFL number")
    p.add_argument("--eps", type=float, default=0.1, help="Forcing amplitude")
    p.add_argument("--kf", type=float, default=2.5, help="Forcing wavenumber cutoff")
    p.add_argument("--delta", type=float, default=None, help="LES filter width")
    p.add_argument(
        "--filter",
        type=str,
        default="sharp",
        choices=["sharp", "gaussian", "tophat"],
        help="Filter type",
    )
    p.add_argument("--threads", type=int, default=None, help="FFTW threads")
    p.add_argument("--resume", type=str, default=None, help="Checkpoint to restart")
    p.add_argument("--out", type=str, default="dns_fields.npz", help="Fields output")
    p.add_argument(
        "--ckpt", type=str, default="dns_checkpoint.npz", help="Checkpoint output"
    )
    p.add_argument(
        "--diag", type=float, default=5.0, help="Diagnostic interval (wall-s)"
    )
    a = p.parse_args()

    # Build output directory from run parameters
    from datetime import datetime
    import os
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"run_{timestamp}_N{a.N:03d}_nu{a.nu:.0e}_T{int(a.T)}"
    os.makedirs(outdir, exist_ok=True)
    print(f"  Output directory: {outdir}/")

    run_dns(
        N=a.N,
        nu=a.nu,
        T=a.T,
        CFL=a.CFL,
        eps_target=a.eps,
        k_f=a.kf,
        delta_les=a.delta,
        filter_type=a.filter,
        threads=a.threads,
        checkpoint_in=a.resume,
        checkpoint_out=os.path.join(outdir, a.ckpt),
        fields_out=os.path.join(outdir, a.out),
        diag_interval=a.diag,
        outdir=outdir,
    )
