"""
SGS Stress Tensor Conditional PDF Analysis
===========================================
Computes and plots p(tau_ij | resolved quantity) to assess whether
a functional relationship exists between SGS stresses and resolved scales.

Usage
-----
  python sgs_conditional_pdf.py              # runs with synthetic data
  python sgs_conditional_pdf.py --data my_dns.npz  # plug in real DNS data

Expected .npz keys for real data (all shape (N,) after flattening):
  tau_11, tau_12, tau_13, tau_22, tau_23, tau_33   -- SGS stress components
  S_11, S_12, S_13, S_22, S_23, S_33               -- filtered strain rate
  Omega_12, Omega_13, Omega_23                      -- filtered vorticity
  k_res                                             -- resolved TKE (optional)
  delta                                             -- filter width scalar or array (optional)
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde, norm, pearsonr
from scipy.ndimage import uniform_filter

# ── colour palette ─────────────────────────────────────────────────────────────
C = dict(exact='#2c7bb6', smag='#d7191c', dyn='#1a9641',
         fill='#abd9e9', grid='#cccccc')

# ══════════════════════════════════════════════════════════════════════════════
#  1.  SYNTHETIC DATA GENERATOR
#      Produces a correlated (tau, resolved) dataset that mimics DNS statistics:
#      tau_ij = C_s^2 * Delta^2 * |S| * S_ij  +  noise
#      with noise amplitude ~ |S| (heteroscedastic, as seen in real DNS)
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_dns(N=80_000, seed=42, delta=1.0, Cs=0.17):
    """
    Generate synthetic SGS stress / resolved-field data.

    Returns a dict with the same keys as real DNS data.
    The noise level is intentionally large to mimic the poor pointwise
    correlation seen in actual DNS a-priori studies.
    """
    rng = np.random.default_rng(seed)

    # --- resolved strain rate: log-normal |S| distribution (intermittent) ---
    log_S_mag = rng.normal(loc=0.5, scale=0.7, size=N)
    S_mag = np.exp(log_S_mag)                          # |S̄|

    # random orientation for strain tensor (simplified to S_12 component)
    theta = rng.uniform(0, 2 * np.pi, N)
    S_12  = S_mag * np.sin(2 * theta) / 2              # traceless, symmetric

    # resolved vorticity magnitude (correlated with |S|)
    Omega_mag = S_mag * rng.lognormal(mean=0.0, sigma=0.4, size=N)

    # resolved TKE
    k_res = 0.5 * (S_mag ** 2) * rng.lognormal(0, 0.3, N)

    # --- SGS stress: Smagorinsky mean + heteroscedastic noise ---
    smag_mean = (Cs ** 2) * (delta ** 2) * S_mag * S_12   # deterministic part
    noise_std  = 1.2 * np.abs(smag_mean) + 0.3 * (Cs**2) * (delta**2) * S_mag**2
    tau_12 = smag_mean + noise_std * rng.standard_normal(N)

    # diagonal component (always negative: energy drain)
    tau_11_mean = -(Cs**2) * (delta**2) * S_mag**2 / 3
    tau_11 = tau_11_mean + 1.5 * np.abs(tau_11_mean) * rng.standard_normal(N)

    return dict(
        tau_12=tau_12,
        tau_11=tau_11,
        S_12=S_12,
        S_mag=S_mag,
        Omega_mag=Omega_mag,
        k_res=k_res,
        delta=delta,
        Cs=Cs,
        smag_mean_12=smag_mean,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  2.  LOAD REAL DATA  (override this function for your DNS format)
# ══════════════════════════════════════════════════════════════════════════════

def load_dns_data(path):
    """
    Load DNS data from a .npz file and compute derived quantities.
    Adapt this function to your own data format.
    """
    d = np.load(path)

    def get(key, default=None):
        return d[key].ravel() if key in d else default

    # strain-rate tensor components
    S11 = get('S_11', np.zeros(1))
    S12 = get('S_12', np.zeros(1))
    S13 = get('S_13', np.zeros(1))
    S22 = get('S_22', np.zeros(1))
    S23 = get('S_23', np.zeros(1))
    S33 = get('S_33', -S11 - S22)      # incompressibility

    S_mag = np.sqrt(2 * (S11**2 + S22**2 + S33**2 +
                         2*(S12**2 + S13**2 + S23**2)))

    # vorticity
    O12 = get('Omega_12', np.zeros_like(S12))
    O13 = get('Omega_13', np.zeros_like(S12))
    O23 = get('Omega_23', np.zeros_like(S12))
    Omega_mag = np.sqrt(2 * (O12**2 + O13**2 + O23**2))

    tau_12 = get('tau_12', np.zeros_like(S12))
    tau_11 = get('tau_11', np.zeros_like(S12))
    k_res  = get('k_res',  0.5 * S_mag**2)
    delta  = float(d['delta']) if 'delta' in d else 1.0
    Cs     = 0.17

    smag_mean_12 = (Cs**2) * (delta**2) * S_mag * S12

    return dict(
        tau_12=tau_12, tau_11=tau_11,
        S_12=S12, S_mag=S_mag,
        Omega_mag=Omega_mag, k_res=k_res,
        delta=delta, Cs=Cs,
        smag_mean_12=smag_mean_12,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  3.  CORE ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def conditional_stats(x, y, n_bins=30, quantile_range=(0.02, 0.98)):
    """
    Bin y conditioned on x.  Returns bin centres, conditional mean,
    std, and 10th/90th percentiles.

    Parameters
    ----------
    x : array-like  – conditioning (resolved) variable
    y : array-like  – target (SGS stress) variable
    """
    x, y = np.asarray(x), np.asarray(y)
    lo, hi = np.quantile(x, quantile_range)
    edges   = np.linspace(lo, hi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    mean, std, p10, p90, count = [], [], [], [], []
    for l, r in zip(edges[:-1], edges[1:]):
        mask = (x >= l) & (x < r)
        vals = y[mask]
        if len(vals) < 10:
            mean.append(np.nan); std.append(np.nan)
            p10.append(np.nan);  p90.append(np.nan)
            count.append(0)
        else:
            mean.append(np.mean(vals))
            std.append(np.std(vals))
            p10.append(np.percentile(vals, 10))
            p90.append(np.percentile(vals, 90))
            count.append(len(vals))

    return dict(centres=centres,
                mean=np.array(mean), std=np.array(std),
                p10=np.array(p10),   p90=np.array(p90),
                count=np.array(count))


def compute_conditional_pdfs(x, y, n_cond=5, n_tau_pts=200,
                              x_quantiles=(0.1, 0.3, 0.5, 0.7, 0.9)):
    """
    For each conditioning value of x, estimate p(y | x ≈ x_q) via KDE.

    Returns list of (x_val, tau_grid, pdf_values) tuples.
    """
    x, y = np.asarray(x), np.asarray(y)
    results = []
    y_lo, y_hi = np.quantile(y, [0.001, 0.999])
    tau_grid   = np.linspace(y_lo, y_hi, n_tau_pts)

    x_vals = np.quantile(x, x_quantiles)
    # bin width = ~10th percentile spacing
    bw = np.diff(np.quantile(x, [0.1, 0.2]))[0]

    for xv in x_vals:
        mask = np.abs(x - xv) < bw
        if mask.sum() < 30:
            continue
        kde = gaussian_kde(y[mask], bw_method='silverman')
        results.append((xv, tau_grid, kde(tau_grid)))

    return results


def pearson_r_curve(x, y, n_bins=20):
    """Local Pearson r: correlation within sliding x-bins."""
    x, y = np.asarray(x), np.asarray(y)
    edges = np.linspace(*np.quantile(x, [0.02, 0.98]), n_bins + 1)
    centres, r_vals = [], []
    for l, r in zip(edges[:-1], edges[1:]):
        mask = (x >= l) & (x < r)
        if mask.sum() < 20:
            continue
        centres.append(0.5 * (l + r))
        r_vals.append(pearsonr(x[mask], y[mask])[0])
    return np.array(centres), np.array(r_vals)


def coefficient_of_determination(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# ══════════════════════════════════════════════════════════════════════════════
#  4.  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_joint_density(ax, x, y, xlabel, ylabel, title):
    """2-D kernel density / hexbin of (x, y)."""
    xlo, xhi = np.quantile(x, [0.01, 0.99])
    ylo, yhi = np.quantile(y, [0.01, 0.99])
    hb = ax.hexbin(x, y, gridsize=60, cmap='Blues',
                   extent=[xlo, xhi, ylo, yhi],
                   mincnt=1, norm=LogNorm())
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10, fontweight='bold')
    return hb


def plot_conditional_mean_std(ax, stats, xlabel, model_x=None, model_y=None,
                               label_model='Smagorinsky'):
    c = stats['centres']
    m = stats['mean']
    s = stats['std']
    ok = stats['count'] > 10

    ax.fill_between(c[ok], (m - s)[ok], (m + s)[ok],
                    alpha=0.25, color=C['fill'], label=r'$\pm 1\sigma$')
    ax.fill_between(c[ok], stats['p10'][ok], stats['p90'][ok],
                    alpha=0.12, color='steelblue', label='10–90th pct.')
    ax.plot(c[ok], m[ok], color=C['exact'], lw=2,
            label=r'$\langle\tau_{12}|\bar{q}\rangle$')
    ax.axhline(0, color='k', lw=0.7, ls='--')

    if model_x is not None and model_y is not None:
        ax.plot(model_x, model_y, color=C['smag'], lw=2, ls='--',
                label=label_model)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(r'$\tau_{12}$', fontsize=10)
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(True, color=C['grid'], lw=0.5)


def plot_conditional_pdfs(ax, cond_pdfs, cmap_name='plasma'):
    cmap   = plt.get_cmap(cmap_name)
    n      = len(cond_pdfs)
    for i, (xv, tau_grid, pdf) in enumerate(cond_pdfs):
        col = cmap(i / max(n - 1, 1))
        ax.plot(tau_grid, pdf, color=col, lw=1.8,
                label=f'$\\bar{{q}}={xv:.2f}$')

    ax.set_xlabel(r'$\tau_{12}$', fontsize=10)
    ax.set_ylabel(r'$p(\tau_{12}\,|\,\bar{q})$', fontsize=10)
    ax.legend(fontsize=7, framealpha=0.8, ncol=2)
    ax.grid(True, color=C['grid'], lw=0.5)


def plot_conditional_variance_ratio(ax, stats, xlabel):
    """
    Plots  σ(τ|q) / |⟨τ|q⟩|  — the relative conditional std.
    If this >> 1, a functional relation is weak.
    """
    c   = stats['centres']
    ok  = (stats['count'] > 10) & (np.abs(stats['mean']) > 1e-12)
    ratio = stats['std'][ok] / np.abs(stats['mean'][ok])

    ax.semilogy(c[ok], ratio, color='purple', lw=2)
    ax.axhline(1, color='k', lw=1, ls='--', label='ratio = 1')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(r'$\sigma(\tau|\bar{q})\,/\,|\langle\tau|\bar{q}\rangle|$',
                  fontsize=10)
    ax.set_title('Relative conditional std\n(< 1 → functional relation plausible)',
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, color=C['grid'], lw=0.5)


def make_figure(data):
    tau   = data['tau_12']
    S12   = data['S_12']
    S_mag = data['S_mag']
    Omega = data['Omega_mag']
    k_res = data['k_res']
    smag  = data['smag_mean_12']
    Cs    = data['Cs']
    delta = data['delta']

    # ── global Pearson r ──────────────────────────────────────────────────────
    r_S12,   _ = pearsonr(S12,   tau)
    r_Smag,  _ = pearsonr(S_mag, tau)
    r_smag,  _ = pearsonr(smag,  tau)
    r_Omega, _ = pearsonr(Omega, tau)
    R2_smag    = coefficient_of_determination(tau, smag)

    print(f"\n── Global statistics ──────────────────────────────────────────")
    print(f"  Pearson r(τ₁₂, S̄₁₂)          = {r_S12:+.3f}")
    print(f"  Pearson r(τ₁₂, |S̄|)           = {r_Smag:+.3f}")
    print(f"  Pearson r(τ₁₂, τ_smag)         = {r_smag:+.3f}  ← key SGS model metric")
    print(f"  Pearson r(τ₁₂, |Ω̄|)           = {r_Omega:+.3f}")
    print(f"  R² (Smagorinsky prediction)     = {R2_smag:.3f}")
    print(f"──────────────────────────────────────────────────────────────\n")

    # ── conditional stats ─────────────────────────────────────────────────────
    stats_S12   = conditional_stats(S12,   tau)
    stats_Smag  = conditional_stats(S_mag, tau)
    stats_Omega = conditional_stats(Omega, tau)

    cond_pdfs_S12  = compute_conditional_pdfs(S12,   tau)
    cond_pdfs_Smag = compute_conditional_pdfs(S_mag, tau)

    # Smagorinsky model curve for comparison
    s12_range  = stats_S12['centres']
    smag_curve = (Cs**2) * (delta**2) * np.abs(s12_range) * s12_range

    # ── layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 18))
    fig.suptitle(
        r'Conditional PDF of SGS Stress $\tau_{12}$ Given Resolved Quantities'
        '\n'
        r'Testing for a Functional Relationship $\tau_{ij} = f(\bar{u}_i,\,\bar{S}_{ij},\,\Delta)$',
        fontsize=13, fontweight='bold', y=0.995
    )

    gs = gridspec.GridSpec(4, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.07, right=0.97,
                           top=0.96, bottom=0.04)

    # Row 0: joint density plots
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])

    hb0 = plot_joint_density(ax00, S12,   tau, r'$\bar{S}_{12}$',
                              r'$\tau_{12}$', r'Joint: $\tau_{12}$ vs $\bar{S}_{12}$')
    hb1 = plot_joint_density(ax01, S_mag, tau, r'$|\bar{S}|$',
                              r'$\tau_{12}$', r'Joint: $\tau_{12}$ vs $|\bar{S}|$')
    hb2 = plot_joint_density(ax02, smag,  tau, r'$\tau^{\rm Smag}_{12}$',
                              r'$\tau_{12}$', r'Joint: exact vs Smagorinsky')

    # add identity line on Smagorinsky plot
    lims = [min(smag.min(), tau.min()), max(smag.max(), tau.max())]
    ax02.plot(lims, lims, 'r--', lw=1.5, label='1:1')
    ax02.legend(fontsize=8)

    for ax, hb in [(ax00, hb0), (ax01, hb1), (ax02, hb2)]:
        plt.colorbar(hb, ax=ax, label='count', shrink=0.8)

    # Row 1: conditional mean ± std
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2])

    plot_conditional_mean_std(ax10, stats_S12, r'$\bar{S}_{12}$',
                               model_x=s12_range, model_y=smag_curve)
    ax10.set_title(r'$\langle\tau_{12}|\bar{S}_{12}\rangle$',
                   fontsize=10, fontweight='bold')

    plot_conditional_mean_std(ax11, stats_Smag, r'$|\bar{S}|$')
    ax11.set_title(r'$\langle\tau_{12}||\bar{S}|\rangle$',
                   fontsize=10, fontweight='bold')

    plot_conditional_mean_std(ax12, stats_Omega, r'$|\bar{\Omega}|$')
    ax12.set_title(r'$\langle\tau_{12}||\bar{\Omega}|\rangle$',
                   fontsize=10, fontweight='bold')

    # Row 2: conditional PDFs
    ax20 = fig.add_subplot(gs[2, 0])
    ax21 = fig.add_subplot(gs[2, 1])

    plot_conditional_pdfs(ax20, cond_pdfs_S12)
    ax20.set_title(r'$p(\tau_{12}|\bar{S}_{12})$ for different $\bar{S}_{12}$ values',
                   fontsize=10, fontweight='bold')

    plot_conditional_pdfs(ax21, cond_pdfs_Smag)
    ax21.set_title(r'$p(\tau_{12}||\bar{S}|)$ for different $|\bar{S}|$ values',
                   fontsize=10, fontweight='bold')

    # Overlay all conditional PDFs on same axis (shifted) to show shape change
    ax22 = fig.add_subplot(gs[2, 2])
    tau_norm = tau / (np.std(tau) + 1e-12)
    cond_pdfs_norm = compute_conditional_pdfs(S_mag, tau_norm)
    plot_conditional_pdfs(ax22, cond_pdfs_norm)
    ax22.set_xlabel(r'$\tau_{12}/\sigma_\tau$', fontsize=10)
    ax22.set_title(r'Normalised $p(\tau_{12}/\sigma||\bar{S}|)$'
                   '\n(shape change with conditioning)',
                   fontsize=10, fontweight='bold')

    # Row 3: relative conditional std + Pearson r curve + summary text
    ax30 = fig.add_subplot(gs[3, 0])
    ax31 = fig.add_subplot(gs[3, 1])
    ax32 = fig.add_subplot(gs[3, 2])

    plot_conditional_variance_ratio(ax30, stats_S12, r'$\bar{S}_{12}$')

    # Pearson r curve
    cx_S,  rv_S  = pearson_r_curve(S12,  tau)
    cx_Sm, rv_Sm = pearson_r_curve(smag, tau)
    ax31.plot(cx_S,  rv_S,  color=C['exact'], lw=2, label=r'$r(\tau_{12}, \bar{S}_{12})$')
    ax31.plot(cx_Sm, rv_Sm, color=C['smag'],  lw=2, ls='--',
              label=r'$r(\tau_{12}, \tau^{\rm Smag})$')
    ax31.axhline(0, color='k', lw=0.7, ls='--')
    ax31.set_xlabel(r'$\bar{S}_{12}$', fontsize=10)
    ax31.set_ylabel('Local Pearson r', fontsize=10)
    ax31.set_title('Local correlation within bins\n'
                   r'(high r $\Rightarrow$ functional relation in that bin)',
                   fontsize=9)
    ax31.legend(fontsize=8)
    ax31.grid(True, color=C['grid'], lw=0.5)
    ax31.set_ylim(-1, 1)

    # Summary panel
    ax32.axis('off')
    summary = (
        "Summary of Functional Relationship Test\n"
        "─────────────────────────────────────────\n\n"
        f"  Global Pearson r(τ₁₂, S̄₁₂)   :  {r_S12:+.3f}\n"
        f"  Global Pearson r(τ₁₂, |S̄|)    :  {r_Smag:+.3f}\n"
        f"  Global Pearson r(τ₁₂, τˢᵐᵃᵍ)  :  {r_smag:+.3f}\n"
        f"  Global Pearson r(τ₁₂, |Ω̄|)   :  {r_Omega:+.3f}\n"
        f"  Smagorinsky R²                 :  {R2_smag:.3f}\n\n"
        "Interpretation\n"
        "─────────────────────────────────────────\n\n"
        "  |r| ~ 0.0–0.3  →  very weak pointwise\n"
        "                     functional relation\n\n"
        "  |r| ~ 0.3–0.6  →  moderate statistical\n"
        "                     dependence\n\n"
        "  |r| > 0.7      →  strong functional\n"
        "                     relation plausible\n\n"
        "  Broad conditional PDFs (row 2) and\n"
        "  σ(τ|q)/|⟨τ|q⟩| >> 1 (bottom-left)\n"
        "  indicate stochastic SGS modelling\n"
        "  is physically warranted."
    )
    ax32.text(0.04, 0.97, summary,
              transform=ax32.transAxes,
              fontsize=8.5, va='top', fontfamily='monospace',
              bbox=dict(boxstyle='round,pad=0.5',
                        facecolor='#f7f7f7', edgecolor='#aaaaaa', lw=1.2))

    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  5.  MULTI-DELTA ANALYSIS  (filter width sensitivity)
# ══════════════════════════════════════════════════════════════════════════════

def plot_delta_sensitivity(deltas=(0.5, 1.0, 2.0, 4.0), N=60_000, Cs=0.17):
    """
    Shows how the conditional PDF narrows as Delta -> 0
    (in the limit Delta->0, tau->0 deterministically).
    """
    fig, axes = plt.subplots(1, len(deltas), figsize=(14, 4), sharey=False)
    fig.suptitle(r'$p(\tau_{12}|\bar{S}_{12})$  for different filter widths $\Delta$',
                 fontsize=12, fontweight='bold')

    cmap = plt.get_cmap('viridis')
    colors_delta = [cmap(i / (len(deltas) - 1)) for i in range(len(deltas))]

    for ax, delta, col in zip(axes, deltas, colors_delta):
        data    = generate_synthetic_dns(N=N, delta=delta, Cs=Cs)
        tau     = data['tau_12']
        S12     = data['S_12']
        cpdfs   = compute_conditional_pdfs(S12, tau, n_cond=3,
                                           x_quantiles=(0.25, 0.5, 0.75))
        for xv, tau_grid, pdf in cpdfs:
            ax.plot(tau_grid, pdf, lw=1.8, color=col, alpha=0.6 + 0.4*(xv > 0))
        ax.set_title(rf'$\Delta = {delta}$', fontsize=10)
        ax.set_xlabel(r'$\tau_{12}$', fontsize=9)
        ax.grid(True, color=C['grid'], lw=0.5)
        # annotate std
        ax.text(0.97, 0.97, f'σ={np.std(tau):.2f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8)

    axes[0].set_ylabel(r'$p(\tau_{12}|\bar{S}_{12})$', fontsize=10)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  6.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='SGS stress conditional PDF analysis')
    parser.add_argument('--data',   default=None,
                        help='Path to .npz DNS data file (omit for synthetic data)')
    parser.add_argument('--N',      type=int, default=80_000,
                        help='Number of synthetic samples (default 80000)')
    parser.add_argument('--out',    default='sgs_conditional_pdf.png',
                        help='Output filename for main figure')
    parser.add_argument('--out2',   default='sgs_delta_sensitivity.png',
                        help='Output filename for filter-width figure')
    args = parser.parse_args()

    if args.data is not None:
        print(f"Loading DNS data from {args.data} ...")
        data = load_dns_data(args.data)
    else:
        print(f"Generating synthetic DNS data (N={args.N}) ...")
        data = generate_synthetic_dns(N=args.N)

    print("Building main figure ...")
    fig_main = make_figure(data)
    fig_main.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"  Saved → {args.out}")

    print("Building filter-width sensitivity figure ...")
    fig_delta = plot_delta_sensitivity()
    fig_delta.savefig(args.out2, dpi=150, bbox_inches='tight')
    print(f"  Saved → {args.out2}")

    print("\nDone.")


if __name__ == '__main__':
    main()
