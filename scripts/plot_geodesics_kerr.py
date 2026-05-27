"""
Grafica geodesicas de fotones alrededor de un agujero negro de Kerr.

Traza rayos con diferentes parametros de impacto en la metrica de Kerr,
mostrando los efectos del frame-dragging:
  - Asimetria progrado/retrogrado
  - Sombra desplazada
  - Orbitas de fotones co-rotantes vs contra-rotantes

Los rayos se integran con SciPy (alta precision) y se proyectan al plano
ecuatorial (x, y) para visualizacion.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np

# Agregar raiz del proyecto al path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.constants import (
    M, kerr_horizons, kerr_isco, kerr_photon_sphere,
)
from src.geodesics import trace_kerr_geodesic


def compute_kerr_xi_eta(alpha_B: float, beta_B: float, theta_obs: float, a: float):
    """Convierte coordenadas Bardeen a constantes de Carter."""
    sin_t = np.sin(theta_obs)
    cos_t = np.cos(theta_obs)
    cot_t = cos_t / sin_t if abs(sin_t) > 1e-12 else 0.0

    xi = -alpha_B * sin_t
    eta = beta_B * beta_B - a * a * cos_t * cos_t + xi * xi * cot_t * cot_t
    return xi, eta


def plot_geodesics_kerr(
    a: float = 0.998,
    r_cam: float = 30.0,
    theta_cam: float = None,
    n_rays_per_side: int = 12,
    save_path: str | None = None,
) -> None:
    """
    Genera graficas de geodesicas en Kerr.

    Traza rayos ecuatoriales (beta_B=0) con diferentes alpha_B,
    separando progrados (alpha_B > 0, xi < 0) y retrogrados (alpha_B < 0, xi > 0).
    """
    if theta_cam is None:
        theta_cam = np.pi / 2.0 - 0.01  # casi ecuatorial

    r_plus, r_minus = kerr_horizons(a)
    r_isco_pro = kerr_isco(a, prograde=True)
    r_isco_retro = kerr_isco(a, prograde=False)
    r_ph_pro = kerr_photon_sphere(a, prograde=True)
    r_ph_retro = kerr_photon_sphere(a, prograde=False)

    print(f"Agujero negro de Kerr: a = {a:.3f}")
    print(f"  r+ = {r_plus:.3f}M, r- = {r_minus:.3f}M")
    print(f"  Esfera de fotones prograda:   r_ph = {r_ph_pro:.3f}M")
    print(f"  Esfera de fotones retrograda: r_ph = {r_ph_retro:.3f}M")
    print(f"  ISCO progrado:  {r_isco_pro:.3f}M")
    print(f"  ISCO retrogrado: {r_isco_retro:.3f}M")

    # Parametro de impacto critico aproximado para cada lado
    # Para el plot, usamos alpha_B como parametro de impacto en el plano celeste
    # Rango: desde rayos capturados hasta rayos muy deflectados
    alpha_max = 12.0  # bien por encima de b_crit ~ 5.2 para Schwarzschild

    # Rayos progrados (alpha_B > 0 -> xi < 0 -> co-rotante)
    alphas_pro = np.linspace(0.5, alpha_max, n_rays_per_side)
    # Rayos retrogrados (alpha_B < 0 -> xi > 0 -> contra-rotante)
    alphas_retro = -np.linspace(0.5, alpha_max, n_rays_per_side)

    all_alphas = np.concatenate([alphas_pro, alphas_retro])
    beta_B = 0.0  # rayos ecuatoriales

    print(f"\nTrazando {len(all_alphas)} rayos desde r_cam = {r_cam:.1f}M")

    # --- Integrar geodesicas ---
    trajectories = []
    for alpha_B in all_alphas:
        xi, eta = compute_kerr_xi_eta(alpha_B, beta_B, theta_cam, a)

        result = trace_kerr_geodesic(
            xi, eta, r_cam, theta_cam, a,
            lam_max=500.0, max_steps=10000, beta_B=beta_B,
        )

        # Convertir a cartesianas (proyeccion ecuatorial)
        r = result["r"]
        theta = result["theta"]
        phi = result["phi"]

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)

        trajectories.append({
            "alpha_B": alpha_B,
            "xi": xi,
            "x": x,
            "y": y,
            "r": r,
            "fate": result["fate"],
        })

    # --- Grafica ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.patch.set_facecolor("#0a0a1a")

    for ax_idx, (ax, title_suffix) in enumerate(zip(axes, ["Vista Completa", "Zona Central"])):
        ax.set_aspect("equal")
        ax.set_facecolor("#0a0a1a")

        # Horizonte exterior
        theta_circle = np.linspace(0, 2 * np.pi, 200)
        # Horizonte de Kerr en Boyer-Lindquist es r=r_plus (esfera)
        ax.fill(r_plus * np.cos(theta_circle), r_plus * np.sin(theta_circle),
                color="black", zorder=10)
        ax.plot(r_plus * np.cos(theta_circle), r_plus * np.sin(theta_circle),
                color="#ff4444", linewidth=1.5, linestyle="--", zorder=11)

        # Horizonte interior
        ax.plot(r_minus * np.cos(theta_circle), r_minus * np.sin(theta_circle),
                color="#ff8888", linewidth=0.8, linestyle=":", zorder=11, alpha=0.5)

        # Esferas de fotones (prograda y retrograda)
        ax.plot(r_ph_pro * np.cos(theta_circle), r_ph_pro * np.sin(theta_circle),
                color="#ffaa00", linewidth=1.0, linestyle=":", zorder=9, alpha=0.7)
        ax.plot(r_ph_retro * np.cos(theta_circle), r_ph_retro * np.sin(theta_circle),
                color="#ff6600", linewidth=1.0, linestyle=":", zorder=9, alpha=0.7)

        # ISCOs
        ax.plot(r_isco_pro * np.cos(theta_circle), r_isco_pro * np.sin(theta_circle),
                color="#44ff44", linewidth=0.8, linestyle="-.", zorder=8, alpha=0.4)
        ax.plot(r_isco_retro * np.cos(theta_circle), r_isco_retro * np.sin(theta_circle),
                color="#88ff44", linewidth=0.8, linestyle="-.", zorder=8, alpha=0.3)

        # Ergosfera (en el ecuatorial: r_ergo = M + sqrt(M^2 - a^2*cos^2(theta)), theta=pi/2 -> r_ergo = 2M)
        r_ergo = 2.0 * M
        ax.plot(r_ergo * np.cos(theta_circle), r_ergo * np.sin(theta_circle),
                color="#aa44ff", linewidth=1.0, linestyle="--", zorder=9, alpha=0.6)

        # Graficar geodesicas
        for traj in trajectories:
            is_prograde = traj["alpha_B"] > 0
            fate = traj["fate"]

            if fate == "captured":
                color = "#ff5555" if is_prograde else "#ff8833"
            elif fate == "escaped":
                color = "#33ccff" if is_prograde else "#33ff99"
            else:
                color = "#ffff33"

            alpha = 0.8
            lw = 0.9

            ax.plot(traj["x"], traj["y"], color=color, linewidth=lw, alpha=alpha)

        # Camara
        cam_x = r_cam * np.sin(theta_cam)
        cam_y = 0.0
        ax.plot(cam_x, cam_y, "w*", markersize=12, zorder=15)

        # Etiquetas
        ax.set_xlabel("x / M", color="white", fontsize=12)
        ax.set_ylabel("y / M", color="white", fontsize=12)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#333333")

        if ax_idx == 0:
            lim = r_cam * 1.1
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
        else:
            lim = 15.0
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)

        ax.set_title(
            f"Geodesicas de Kerr (a={a:.3f}) — {title_suffix}",
            color="white", fontsize=13, pad=10,
        )

    # Leyenda global
    legend_elements = [
        Line2D([0], [0], color="#33ccff", lw=2, label="Escapado progrado"),
        Line2D([0], [0], color="#33ff99", lw=2, label="Escapado retrogrado"),
        Line2D([0], [0], color="#ff5555", lw=2, label="Capturado progrado"),
        Line2D([0], [0], color="#ff8833", lw=2, label="Capturado retrogrado"),
        Line2D([0], [0], color="#ffff33", lw=2, label="Orbitando"),
        Line2D([0], [0], color="#ff4444", lw=1.5, linestyle="--",
               label=f"Horizonte (r+ = {r_plus:.3f}M)"),
        Line2D([0], [0], color="#ffaa00", lw=1, linestyle=":",
               label=f"Esfera fotones pro (r = {r_ph_pro:.3f}M)"),
        Line2D([0], [0], color="#ff6600", lw=1, linestyle=":",
               label=f"Esfera fotones retro (r = {r_ph_retro:.3f}M)"),
        Line2D([0], [0], color="#aa44ff", lw=1, linestyle="--",
               label=f"Ergosfera (r = {r_ergo:.1f}M)"),
    ]
    axes[0].legend(handles=legend_elements, loc="upper left", fontsize=9,
                   facecolor="#1a1a2e", edgecolor="#333333", labelcolor="white")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\nGrafica guardada en {save_path}")

    plt.show()


def print_kerr_validation(a: float = 0.998) -> None:
    """Reporte de validacion para geodesicas de Kerr."""
    r_plus, _ = kerr_horizons(a)
    r_isco_pro = kerr_isco(a, prograde=True)
    r_ph_pro = kerr_photon_sphere(a, prograde=True)

    print("\n" + "=" * 60)
    print(f"REPORTE DE VALIDACION — KERR a = {a:.3f}")
    print("=" * 60)

    theta_cam = np.pi / 2.0 - 0.01
    r_cam = 30.0

    # Rayo radial (xi=0, eta=0)
    print("\n--- Rayo radial (xi=0, eta=0) ---")
    result = trace_kerr_geodesic(0.0, 0.0, r_cam, theta_cam, a,
                                  lam_max=500.0, beta_B=0.0)
    print(f"  Destino: {result['fate']}")
    print(f"  r_min = {result['r'].min():.4f}M (r+ = {r_plus:.4f}M)")

    # Rayos con diferentes xi (progrados y retrogrados)
    print(f"\n--- Rayos ecuatoriales (beta_B=0) ---")
    print(f"  {'alpha_B':>8s}  {'xi':>8s}  {'destino':>10s}  {'r_min':>8s}  {'orbitas':>8s}")
    for alpha_B in [-10, -7, -5, -3, 0, 3, 5, 7, 10]:
        xi, eta = compute_kerr_xi_eta(float(alpha_B), 0.0, theta_cam, a)
        result = trace_kerr_geodesic(xi, eta, r_cam, theta_cam, a,
                                      lam_max=500.0, beta_B=0.0)
        r_min = result["r"].min()
        n_orbits = (result["phi"][-1] - result["phi"][0]) / (2 * np.pi)
        print(f"  {alpha_B:8.1f}  {xi:8.3f}  {result['fate']:>10s}  "
              f"{r_min:8.3f}  {abs(n_orbits):8.2f}")

    print("=" * 60)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")

    a = 0.998  # Gargantua
    print_kerr_validation(a)
    plot_geodesics_kerr(
        a=a,
        save_path="outputs/geodesic_plots/kerr_geodesics.png",
    )


if __name__ == "__main__":
    main()
