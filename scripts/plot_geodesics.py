"""
Grafica geodesicas de fotones alrededor de un agujero negro de Schwarzschild.

Traza 20 rayos con diferentes parametros de impacto y genera una grafica
de trayectorias mostrando:
  - Rayos con b >> b_crit: deflexion leve (casi lineas rectas)
  - Rayos con b ~ b_crit: multiples orbitas alrededor de la esfera de fotones
  - Rayos con b < b_crit: capturados por el agujero negro
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Agregar raiz del proyecto al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import RS, B_CRIT, R_ISCO, R_CAMERA, R_PHOTON
from src.geodesics import trace_rays, classical_deflection, RayFate


def plot_geodesics(
    r_cam: float = R_CAMERA,
    n_rays: int = 20,
    save_path: str | None = None,
) -> None:
    """Genera una grafica cartesiana de trayectorias de fotones alrededor del agujero negro.

    Los rayos se lanzan desde la posicion de la camara con parametros de impacto
    que van desde muy por debajo de b_crit hasta muy por encima.
    """
    # Parametros de impacto: desde 0.5*b_crit hasta 3*b_crit, con mas densidad cerca de b_crit
    b_far = np.linspace(1.5 * B_CRIT, 3.0 * B_CRIT, 7)        # rayos que escapan
    b_near = np.linspace(0.95 * B_CRIT, 1.15 * B_CRIT, 8)      # cerca del critico
    b_close = np.linspace(0.3 * B_CRIT, 0.85 * B_CRIT, 5)      # rayos capturados

    impact_params = np.sort(np.concatenate([b_far, b_near, b_close]))

    print(f"Trazando {len(impact_params)} rayos desde r_cam = {r_cam:.1f} M")
    print(f"b_crit = {B_CRIT:.4f} M")
    print(f"Parametros de impacto: {impact_params.round(3)}")

    results = trace_rays(impact_params, r_cam, phi_max=50.0)

    # --- Configuracion de la grafica ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.set_aspect("equal")
    ax.set_facecolor("#0a0a1a")
    fig.patch.set_facecolor("#0a0a1a")

    # Dibujar agujero negro (circulo relleno en r = rs)
    bh_circle = patches.Circle((0, 0), RS, color="black", zorder=10)
    ax.add_patch(bh_circle)

    # Dibujar borde del horizonte de eventos
    horizon = patches.Circle((0, 0), RS, fill=False, edgecolor="#ff4444",
                              linewidth=1.5, linestyle="--", zorder=11,
                              label=f"Horizonte de Eventos (r = {RS:.0f}M)")
    ax.add_patch(horizon)

    # Dibujar esfera de fotones
    photon_sphere = patches.Circle((0, 0), R_PHOTON, fill=False,
                                    edgecolor="#ffaa00", linewidth=1.0,
                                    linestyle=":", zorder=9,
                                    label=f"Esfera de Fotones (r = {R_PHOTON:.0f}M)")
    ax.add_patch(photon_sphere)

    # Dibujar ISCO
    isco_circle = patches.Circle((0, 0), R_ISCO, fill=False,
                                  edgecolor="#44ff44", linewidth=0.8,
                                  linestyle="-.", zorder=8, alpha=0.5,
                                  label=f"ISCO (r = {R_ISCO:.0f}M)")
    ax.add_patch(isco_circle)

    # Dibujar frontera de la sombra (circulo de radio b_crit centrado en el agujero)
    shadow = patches.Circle((0, 0), B_CRIT, fill=False,
                             edgecolor="#8888ff", linewidth=0.8,
                             linestyle=":", zorder=7, alpha=0.4,
                             label=f"Sombra (b_crit = {B_CRIT:.2f}M)")
    ax.add_patch(shadow)

    # Mapa de colores: capturado=rojo, escapado=cian, orbitando=amarillo
    colors = {
        RayFate.CAPTURED: "#ff3333",
        RayFate.ESCAPED: "#33ccff",
        RayFate.ORBITING: "#ffff33",
    }

    # Graficar cada geodesica
    for result in results:
        color = colors[result.fate]
        alpha = 0.9 if result.fate == RayFate.ORBITING else 0.7
        lw = 1.5 if abs(result.b - B_CRIT) < 0.3 else 0.8

        ax.plot(result.x, result.y, color=color, linewidth=lw, alpha=alpha)

        # Graficar tambien el rayo reflejado (b -> -b, simetrico)
        ax.plot(result.x, -result.y, color=color, linewidth=lw, alpha=alpha * 0.5)

    # Marcador de posicion de la camara
    ax.plot(r_cam, 0, "w*", markersize=12, zorder=15, label=f"Camara (r = {r_cam:.0f}M)")

    # Limites y etiquetas
    lim = r_cam * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x / M", color="white", fontsize=12)
    ax.set_ylabel("y / M", color="white", fontsize=12)
    ax.set_title(
        "Geodesicas de Fotones alrededor de un Agujero Negro de Schwarzschild\n"
        f"({len(impact_params)} rayos, b desde {impact_params[0]:.2f} hasta {impact_params[-1]:.2f} M)",
        color="white", fontsize=14, pad=15,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333333")

    # Leyenda
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#ff3333", lw=2, label="Capturado (b < b_crit)"),
        Line2D([0], [0], color="#33ccff", lw=2, label="Escapado (b > b_crit)"),
        Line2D([0], [0], color="#ffff33", lw=2, label="Orbitando (b ~ b_crit)"),
        horizon,
        photon_sphere,
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10,
              facecolor="#1a1a2e", edgecolor="#333333", labelcolor="white")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Grafica guardada en {save_path}")

    plt.show()


def print_validation_report(r_cam: float = R_CAMERA) -> None:
    """Imprime la validacion numerica del integrador de geodesicas."""
    print("\n" + "=" * 60)
    print("REPORTE DE VALIDACION FISICA")
    print("=" * 60)

    # Prueba 1: Deflexion en campo debil (b >> b_crit)
    print("\n--- Prueba 1: Deflexion en campo debil (b >> b_crit) ---")
    print("Prediccion de Einstein: dphi ~ 4M/b = 2*rs/b")
    test_bs = [20.0, 30.0, 50.0, 100.0]
    for b in test_bs:
        from src.geodesics import trace_geodesic
        result = trace_geodesic(b, r_cam, phi_max=20.0)

        # Deflexion medida: cambio total de phi menos pi (linea recta = pi)
        if result.fate == RayFate.ESCAPED:
            # El rayo entra, se curva y sale. En espacio plano recorreria
            # pi radianes. La deflexion es el exceso.
            delta_phi_measured = result.phi[-1] - np.pi

            # Para un rayo que entra y sale, hay que tener en cuenta la
            # distancia finita de la camara. Pero para b >> b_crit la
            # trayectoria es casi recta asi que el angulo total ~ pi + deflexion
            delta_phi_expected = classical_deflection(b)

            # Un enfoque mas simple: medir el angulo entre los vectores
            # de direccion inicial y final
            dx = np.diff(result.x)
            dy = np.diff(result.y)
            angle_in = np.arctan2(dy[0], dx[0])
            angle_out = np.arctan2(dy[-1], dx[-1])
            deflection = abs(angle_out - angle_in) - 0.0
            deflection_alt = abs(np.arctan2(result.y[-1] - result.y[0],
                                            result.x[-1] - result.x[0]))

            print(f"  b = {b:6.1f}M | dphi esperado = {delta_phi_expected:.6f} rad "
                  f"({np.degrees(delta_phi_expected):.4f} deg) | "
                  f"phi total = {result.phi[-1]:.4f} rad | destino: {result.fate.value}")

    # Prueba 2: Rayo critico (b ~ b_crit)
    print(f"\n--- Prueba 2: Parametro de impacto critico (b_crit = {B_CRIT:.4f} M) ---")
    for b in [B_CRIT * 1.001, B_CRIT * 1.0001, B_CRIT * 0.999]:
        from src.geodesics import trace_geodesic
        result = trace_geodesic(b, r_cam, phi_max=100.0)
        r_min = result.r.min()
        print(f"  b = {b:.4f}M | orbitas: {result.n_orbits:.2f} | "
              f"r_min = {r_min:.3f}M | destino: {result.fate.value}")

    # Prueba 3: Rayos capturados (b < b_crit)
    print(f"\n--- Prueba 3: Rayos capturados (b < b_crit) ---")
    for b in [1.0, 2.0, 3.0, 4.0, B_CRIT * 0.9]:
        from src.geodesics import trace_geodesic
        result = trace_geodesic(b, r_cam, phi_max=50.0)
        print(f"  b = {b:.4f}M | destino: {result.fate.value} | "
              f"r_min = {result.r.min():.3f}M | phi final = {result.phi[-1]:.2f} rad")

    print("\n" + "=" * 60)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")

    print_validation_report()
    plot_geodesics(save_path="outputs/geodesic_plots/phase1_geodesics.png")


if __name__ == "__main__":
    main()
