"""
Renderiza un solo frame del agujero negro de Kerr con disco de acrecion.

Usa el integrador RK4 de Carter (6 variables) con Numba paralelo.
A diferencia de Schwarzschild, el spin rompe la simetria esferica:
  - El ISCO depende del spin (Bardeen-Press-Teukolsky)
  - No hay simetria horizontal para explotar
  - El frame-dragging genera asimetria progrado/retrogrado visible

Uso:
  python scripts/render_frame_kerr.py              # spin por defecto (0.998)
  python scripts/render_frame_kerr.py --spin 0.5   # spin personalizado
"""

import sys
import time
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.camera import Camera
from src.constants import M, R_DISK_OUTER, kerr_isco, kerr_horizons
from src.numba_kernels import kerr_render_frame_parallel, warmup_kerr


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Kerr black hole frame")
    parser.add_argument("--spin", type=float, default=0.998,
                        help="Spin parameter a (0 to <1)")
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument("--height", type=int, default=225)
    parser.add_argument("--fov", type=float, default=30.0)
    parser.add_argument("--theta", type=float, default=75.0,
                        help="Camera polar angle in degrees")
    parser.add_argument("--r-cam", type=float, default=30.0,
                        help="Camera distance in M")
    args = parser.parse_args()

    a = max(0.0, min(args.spin, 0.9999))
    r_cam = args.r_cam
    theta_cam = np.radians(args.theta)

    # ISCO y horizonte dinamicos
    r_isco = kerr_isco(a, prograde=True)
    r_plus, r_minus = kerr_horizons(a)
    r_disk_outer = R_DISK_OUTER

    camera = Camera(
        r=r_cam,
        theta=theta_cam,
        width=args.width,
        height=args.height,
        fov=args.fov,
    )

    print("=" * 60)
    print("  Ray Tracer de Agujero Negro de Kerr — Render Numba")
    print("=" * 60)
    print(f"Spin: a = {a:.3f}")
    print(f"Resolucion: {camera.width}x{camera.height} "
          f"({camera.width * camera.height:,} pixeles)")
    print(f"Camara: r={r_cam:.0f}M, theta={np.degrees(theta_cam):.1f} grados")
    print(f"Horizonte: r+ = {r_plus:.3f}M, r- = {r_minus:.3f}M")
    print(f"ISCO (progrado): {r_isco:.3f}M")
    print(f"Disco: r_interior={r_isco:.3f}M, r_exterior={r_disk_outer:.0f}M")
    print(f"Efectos: redshift gravitacional + Doppler Kerr + beaming")

    # Warmup JIT
    print("\nCalentando JIT Kerr (primera vez toma ~20-30s)...")
    t0_warmup = time.perf_counter()
    warmup_kerr()
    t_warmup = time.perf_counter() - t0_warmup
    print(f"JIT listo en {t_warmup:.1f}s")

    # Pre-computar constantes de Carter para cada pixel
    print("\nPre-computando constantes de Carter...")
    t0_rays = time.perf_counter()
    xi_arr, eta_arr, beta_B_arr = camera.all_carter_params(a)
    t_rays = time.perf_counter() - t0_rays
    print(f"Carter params en {t_rays:.2f}s")

    # Render paralelo
    n_steps = 3000
    lam_max = min(float(r_cam + 300), 0.15 * n_steps)
    base_temp = 2200.0
    beaming_power = 3.0

    print(f"\nRenderizando ({n_steps} pasos RK4 por geodesica, "
          f"lam_max={lam_max:.0f})...")
    t0_render = time.perf_counter()
    image = kerr_render_frame_parallel(
        xi_arr, eta_arr, beta_B_arr,
        r_cam, theta_cam, a, M,
        lam_max, n_steps,
        r_isco, r_disk_outer, base_temp, beaming_power,
    )
    t_render = time.perf_counter() - t0_render

    total_pixels = camera.width * camera.height
    px_per_sec = total_pixels / t_render
    print(f"Render completado en {t_render:.2f}s ({px_per_sec:,.0f} px/s)")

    # Correccion gamma
    gamma = 0.45
    image_display = np.power(np.clip(image, 0, 1), gamma)

    # Guardar imagen
    out_path = Path("outputs/frames/frame_kerr.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor="black")
    ax.imshow(image_display, origin="upper", aspect="equal")
    ax.set_axis_off()
    ax.set_title(
        f"Agujero Negro de Kerr (a={a:.3f}) — Render Numba\n"
        f"RK4 Carter eqs + prange paralelo | ISCO={r_isco:.2f}M",
        color="white", fontsize=13, pad=10,
    )
    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    print(f"\nGuardado: {out_path}")

    # Datos crudos
    np.save("outputs/frames/frame_kerr_raw.npy", image)
    print("Datos crudos: outputs/frames/frame_kerr_raw.npy")

    # Estadisticas
    brightness = np.mean(image, axis=2)
    n_black = np.sum(brightness == 0)
    n_disk = np.sum(brightness > 0)
    print(f"\nEstadisticas: {n_black:,} pixeles negro, {n_disk:,} disco "
          f"({100 * n_disk / brightness.size:.1f}%)")
    if n_disk > 0:
        print(f"Brillo max: {brightness.max():.4f}, promedio disco: "
              f"{brightness[brightness > 0].mean():.4f}")

    # Resumen de tiempos
    print(f"\n{'=' * 60}")
    print(f"  Warmup JIT:     {t_warmup:>8.1f}s")
    print(f"  Pre-compute:    {t_rays:>8.2f}s")
    print(f"  Render Numba:   {t_render:>8.2f}s")
    print(f"  Total:          {t_warmup + t_rays + t_render:>8.2f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
