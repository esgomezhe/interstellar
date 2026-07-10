"""
Renderiza un solo frame del agujero negro con disco de acrecion.

Usa los kernels Numba para integracion RK4 paralela con prange.
La primera ejecucion incluye compilacion JIT (~10-20s); las siguientes
usan el cache en disco y son instantaneas.
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera import Camera
from src.constants import RS, M, R_CAMERA, R_ISCO, R_DISK_OUTER
from src.numba_kernels import render_frame_parallel, warmup


def precompute_rays(camera):
    """
    Extrae los datos de cada rayo como arreglos numpy para Numba.

    Vectorizado con numpy: calcula b, e1, e2 para todos los pixeles
    de una sola vez usando broadcasting, sin loops Python.

    Returns:
        (b_arr, e1_arr, e2_arr) con formas (h, w), (h, w, 3), (h, w, 3).
    """
    h, w = camera.height, camera.width
    e_r, e_theta, e_phi = camera._basis

    fov_rad = np.radians(camera.fov)
    pixel_size = fov_rad / h

    # Meshgrid de indices de pixel
    ii, jj = np.meshgrid(np.arange(w), np.arange(h))
    alpha = (ii - w / 2.0 + 0.5) * pixel_size
    beta = (h / 2.0 - jj - 0.5) * pixel_size

    # Direcciones de rayo: d = -e_r + tan(alpha)*e_phi + tan(beta)*(-e_theta)
    tan_a = np.tan(alpha)[:, :, np.newaxis]  # (h, w, 1)
    tan_b = np.tan(beta)[:, :, np.newaxis]

    d = (-e_r + tan_a * e_phi + tan_b * (-e_theta))  # broadcast a (h, w, 3)
    d_norm = np.linalg.norm(d, axis=2, keepdims=True)
    d /= d_norm

    # Parametro de impacto: b = r * |e_r x d|
    r_hat = e_r.reshape(1, 1, 3)
    cross = np.cross(np.broadcast_to(r_hat, d.shape), d)
    sin_psi = np.linalg.norm(cross, axis=2)
    # Observador estatico a distancia finita (Synge 1966): b = r*sin(psi)/lapse
    b_arr = camera.r * sin_psi / np.sqrt(1.0 - RS / camera.r)

    # Plano orbital: e1 = r_hat, e2 = (cross/|cross|) x e1
    e1_arr = np.broadcast_to(e_r, (h, w, 3)).copy()

    # Normal al plano orbital: n = cross / sin_psi
    safe_sin = np.where(sin_psi[:, :, np.newaxis] > 1e-12,
                        sin_psi[:, :, np.newaxis], 1.0)
    n = cross / safe_sin

    # e2 = n x e1
    e2_arr = np.cross(n, e1_arr)
    e2_norm = np.linalg.norm(e2_arr, axis=2, keepdims=True)
    e2_norm = np.where(e2_norm < 1e-12, 1.0, e2_norm)
    e2_arr /= e2_norm

    # Pixeles con sin_psi ~ 0 (apuntan directo al centro): e2 = e_theta
    mask = sin_psi < 1e-12
    if np.any(mask):
        e2_arr[mask] = e_theta

    return b_arr, e1_arr, e2_arr


def main() -> None:
    # Resolucion recomendada por el documento (ahora viable con Numba)
    camera = Camera(
        r=R_CAMERA,
        theta=np.radians(75.0),
        width=400,
        height=225,
        fov=30.0,
    )

    print("=" * 60)
    print("  Ray Tracer de Agujero Negro — Render Optimizado (Numba)")
    print("=" * 60)
    print(f"Resolucion: {camera.width}x{camera.height} ({camera.width * camera.height:,} pixeles)")
    print(f"Camara: r={camera.r:.0f}M, theta={np.degrees(camera.theta):.1f} grados")
    print(f"Disco: r_interior={R_ISCO:.0f}M, r_exterior={R_DISK_OUTER:.0f}M")
    print(f"Efectos: redshift gravitacional + Doppler kepleriano + beaming")
    print(f"Optimizacion: RK4 @njit + prange paralelo + cache por simetria")

    # Warmup JIT (primera vez compila, despues usa cache en disco)
    print("\nCalentando JIT (primera vez toma ~10-20s, luego es instantaneo)...")
    t0_warmup = time.perf_counter()
    warmup()
    t_warmup = time.perf_counter() - t0_warmup
    print(f"JIT listo en {t_warmup:.1f}s")

    # Pre-computar datos de rayos (Python puro, no paralelizado)
    print("\nPre-computando rayos...")
    t0_rays = time.perf_counter()
    b_arr, e1_arr, e2_arr = precompute_rays(camera)
    t_rays = time.perf_counter() - t0_rays
    print(f"Rayos pre-computados en {t_rays:.2f}s")

    # Render paralelo con Numba
    cam_pos = camera.position
    n_steps = 3000
    phi_max = 50.0

    print(f"\nRenderizando ({n_steps} pasos RK4 por geodesica)...")
    t0_render = time.perf_counter()
    image = render_frame_parallel(
        b_arr, e1_arr, e2_arr, cam_pos,
        camera.r, RS, M, phi_max, n_steps,
        R_ISCO, R_DISK_OUTER, 4000.0, 3.0,
    )
    t_render = time.perf_counter() - t0_render

    total_pixels = camera.width * camera.height
    px_per_sec = total_pixels / t_render
    print(f"Render completado en {t_render:.2f}s ({px_per_sec:,.0f} px/s)")

    # Speedup estimado vs scipy (baseline: ~8 min para 200x150 = 30k px)
    baseline_px_per_sec = 30000 / (8 * 60)  # ~62.5 px/s
    speedup = px_per_sec / baseline_px_per_sec
    print(f"Speedup estimado vs scipy: ~{speedup:.0f}x")

    # Correccion gamma para mejor contraste visual
    gamma = 0.45
    image_display = np.power(np.clip(image, 0, 1), gamma)

    # Guardar imagen
    out_path = Path("outputs/frames/frame_numba.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6.75), facecolor="black")
    ax.imshow(image_display, origin="upper", aspect="equal")
    ax.set_axis_off()
    ax.set_title(
        "Agujero Negro de Schwarzschild — Render Numba\n"
        "RK4 @njit + prange paralelo + cache por simetria",
        color="white", fontsize=13, pad=10,
    )
    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    print(f"\nGuardado: {out_path}")

    # Guardar datos crudos
    np.save("outputs/frames/frame_numba_raw.npy", image)
    print("Datos crudos: outputs/frames/frame_numba_raw.npy")

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
