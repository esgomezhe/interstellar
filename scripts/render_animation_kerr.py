"""
Animacion cinematografica de 10 segundos del agujero negro de Kerr.

Trayectoria con keyframes suavizados (CubicSpline periodica):
  - Descenso desde vista cenital (theta=15) hasta ecuatorial (theta=87)
  - Pausa prolongada cerca del plano ecuatorial para detallar el lensing
  - Retorno suave al punto de partida (loop continuo)

La animacion muestra el lensing de Kerr con frame-dragging:
  - Sombra asimetrica que se desplaza con el spin
  - Efecto Doppler progrado/retrogrado visible
  - ISCO dinamico segun spin (Bardeen-Press-Teukolsky)
  - Anillo de fotones brillante con beaming relativista

Post-procesado por frame: seam fix + despeckle + bloom + gamma.

Salida: PNGs individuales + GIF animado + MP4.

Uso:
  python scripts/render_animation_kerr.py                # spin 0.998
  python scripts/render_animation_kerr.py --spin 0.5     # spin personalizado
  python scripts/render_animation_kerr.py --fast          # 120 frames, menor res
"""

import sys
import time
import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter, median_filter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.camera import Camera
from src.constants import M, R_DISK_OUTER, kerr_isco, kerr_horizons
from src.numba_kernels import kerr_render_frame_parallel, warmup_kerr


# ---------------------------------------------------------------------------
# Post-procesado (identico al de render_frame_kerr.py)
# ---------------------------------------------------------------------------

def fix_meridional_seam(image, half_width=4):
    """Interpola la columna central (xi ~ 0) para eliminar artefacto meridional."""
    fixed = image.copy()
    h, w = image.shape[:2]
    mid = w // 2

    col_start = max(0, mid - half_width)
    col_end = min(w, mid + half_width + 1)
    left_col = max(0, col_start - 1)
    right_col = min(w - 1, col_end)

    for c in range(col_start, col_end):
        t = (c - col_start + 0.5) / (col_end - col_start)
        fixed[:, c, :] = (1.0 - t) * image[:, left_col, :] + t * image[:, right_col, :]

    return fixed


def despeckle(image, size=3):
    """Median filter 3x3 para eliminar hot/dead pixels."""
    cleaned = np.empty_like(image)
    for c in range(3):
        cleaned[:, :, c] = median_filter(image[:, :, c], size=size)
    return cleaned


def apply_bloom(image, sigma_narrow=2.0, sigma_wide=12.0,
                strength_narrow=0.35, strength_wide=0.18):
    """Bloom de dos pasadas: scattering cercano + halo difuso."""
    bloomed = image.copy()
    for c in range(3):
        narrow = gaussian_filter(image[:, :, c], sigma=sigma_narrow)
        wide = gaussian_filter(image[:, :, c], sigma=sigma_wide)
        bloomed[:, :, c] += strength_narrow * narrow + strength_wide * wide
    return np.clip(bloomed, 0, 1)


def postprocess_frame(image, height):
    """Pipeline completo de post-procesado para un frame."""
    scale = height / 540.0

    img = fix_meridional_seam(image, half_width=4)
    img = despeckle(img, size=3)
    img = apply_bloom(
        img,
        sigma_narrow=2.0 * scale,
        sigma_wide=10.0 * scale,
        strength_narrow=0.35,
        strength_wide=0.15,
    )
    img = np.power(np.clip(img, 0, 1), 0.45)  # gamma
    return img


# ---------------------------------------------------------------------------
# Trayectoria de camara
# ---------------------------------------------------------------------------

def build_trajectory(n_frames):
    """
    Trayectoria theta con keyframes suavizados (CubicSpline periodica).

    Keyframes:
      0.00 -> 15   cenital
      0.20 -> 55   descenso intermedio
      0.35 -> 80   acercamiento al ecuador
      0.50 -> 87   vista ecuatorial (lensing maximo)
      0.65 -> 80   salida del ecuador
      0.80 -> 55   ascenso intermedio
      1.00 -> 15   regreso a cenital (cierra loop)
    """
    kf_t = np.array([0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0])
    kf_theta = np.array([15.0, 55.0, 80.0, 87.0, 80.0, 55.0, 15.0])

    cs = CubicSpline(kf_t, kf_theta, bc_type="periodic")
    t_eval = np.linspace(0.0, 1.0, n_frames, endpoint=False)
    theta_deg = cs(t_eval)
    theta_deg = np.clip(theta_deg, 10.0, 89.0)

    return np.radians(theta_deg), theta_deg


def main() -> None:
    parser = argparse.ArgumentParser(description="Kerr black hole animation")
    parser.add_argument("--spin", type=float, default=0.998,
                        help="Spin parameter a (0 to <1)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: 120 frames, lower resolution")
    parser.add_argument("--r-cam", type=float, default=30.0,
                        help="Camera distance in M")
    args = parser.parse_args()

    a = max(0.0, min(args.spin, 0.9999))
    r_cam = args.r_cam

    if args.fast:
        n_frames = 120
        width = 480
        height = 272  # divisible entre 16
        fps = 24
    else:
        n_frames = 240
        width = 640
        height = 368  # divisible entre 16
        fps = 24

    fov = 30.0
    n_steps = 3000
    base_temp = 4000.0
    beaming_power = 3.0

    # ISCO y horizonte dinamicos
    r_isco = kerr_isco(a, prograde=True)
    r_plus, r_minus = kerr_horizons(a)
    r_disk_outer = R_DISK_OUTER

    theta_values, theta_deg_values = build_trajectory(n_frames)

    frames_dir = Path("outputs/frames/animation_kerr")
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Ray Tracer — Animacion Kerr Cinematografica")
    print("=" * 65)
    print(f"Spin: a = {a:.3f}")
    print(f"Resolucion: {width}x{height} ({width * height:,} px/frame)")
    print(f"Frames: {n_frames} | FPS: {fps} | Duracion: {n_frames / fps:.1f}s")
    print(f"Camara: r={r_cam:.0f}M, FOV={fov} grados")
    print(f"Horizonte: r+ = {r_plus:.3f}M | ISCO = {r_isco:.3f}M")
    print(f"Trayectoria: theta 15 -> 87 -> 15 grados (keyframes suavizados)")
    print(f"Disco: r_inner={r_isco:.3f}M, r_outer={r_disk_outer:.0f}M")
    print(f"Post-procesado: seam fix + despeckle + bloom + gamma")

    # Warmup JIT
    print("\nCalentando JIT Kerr...")
    t0 = time.perf_counter()
    warmup_kerr()
    print(f"JIT listo en {time.perf_counter() - t0:.1f}s")

    frames_rgb = []
    t_total_start = time.perf_counter()

    for frame_idx, theta_cam in enumerate(theta_values):
        t_frame = time.perf_counter()

        camera = Camera(
            r=r_cam,
            theta=theta_cam,
            width=width,
            height=height,
            fov=fov,
        )

        # Pre-computar constantes de Carter
        xi_arr, eta_arr, beta_B_arr = camera.all_carter_params(a)

        # lam_max adaptativo
        lam_max = min(float(r_cam + 300), 0.15 * n_steps)

        # Render
        image = kerr_render_frame_parallel(
            xi_arr, eta_arr, beta_B_arr,
            r_cam, theta_cam, a, M,
            lam_max, n_steps,
            r_isco, r_disk_outer, base_temp, beaming_power,
        )

        # Post-procesado
        image_display = postprocess_frame(image, height)

        # Convertir a uint8
        frame_uint8 = (image_display * 255).astype(np.uint8)
        frames_rgb.append(frame_uint8)

        # Guardar PNG individual
        png_path = frames_dir / f"frame_{frame_idx:03d}.png"
        iio.imwrite(png_path, frame_uint8)

        dt = time.perf_counter() - t_frame
        deg = theta_deg_values[frame_idx]
        elev = 90.0 - deg
        pct = 100 * (frame_idx + 1) / n_frames
        elapsed = time.perf_counter() - t_total_start
        eta_sec = (elapsed / (frame_idx + 1)) * (n_frames - frame_idx - 1)
        print(f"  [{pct:5.1f}%] Frame {frame_idx + 1:3d}/{n_frames} | "
              f"theta={deg:5.1f} ({elev:+5.1f}) | {dt:.2f}s | "
              f"ETA {eta_sec:.0f}s")

    t_render_total = time.perf_counter() - t_total_start
    avg_frame = t_render_total / n_frames
    print(f"\nRender completado: {t_render_total:.1f}s total ({avg_frame:.2f}s/frame)")

    # Ensamblar GIF
    print("\nEnsamblando GIF...")
    gif_path = Path("outputs/animation_kerr.gif")
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / fps)
    iio.imwrite(
        gif_path,
        frames_rgb,
        loop=0,
        duration=duration_ms,
    )
    gif_size_mb = gif_path.stat().st_size / (1024 * 1024)
    print(f"GIF guardado: {gif_path} ({gif_size_mb:.1f} MB)")

    # Ensamblar MP4
    print("Ensamblando MP4...")
    mp4_path = Path("outputs/animation_kerr.mp4")
    iio.imwrite(
        mp4_path,
        frames_rgb,
        fps=fps,
        codec="libx264",
    )
    mp4_size_mb = mp4_path.stat().st_size / (1024 * 1024)
    print(f"MP4 guardado: {mp4_path} ({mp4_size_mb:.1f} MB)")

    # Resumen
    print(f"\n{'=' * 65}")
    print(f"  Spin:       a = {a:.3f}")
    print(f"  Frames:     {n_frames} x {width}x{height}")
    print(f"  Render:     {t_render_total:.1f}s ({avg_frame:.2f}s/frame)")
    print(f"  GIF:        {gif_path} ({gif_size_mb:.1f} MB)")
    print(f"  MP4:        {mp4_path} ({mp4_size_mb:.1f} MB)")
    print(f"  Duracion:   {n_frames / fps:.1f}s a {fps} fps")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
