"""
Animacion cinematografica de 10 segundos del agujero negro de Schwarzschild.

Trayectoria con keyframes suavizados (CubicSpline periodica):
  - Descenso desde vista cenital (theta=15) hasta ecuatorial (theta=87)
  - Pausa prolongada cerca del plano ecuatorial para detallar el lensing
  - Retorno suave al punto de partida (loop continuo)

La animacion muestra la progresion dramatica del lensing gravitacional:
  - Cenital: disco casi circular, sombra redonda, Doppler sutil
  - Intermedio (~75): el clasico look de Interstellar con disco doblado
  - Ecuatorial (~87): lensing extremo, multiples imagenes, anillo de fotones

Salida: PNGs individuales + GIF animado + MP4.
"""

import sys
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera import Camera
from src.constants import RS, M, R_CAMERA, R_ISCO, R_DISK_OUTER
from src.numba_kernels import render_frame_parallel, warmup


def precompute_rays(camera):
    """Pre-computa b, e1, e2 como arreglos numpy vectorizados."""
    h, w = camera.height, camera.width
    e_r, e_theta, e_phi = camera._basis

    fov_rad = np.radians(camera.fov)
    pixel_size = fov_rad / h

    ii, jj = np.meshgrid(np.arange(w), np.arange(h))
    alpha = (ii - w / 2.0 + 0.5) * pixel_size
    beta = (h / 2.0 - jj - 0.5) * pixel_size

    tan_a = np.tan(alpha)[:, :, np.newaxis]
    tan_b = np.tan(beta)[:, :, np.newaxis]

    d = (-e_r + tan_a * e_phi + tan_b * (-e_theta))
    d_norm = np.linalg.norm(d, axis=2, keepdims=True)
    d /= d_norm

    r_hat = e_r.reshape(1, 1, 3)
    cross = np.cross(np.broadcast_to(r_hat, d.shape), d)
    sin_psi = np.linalg.norm(cross, axis=2)
    # Observador estatico a distancia finita (Synge 1966): b = r*sin(psi)/lapse
    b_arr = camera.r * sin_psi / np.sqrt(1.0 - RS / camera.r)

    e1_arr = np.broadcast_to(e_r, (h, w, 3)).copy()

    safe_sin = np.where(sin_psi[:, :, np.newaxis] > 1e-12,
                        sin_psi[:, :, np.newaxis], 1.0)
    n = cross / safe_sin

    e2_arr = np.cross(n, e1_arr)
    e2_norm = np.linalg.norm(e2_arr, axis=2, keepdims=True)
    e2_norm = np.where(e2_norm < 1e-12, 1.0, e2_norm)
    e2_arr /= e2_norm

    mask = sin_psi < 1e-12
    if np.any(mask):
        e2_arr[mask] = e_theta

    return b_arr, e1_arr, e2_arr


def build_trajectory(n_frames):
    """
    Construye la trayectoria de theta con keyframes suavizados.

    Keyframes (tiempo normalizado -> theta en grados):
      0.00 -> 15   (cenital, inicio)
      0.20 -> 55   (descenso intermedio)
      0.35 -> 80   (acercamiento al ecuador)
      0.50 -> 87   (vista ecuatorial, lensing maximo)
      0.65 -> 80   (salida del ecuador)
      0.80 -> 55   (ascenso intermedio)
      1.00 -> 15   (regreso a cenital, cierra loop)

    Usa CubicSpline periodica para transiciones suaves y loop continuo.
    """
    kf_t = np.array([0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0])
    kf_theta = np.array([15.0, 55.0, 80.0, 87.0, 80.0, 55.0, 15.0])

    cs = CubicSpline(kf_t, kf_theta, bc_type="periodic")
    t_eval = np.linspace(0.0, 1.0, n_frames, endpoint=False)
    theta_deg = cs(t_eval)

    # Clamp por seguridad (la spline puede oscilar ligeramente fuera de rango)
    theta_deg = np.clip(theta_deg, 10.0, 89.0)

    return np.radians(theta_deg), theta_deg


def main() -> None:
    n_frames = 240
    width = 640
    height = 368  # divisible entre 16 para codecs de video
    fps = 24
    fov = 30.0
    n_steps = 3000
    phi_max = 50.0
    gamma_correction = 0.45

    theta_values, theta_deg_values = build_trajectory(n_frames)

    frames_dir = Path("outputs/frames/animation")
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Ray Tracer — Animacion Cinematografica (10s)")
    print("=" * 65)
    print(f"Resolucion: {width}x{height} ({width * height:,} px/frame)")
    print(f"Frames: {n_frames} | FPS: {fps} | Duracion: {n_frames / fps:.1f}s")
    print(f"Camara: r={R_CAMERA:.0f}M, FOV={fov} grados")
    print(f"Trayectoria: theta 15 -> 87 -> 15 grados (keyframes suavizados)")
    print(f"Disco: r_inner={R_ISCO:.0f}M, r_outer={R_DISK_OUTER:.0f}M")

    # Warmup JIT
    print("\nCalentando JIT...")
    t0 = time.perf_counter()
    warmup()
    print(f"JIT listo en {time.perf_counter() - t0:.1f}s")

    frames_rgb = []
    t_total_start = time.perf_counter()

    for frame_idx, theta_cam in enumerate(theta_values):
        t_frame = time.perf_counter()

        camera = Camera(
            r=R_CAMERA,
            theta=theta_cam,
            width=width,
            height=height,
            fov=fov,
        )

        b_arr, e1_arr, e2_arr = precompute_rays(camera)
        cam_pos = camera.position

        image = render_frame_parallel(
            b_arr, e1_arr, e2_arr, cam_pos,
            camera.r, RS, M, phi_max, n_steps,
            R_ISCO, R_DISK_OUTER, 4000.0, 3.0,
        )

        # Correccion gamma
        image_display = np.power(np.clip(image, 0, 1), gamma_correction)

        # Convertir a uint8 para imageio
        frame_uint8 = (image_display * 255).astype(np.uint8)
        frames_rgb.append(frame_uint8)

        # Guardar PNG individual
        png_path = frames_dir / f"frame_{frame_idx:03d}.png"
        iio.imwrite(png_path, frame_uint8)

        dt = time.perf_counter() - t_frame
        deg = theta_deg_values[frame_idx]
        elev = 90.0 - deg
        # Barra de progreso compacta
        pct = 100 * (frame_idx + 1) / n_frames
        print(f"  [{pct:5.1f}%] Frame {frame_idx + 1:3d}/{n_frames} | "
              f"theta={deg:5.1f} ({elev:+5.1f} sobre ecuador) | {dt:.2f}s")

    t_render_total = time.perf_counter() - t_total_start
    avg_frame = t_render_total / n_frames
    print(f"\nRender completado: {t_render_total:.1f}s total ({avg_frame:.2f}s/frame)")

    # Ensamblar GIF (calidad reducida para tamanio razonable)
    print("\nEnsamblando GIF...")
    gif_path = Path("outputs/animation.gif")
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
    mp4_path = Path("outputs/animation.mp4")
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
    print(f"  Frames:     {n_frames} x {width}x{height}")
    print(f"  Render:     {t_render_total:.1f}s ({avg_frame:.2f}s/frame)")
    print(f"  GIF:        {gif_path} ({gif_size_mb:.1f} MB)")
    print(f"  MP4:        {mp4_path} ({mp4_size_mb:.1f} MB)")
    print(f"  Duracion:   {n_frames / fps:.1f}s a {fps} fps")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
