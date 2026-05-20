"""
Renderiza una animacion del agujero negro variando el angulo polar.

La camara desciende desde una vista casi cenital (theta=30 grados, mirando
casi directo al polo del disco) hasta una vista casi ecuatorial (theta=85
grados, vista rasante tipo "Gargantua"), y luego regresa. Este movimiento
muestra la progresion dramatica de los efectos de lente gravitacional:

  - Cenital: disco casi circular, asimetria Doppler sutil
  - Intermedio: el clasico look de Interstellar con disco doblado
  - Ecuatorial: lensing extremo con multiples imagenes del disco

Salida: PNGs individuales + GIF animado + MP4.
"""

import sys
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np

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
    b_arr = camera.r * sin_psi

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


def main() -> None:
    n_frames = 60
    width = 480
    height = 272  # divisible entre 16 para compatibilidad con codecs de video
    fps = 24
    fov = 30.0
    n_steps = 3000
    phi_max = 50.0
    gamma_correction = 0.45

    # Trayectoria de theta: descenso de cenital a ecuatorial y regreso
    # theta=30 grados (60 grados arriba del ecuador) a theta=85 grados (5 grados)
    theta_min = np.radians(30.0)   # vista casi cenital
    theta_max = np.radians(85.0)   # vista casi ecuatorial
    # Oscilacion sinusoidal: ida y vuelta suave
    t = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)
    theta_values = 0.5 * (theta_min + theta_max) + 0.5 * (theta_min - theta_max) * np.cos(t)

    frames_dir = Path("outputs/frames/animation")
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Ray Tracer — Animacion: Descenso Polar")
    print("=" * 60)
    print(f"Resolucion: {width}x{height} ({width * height:,} px/frame)")
    print(f"Frames: {n_frames} | FPS: {fps} | Duracion: {n_frames / fps:.1f}s")
    print(f"Camara: r={R_CAMERA:.0f}M")
    print(f"Theta: {np.degrees(theta_min):.0f} grados (cenital) "
          f"<-> {np.degrees(theta_max):.0f} grados (ecuatorial)")

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
            R_ISCO, R_DISK_OUTER, 2200.0, 3.0,
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
        deg = np.degrees(theta_cam)
        elev = 90.0 - deg
        print(f"  Frame {frame_idx + 1:3d}/{n_frames} | "
              f"theta={deg:5.1f} ({elev:+5.1f} sobre ecuador) | {dt:.2f}s")

    t_render_total = time.perf_counter() - t_total_start
    print(f"\nTodos los frames renderizados en {t_render_total:.1f}s "
          f"({t_render_total / n_frames:.2f}s/frame)")

    # Ensamblar GIF
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
    print(f"\n{'=' * 60}")
    print(f"  Frames:     {n_frames} x {width}x{height}")
    print(f"  Render:     {t_render_total:.1f}s total ({t_render_total / n_frames:.2f}s/frame)")
    print(f"  GIF:        {gif_path} ({gif_size_mb:.1f} MB)")
    print(f"  MP4:        {mp4_path} ({mp4_size_mb:.1f} MB)")
    print(f"  Duracion:   {n_frames / fps:.1f}s a {fps} fps")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
