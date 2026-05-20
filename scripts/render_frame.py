"""Renderiza un solo frame del agujero negro con disco de acrecion."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera import Camera
from src.constants import R_CAMERA
from src.disk import AccretionDisk
from src.raytracer import render_frame


def main() -> None:
    # Camara ligeramente arriba del plano ecuatorial para el look clasico de Interstellar
    # theta = 75 grados significa 15 grados arriba del ecuador
    camera = Camera(
        r=R_CAMERA,
        theta=np.radians(75.0),
        width=200,
        height=150,
        fov=30.0,
    )

    disk = AccretionDisk()

    print(f"Renderizando frame {camera.width}x{camera.height}...")
    print(f"Camara: r={camera.r:.0f}M, theta={np.degrees(camera.theta):.1f} grados")
    print(f"Disco: r_interior={disk.r_inner:.0f}M, r_exterior={disk.r_outer:.0f}M")
    print("Efectos: redshift gravitacional + Doppler kepleriano + beaming")

    image = render_frame(camera, disk, phi_max=50.0, enable_effects=True)

    # Correccion gamma para mejor contraste visual
    gamma = 0.45
    image_display = np.power(np.clip(image, 0, 1), gamma)

    # Guardar imagen en color
    out_path = Path("outputs/frames/frame_phase3.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7.5), facecolor="black")
    ax.imshow(image_display, origin="upper", aspect="equal")
    ax.set_axis_off()
    ax.set_title(
        "Agujero Negro de Schwarzschild con Disco de Acrecion\n"
        "Redshift gravitacional + Doppler kepleriano + beaming",
        color="white", fontsize=13, pad=10,
    )
    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    print(f"Guardado: {out_path}")

    # Guardar datos crudos
    np.save("outputs/frames/frame_phase3_raw.npy", image)
    print("Datos crudos guardados: outputs/frames/frame_phase3_raw.npy")

    # Estadisticas
    brightness = np.mean(image, axis=2)
    n_black = np.sum(brightness == 0)
    n_disk = np.sum(brightness > 0)
    print(f"\nEstadisticas: {n_black} pixeles negro, {n_disk} disco "
          f"({100 * n_disk / brightness.size:.1f}%)")
    print(f"Brillo max: {brightness.max():.4f}, promedio disco: "
          f"{brightness[brightness > 0].mean():.4f}")


if __name__ == "__main__":
    main()
