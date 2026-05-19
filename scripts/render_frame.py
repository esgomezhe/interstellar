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

    image = render_frame(camera, disk, phi_max=50.0)

    # Correccion gamma para mejor contraste visual
    gamma = 0.5
    image_display = np.power(np.clip(image, 0, 1), gamma)

    # Guardar imagen en escala de grises
    out_path = Path("outputs/frames/frame_phase2.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7.5), facecolor="black")
    ax.imshow(image_display, cmap="inferno", origin="upper",
              vmin=0, vmax=1, aspect="equal")
    ax.set_axis_off()
    ax.set_title(
        "Agujero Negro de Schwarzschild con Disco de Acrecion\n"
        f"({camera.width}x{camera.height}, theta={np.degrees(camera.theta):.0f} grados)",
        color="white", fontsize=13, pad=10,
    )
    plt.tight_layout(pad=0.5)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    print(f"Guardado: {out_path}")

    # Guardar arreglo numpy crudo para fases posteriores
    np.save("outputs/frames/frame_phase2_raw.npy", image)
    print("Datos crudos guardados: outputs/frames/frame_phase2_raw.npy")

    # Estadisticas
    n_black = np.sum(image == 0)
    n_disk = np.sum(image > 0)
    print(f"\nEstadisticas: {n_black} pixeles agujero negro/cielo, {n_disk} disco "
          f"({100 * n_disk / image.size:.1f}%)")


if __name__ == "__main__":
    main()
