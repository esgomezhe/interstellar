"""
Ray tracer: traza geodesicas para cada pixel y clasifica el resultado.

Para cada pixel:
  1. Obtener info del rayo desde la camara (parametro de impacto + plano orbital)
  2. Integrar la geodesica usando la ecuacion de orbita
  3. Detectar cruces con el plano ecuatorial (intersecciones con el disco)
  4. Clasificar: agujero negro / disco / cielo
  5. Calcular brillo si toco el disco
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from src.camera import Camera, RayInfo
from src.constants import RS, R_ISCO, R_DISK_OUTER
from src.disk import AccretionDisk
from src.geodesics import trace_geodesic, RayFate


class PixelType(Enum):
    """Lo que ve un pixel."""
    BLACK_HOLE = 0
    DISK = 1
    SKY = 2


@dataclass
class PixelResult:
    """Resultado para un solo pixel."""
    pixel_type: PixelType
    intensity: float = 0.0       # brillo [0, 1]
    disk_radius: float = 0.0     # radio de interseccion con el disco (si hay)
    n_crossings: int = 0         # cuantas veces el rayo cruzo el disco


def trace_pixel(
    ray: RayInfo,
    r_cam: float,
    disk: AccretionDisk,
    rs: float = RS,
    phi_max: float = 50.0,
) -> PixelResult:
    """Traza el rayo de un pixel a traves del espacio-tiempo curvo.

    Args:
        ray: RayInfo desde la camara (parametro de impacto + plano orbital).
        r_cam: Distancia de la camara.
        disk: Modelo del disco de acrecion.
        rs: Radio de Schwarzschild.
        phi_max: Angulo maximo de integracion.

    Returns:
        PixelResult con clasificacion y brillo.
    """
    if ray.b < 1e-6:
        return PixelResult(pixel_type=PixelType.BLACK_HOLE)

    result = trace_geodesic(ray.b, r_cam, phi_max=phi_max, rs=rs)

    # Detectar cruces con el disco usando geometria 3D
    crossings = disk.find_equatorial_crossings(
        result.phi, result.r, ray.e1, ray.e2,
    )

    if crossings:
        # Usar el PRIMER cruce (mas cercano a la camara)
        r_hit = crossings[0]
        intensity = disk.emission(r_hit)
        # Multiples cruces agregan brillo (la luz envuelve al agujero)
        for r_extra in crossings[1:]:
            intensity += disk.emission(r_extra) * 0.5  # imagenes secundarias mas tenues
        intensity = min(intensity, 1.0)
        return PixelResult(
            pixel_type=PixelType.DISK,
            intensity=intensity,
            disk_radius=r_hit,
            n_crossings=len(crossings),
        )

    if result.fate == RayFate.CAPTURED:
        return PixelResult(pixel_type=PixelType.BLACK_HOLE)

    return PixelResult(pixel_type=PixelType.SKY)


def render_frame(
    camera: Camera,
    disk: AccretionDisk | None = None,
    phi_max: float = 50.0,
    show_progress: bool = True,
) -> NDArray[np.float64]:
    """Renderiza un frame completo trazando cada pixel.

    Args:
        camera: Camara con posicion, resolucion, FOV.
        disk: Modelo del disco de acrecion (por defecto: disco estandar).
        phi_max: Angulo maximo de integracion por rayo.
        show_progress: Mostrar barra de progreso con tqdm.

    Returns:
        Arreglo 2D de intensidades, forma (height, width), valores en [0, 1].
    """
    if disk is None:
        disk = AccretionDisk()

    image = np.zeros((camera.height, camera.width))
    total_pixels = camera.height * camera.width

    pixel_iter = range(total_pixels)
    if show_progress:
        pixel_iter = tqdm(pixel_iter, desc="Renderizando", unit="px")

    for idx in pixel_iter:
        j = idx // camera.width
        i = idx % camera.width
        ray = camera.pixel_to_ray(i, j)
        result = trace_pixel(ray, camera.r, disk, phi_max=phi_max)
        image[j, i] = result.intensity

    return image
