"""
Ray tracer: traza geodesicas para cada pixel y clasifica el resultado.

Para cada pixel:
  1. Obtener info del rayo desde la camara (parametro de impacto + plano orbital)
  2. Integrar la geodesica usando la ecuacion de orbita
  3. Detectar cruces con el plano ecuatorial (intersecciones con el disco)
  4. Clasificar: agujero negro / disco / cielo
  5. Aplicar redshift gravitacional, Doppler y beaming
  6. Convertir a color RGB de cuerpo negro
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from src.camera import Camera, RayInfo
from src.colormap import disk_color
from src.constants import RS
from src.disk import AccretionDisk
from src.effects import (
    cunningham_g_factor,
    observed_intensity,
    photon_angular_momentum,
)
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
    color: tuple[float, float, float] = (0.0, 0.0, 0.0)  # RGB [0, 1]
    g_d: float = 1.0             # factor combinado gravitacional * Doppler


def trace_pixel(
    ray: RayInfo,
    r_cam: float,
    disk: AccretionDisk,
    cam_position: NDArray[np.float64],
    rs: float = RS,
    phi_max: float = 50.0,
    enable_effects: bool = True,
) -> PixelResult:
    """Traza el rayo de un pixel a traves del espacio-tiempo curvo.

    Args:
        ray: RayInfo desde la camara (parametro de impacto + plano orbital).
        r_cam: Distancia de la camara.
        disk: Modelo del disco de acrecion.
        cam_position: Posicion 3D de la camara (para calculo Doppler).
        rs: Radio de Schwarzschild.
        phi_max: Angulo maximo de integracion.
        enable_effects: Si se aplican redshift, Doppler y colormap.

    Returns:
        PixelResult con clasificacion, brillo y color.
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
        r_hit, psi_hit = crossings[0]
        base_emission = disk.emission(r_hit)

        if enable_effects:
            # g-factor exacto de Cunningham via L_z/E conservada del foton
            lam = photon_angular_momentum(ray.b, ray.e1, ray.e2)
            g_d = cunningham_g_factor(r_hit, lam)
            # Intensidad con beaming relativista
            intensity = observed_intensity(base_emission, g_d)

            # Contribucion de cruces adicionales (imagenes secundarias)
            for r_extra, psi_extra in crossings[1:]:
                g_d_extra = cunningham_g_factor(r_extra, lam)
                intensity += observed_intensity(
                    disk.emission(r_extra), g_d_extra,
                ) * 0.5

            intensity = min(intensity, 1.0)
            color = disk_color(intensity, g_d)
        else:
            # Modo sin efectos: escala de grises como en Fase 2
            intensity = base_emission
            for r_extra, _ in crossings[1:]:
                intensity += disk.emission(r_extra) * 0.5
            intensity = min(intensity, 1.0)
            g_d = 1.0
            color = (intensity, intensity, intensity)

        return PixelResult(
            pixel_type=PixelType.DISK,
            intensity=intensity,
            disk_radius=r_hit,
            n_crossings=len(crossings),
            color=color,
            g_d=g_d,
        )

    if result.fate == RayFate.CAPTURED:
        return PixelResult(pixel_type=PixelType.BLACK_HOLE)

    return PixelResult(pixel_type=PixelType.SKY)


def render_frame(
    camera: Camera,
    disk: AccretionDisk | None = None,
    phi_max: float = 50.0,
    show_progress: bool = True,
    enable_effects: bool = True,
) -> NDArray[np.float64]:
    """Renderiza un frame completo trazando cada pixel.

    Args:
        camera: Camara con posicion, resolucion, FOV.
        disk: Modelo del disco de acrecion (por defecto: disco estandar).
        phi_max: Angulo maximo de integracion por rayo.
        show_progress: Mostrar barra de progreso con tqdm.
        enable_effects: Si se aplican redshift, Doppler y colormap.

    Returns:
        Arreglo 3D de colores RGB, forma (height, width, 3), valores en [0, 1].
    """
    if disk is None:
        disk = AccretionDisk()

    image = np.zeros((camera.height, camera.width, 3))
    cam_pos = camera.position
    total_pixels = camera.height * camera.width

    pixel_iter = range(total_pixels)
    if show_progress:
        pixel_iter = tqdm(pixel_iter, desc="Renderizando", unit="px")

    for idx in pixel_iter:
        j = idx // camera.width
        i = idx % camera.width
        ray = camera.pixel_to_ray(i, j)
        result = trace_pixel(
            ray, camera.r, disk, cam_pos,
            phi_max=phi_max, enable_effects=enable_effects,
        )
        image[j, i] = result.color

    return image
