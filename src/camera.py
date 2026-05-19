"""
Modelo de camara para el ray tracer de agujero negro.

Mapea cada pixel (i, j) a una direccion de rayo en 3D, luego calcula el
parametro de impacto y el plano orbital necesarios para el integrador de geodesicas.

Convencion de coordenadas:
  - Esfericas (r, theta, phi): theta desde el polo +z, phi en el plano x-y.
  - Camara en (r_cam, theta_cam, 0). theta_cam < pi/2 significa arriba del ecuador.
  - La camara mira hacia el origen.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class RayInfo:
    """Todo lo necesario para trazar un rayo y detectar cruces con el disco."""
    b: float                         # parametro de impacto
    e1: NDArray[np.float64]          # base del plano orbital: radial (hacia la camara)
    e2: NDArray[np.float64]          # base del plano orbital: perpendicular
    psi_eq_base: float               # angulo base de cruce ecuatorial en el plano orbital


@dataclass
class Camera:
    """Camara perspectiva mirando al centro del agujero negro.

    Args:
        r: Distancia radial desde el centro (en unidades de M).
        theta: Angulo polar desde el eje +z (radianes). ~80 deg = ligeramente arriba del ecuador.
        width: Ancho de la imagen en pixeles.
        height: Alto de la imagen en pixeles.
        fov: Campo de vision vertical en grados.
    """
    r: float
    theta: float
    width: int
    height: int
    fov: float = 16.0

    @property
    def position(self) -> NDArray[np.float64]:
        """Posicion de la camara en coordenadas cartesianas."""
        return self.r * np.array([
            np.sin(self.theta),
            0.0,
            np.cos(self.theta),
        ])

    @property
    def _basis(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Base ortonormal local en la camara.

        Retorna (e_r, e_theta, e_phi) en coordenadas cartesianas.
        """
        st, ct = np.sin(self.theta), np.cos(self.theta)
        e_r = np.array([st, 0.0, ct])
        e_theta = np.array([ct, 0.0, -st])
        e_phi = np.array([0.0, 1.0, 0.0])
        return e_r, e_theta, e_phi

    def pixel_to_ray(self, i: int, j: int) -> RayInfo:
        """Convierte coordenadas de pixel a informacion del rayo.

        Args:
            i: Indice de columna (0 = izquierda).
            j: Indice de fila (0 = arriba).

        Returns:
            RayInfo con parametro de impacto y datos del plano orbital.
        """
        e_r, e_theta, e_phi = self._basis

        fov_rad = np.radians(self.fov)
        pixel_size = fov_rad / self.height

        # Desplazamientos angulares desde el centro de la imagen
        alpha = (i - self.width / 2.0 + 0.5) * pixel_size    # horizontal
        beta = (self.height / 2.0 - j - 0.5) * pixel_size    # vertical (arriba = positivo)

        # Direccion del rayo: hacia el origen + desplazamientos angulares
        d = -e_r + np.tan(alpha) * e_phi + np.tan(beta) * (-e_theta)
        d /= np.linalg.norm(d)

        # Parametro de impacto: b = r_cam * |r_hat x d_hat|
        r_hat = e_r
        cross = np.cross(r_hat, d)
        sin_psi = np.linalg.norm(cross)
        b = self.r * sin_psi

        # Base del plano orbital
        if sin_psi < 1e-12:
            # Rayo apuntando directo al centro: se elige plano arbitrario
            e1 = r_hat.copy()
            e2 = e_theta.copy()
            psi_eq_base = np.arctan2(-e1[2], e2[2]) if abs(e2[2]) > 1e-12 else 0.0
        else:
            n = cross / sin_psi  # normal al plano orbital
            e1 = r_hat.copy()
            e2 = np.cross(n, e1)
            e2 /= np.linalg.norm(e2)
            psi_eq_base = np.arctan2(-e1[2], e2[2]) if abs(e1[2]) + abs(e2[2]) > 1e-12 else 0.0

        return RayInfo(b=b, e1=e1, e2=e2, psi_eq_base=psi_eq_base)

    def all_rays(self) -> list[list[RayInfo]]:
        """Genera RayInfo para cada pixel. Retorna lista anidada [fila][columna]."""
        return [
            [self.pixel_to_ray(i, j) for i in range(self.width)]
            for j in range(self.height)
        ]

    def all_impact_params(self) -> NDArray[np.float64]:
        """Retorna un arreglo 2D de parametros de impacto, forma (height, width)."""
        b_array = np.zeros((self.height, self.width))
        fov_rad = np.radians(self.fov)
        pixel_size = fov_rad / self.height
        e_r, e_theta, e_phi = self._basis

        for j in range(self.height):
            for i in range(self.width):
                alpha = (i - self.width / 2.0 + 0.5) * pixel_size
                beta = (self.height / 2.0 - j - 0.5) * pixel_size
                d = -e_r + np.tan(alpha) * e_phi + np.tan(beta) * (-e_theta)
                d /= np.linalg.norm(d)
                cross = np.cross(e_r, d)
                b_array[j, i] = self.r * np.linalg.norm(cross)
        return b_array
