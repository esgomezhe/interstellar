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

from src.constants import RS


@dataclass
class RayInfo:
    """Todo lo necesario para trazar un rayo y detectar cruces con el disco."""
    b: float                         # parametro de impacto
    e1: NDArray[np.float64]          # base del plano orbital: radial (hacia la camara)
    e2: NDArray[np.float64]          # base del plano orbital: perpendicular
    psi_eq_base: float               # angulo base de cruce ecuatorial en el plano orbital


@dataclass
class Camera:
    """
    Camara perspectiva mirando al centro del agujero negro.

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
    phi: float = 0.0

    @property
    def position(self) -> NDArray[np.float64]:
        """Posicion de la camara en coordenadas cartesianas."""
        st, ct = np.sin(self.theta), np.cos(self.theta)
        sp, cp = np.sin(self.phi), np.cos(self.phi)
        return self.r * np.array([st * cp, st * sp, ct])

    @property
    def _basis(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Base ortonormal local en la camara.

        Retorna (e_r, e_theta, e_phi) en coordenadas cartesianas.
        """
        st, ct = np.sin(self.theta), np.cos(self.theta)
        sp, cp = np.sin(self.phi), np.cos(self.phi)
        e_r = np.array([st * cp, st * sp, ct])
        e_theta = np.array([ct * cp, ct * sp, -st])
        e_phi = np.array([-sp, cp, 0.0])
        return e_r, e_theta, e_phi

    def pixel_to_ray(self, i: int, j: int) -> RayInfo:
        """
        Convierte coordenadas de pixel a informacion del rayo.

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

        # Parametro de impacto para observador ESTATICO a distancia finita
        # (Synge 1966): b = r * sin(psi) / sqrt(1 - rs/r).
        # Sin el factor de lapse la escala angular de la imagen queda ~3.5%
        # mas grande a r=30M.
        r_hat = e_r
        cross = np.cross(r_hat, d)
        sin_psi = np.linalg.norm(cross)
        b = self.r * sin_psi / np.sqrt(1.0 - RS / self.r)

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
        inv_lapse = 1.0 / np.sqrt(1.0 - RS / self.r)

        for j in range(self.height):
            for i in range(self.width):
                alpha = (i - self.width / 2.0 + 0.5) * pixel_size
                beta = (self.height / 2.0 - j - 0.5) * pixel_size
                d = -e_r + np.tan(alpha) * e_phi + np.tan(beta) * (-e_theta)
                d /= np.linalg.norm(d)
                cross = np.cross(e_r, d)
                b_array[j, i] = self.r * np.linalg.norm(cross) * inv_lapse
        return b_array

    # -----------------------------------------------------------------
    # Kerr: mapeo de pixel a constantes de Carter (xi, eta)
    # -----------------------------------------------------------------

    def pixel_to_carter(self, i: int, j: int, a: float = 0.0) -> tuple[float, float, float, float]:
        """
        Convierte coordenadas de pixel a constantes de Carter para Kerr.

        Las coordenadas celestes de Bardeen (alpha_B, beta_B) estan en unidades
        de M (longitud), no radianes. Para un observador a distancia r_obs:
            alpha_B = r_obs * tan(alpha_pixel)
            beta_B = r_obs * tan(beta_pixel)

        Luego (Bardeen 1973):
            xi = -alpha_B * sin(theta_obs)
            eta = beta_B^2 + a^2*cos^2(theta_obs) - xi^2*cot^2(theta_obs)

        Args:
            i: indice de columna
            j: indice de fila
            a: parametro de spin

        Returns:
            (xi, eta, alpha_B, beta_B): constantes de Carter y coordenadas Bardeen.
        """
        fov_rad = np.radians(self.fov)
        pixel_size = fov_rad / self.height

        alpha_pix = (i - self.width / 2.0 + 0.5) * pixel_size
        beta_pix = (self.height / 2.0 - j - 0.5) * pixel_size

        # Coordenadas Bardeen en unidades de M, con el factor de lapse del
        # observador estatico a distancia finita (correccion dominante
        # O(M/r); los terminos de frame-dragging son O(aM^2/r^3), ~1e-5
        # a r=30M, y se desprecian)
        inv_lapse = 1.0 / np.sqrt(1.0 - RS / self.r)
        alpha_B = self.r * np.tan(alpha_pix) * inv_lapse
        beta_B = self.r * np.tan(beta_pix) * inv_lapse

        sin_t = np.sin(self.theta)
        cos_t = np.cos(self.theta)

        xi = -alpha_B * sin_t
        cot_t = cos_t / sin_t if abs(sin_t) > 1e-12 else 0.0
        eta = beta_B * beta_B - a * a * cos_t * cos_t + xi * xi * cot_t * cot_t

        return xi, eta, alpha_B, beta_B

    def all_carter_params(self, a: float = 0.0
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Retorna arreglos 2D de (xi, eta, beta_B) para cada pixel, forma (height, width)."""
        xi_arr = np.zeros((self.height, self.width))
        eta_arr = np.zeros((self.height, self.width))
        beta_B_arr = np.zeros((self.height, self.width))
        fov_rad = np.radians(self.fov)
        pixel_size = fov_rad / self.height
        sin_t = np.sin(self.theta)
        cos_t = np.cos(self.theta)
        cot_t = cos_t / sin_t if abs(sin_t) > 1e-12 else 0.0
        a2cos2 = a * a * cos_t * cos_t
        cot2 = cot_t * cot_t
        # Lapse del observador estatico a distancia finita (ver pixel_to_carter)
        inv_lapse = 1.0 / np.sqrt(1.0 - RS / self.r)

        for j in range(self.height):
            beta_pix = (self.height / 2.0 - j - 0.5) * pixel_size
            beta_B = self.r * np.tan(beta_pix) * inv_lapse
            beta_B2 = beta_B * beta_B
            for i in range(self.width):
                alpha_pix = (i - self.width / 2.0 + 0.5) * pixel_size
                alpha_B = self.r * np.tan(alpha_pix) * inv_lapse
                xi = -alpha_B * sin_t
                xi_arr[j, i] = xi
                eta_arr[j, i] = beta_B2 - a2cos2 + xi * xi * cot2
                beta_B_arr[j, i] = beta_B

        return xi_arr, eta_arr, beta_B_arr
