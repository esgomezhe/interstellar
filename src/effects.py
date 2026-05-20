"""
Efectos relativistas visuales para el disco de acrecion.

Implementa:
  - Redshift gravitacional: g = sqrt(1 - rs/r)
  - Doppler kepleriano: asimetria de brillo por rotacion del disco
  - Beaming relativista: la intensidad observada escala como (g * D)^4
"""

import numpy as np
from numpy.typing import NDArray

from src.constants import RS, M


def gravitational_redshift(r: float, rs: float = RS) -> float:
    """Factor de redshift gravitacional.

    g = sqrt(1 - rs/r). Valores cercanos al horizonte -> g ~ 0 (muy corrido al rojo).
    Valores lejanos -> g ~ 1 (sin correccion).

    Args:
        r: Radio de emision.
        rs: Radio de Schwarzschild.

    Returns:
        Factor g en [0, 1].
    """
    if r <= rs:
        return 0.0
    return np.sqrt(1.0 - rs / r)


def keplerian_velocity(r: float) -> float:
    """Velocidad orbital kepleriana del material del disco.

    v_phi = sqrt(M / r) en unidades geometrizadas (c = 1).
    Para r = 6M (ISCO): v ~ 0.408c. Para r = 20M: v ~ 0.224c.

    Args:
        r: Radio en el disco.

    Returns:
        Velocidad tangencial (fraccion de c).
    """
    return np.sqrt(M / r)


def doppler_factor(
    r_hit: float,
    psi_hit: float,
    e1: NDArray[np.float64],
    e2: NDArray[np.float64],
    cam_position: NDArray[np.float64],
) -> float:
    """Factor Doppler relativista en el punto de interseccion con el disco.

    Calcula D = 1 / (gamma * (1 - v . n_hat)) donde:
      - v es la velocidad kepleriana del material del disco
      - n_hat es la direccion del punto de emision hacia el observador
      - gamma es el factor de Lorentz

    D > 1: lado que se acerca (blueshift, mas brillante)
    D < 1: lado que se aleja (redshift, mas tenue)

    Args:
        r_hit: Radio de interseccion con el disco.
        psi_hit: Angulo en el plano orbital donde se da el cruce.
        e1: Vector base del plano orbital (radial).
        e2: Vector base del plano orbital (perpendicular).
        cam_position: Posicion 3D de la camara.

    Returns:
        Factor Doppler D.
    """
    # Posicion 3D del punto de impacto en el plano ecuatorial
    hit_pos = r_hit * (np.cos(psi_hit) * e1 + np.sin(psi_hit) * e2)

    # Direccion desde el punto de emision hacia el observador
    to_observer = cam_position - hit_pos
    dist = np.linalg.norm(to_observer)
    if dist < 1e-12:
        return 1.0
    n_hat = to_observer / dist

    # Velocidad kepleriana: v = sqrt(M/r) en direccion azimutal
    # En el plano ecuatorial (z=0), la direccion azimutal en (x, y) es (-y, x, 0) / r
    x, y = hit_pos[0], hit_pos[1]
    r_xy = np.sqrt(x**2 + y**2)
    if r_xy < 1e-12:
        return 1.0

    # Direccion azimutal (rotacion prograba: sentido antihorario visto desde +z)
    phi_hat = np.array([-y, x, 0.0]) / r_xy

    v_mag = keplerian_velocity(r_hit)
    v_disk = v_mag * phi_hat

    # Factor Doppler relativista: D = 1 / (gamma * (1 - v . n_hat))
    v_dot_n = np.dot(v_disk, n_hat)
    gamma = 1.0 / np.sqrt(1.0 - v_mag**2)

    D = 1.0 / (gamma * (1.0 - v_dot_n))
    return D


def total_shift_factor(
    r_hit: float,
    psi_hit: float,
    e1: NDArray[np.float64],
    e2: NDArray[np.float64],
    cam_position: NDArray[np.float64],
    rs: float = RS,
) -> float:
    """Factor combinado de frecuencia: gravitacional * Doppler.

    nu_obs / nu_emit = g * D

    La intensidad observada de radiacion termica escala como (g * D)^4.

    Args:
        r_hit: Radio de interseccion con el disco.
        psi_hit: Angulo en el plano orbital del cruce.
        e1, e2: Base del plano orbital.
        cam_position: Posicion 3D de la camara.
        rs: Radio de Schwarzschild.

    Returns:
        Factor combinado g * D.
    """
    g = gravitational_redshift(r_hit, rs)
    D = doppler_factor(r_hit, psi_hit, e1, e2, cam_position)
    return g * D


def observed_intensity(
    emission: float,
    g_d: float,
    beaming_power: float = 3.0,
) -> float:
    """Intensidad observada con beaming relativista.

    I_obs = I_emit * (g * D)^n donde n=3 para emision monocromatica
    (n=4 para radiacion termica integrada en frecuencia).

    Args:
        emission: Intensidad emitida por el disco.
        g_d: Factor combinado g * D.
        beaming_power: Exponente del beaming (4 para termico).

    Returns:
        Intensidad observada.
    """
    return emission * g_d**beaming_power
