"""
Efectos relativistas visuales para el disco de acrecion.

Implementa:
  - g-factor exacto de Cunningham (1975): redshift gravitacional + Doppler
    en un solo factor, usando la constante conservada L_z/E del foton
  - Redshift gravitacional estatico y Doppler de campo debil (referencias
    pedagogicas; el render usa el g-factor exacto)
  - Beaming relativista: la intensidad observada escala como g^n
"""

import numpy as np
from numpy.typing import NDArray

from src.constants import RS, M


def cunningham_g_factor(
    r_hit: float,
    lam: float,
    a: float = 0.0,
    m: float = M,
) -> float:
    """Factor g = nu_obs/nu_em exacto para emisor en orbita Kepleriana.

    g = sqrt(1 - 3M/r + 2a*sqrt(M)/r^(3/2))
        / [(1 + a*sqrt(M)/r^(3/2)) * (1 - Omega*lam)]

    El numerador junto con (1 + a*sqrt(M)/r^(3/2)) es 1/u^t del emisor
    (Bardeen-Press-Teukolsky 1972) e incluye el redshift gravitacional Y el
    Doppler transversal. (1 - Omega*lam) es el Doppler azimutal, donde
    lam = L_z/E es una constante de movimiento del foton: no hace falta
    conocer la direccion local de emision (que en espacio-tiempo curvo NO
    es la linea recta hacia la camara).

    Con a=0: g = sqrt(1 - 3M/r) / (1 - Omega*lam). En el ISCO (r=6M) y
    foton sin momento angular: g = sqrt(1/2) ~ 0.707.

    Args:
        r_hit: Radio de emision en el disco.
        lam: L_z/E del foton (xi de Bardeen en Kerr; -b*n_z en Schwarzschild,
             con n la normal al plano orbital del rayo).
        a: Spin del agujero negro.
        m: Masa.

    Returns:
        Factor g, acotado a [0.05, 5.0].
    """
    sqrt_m = np.sqrt(m)
    r32 = r_hit ** 1.5
    omega = sqrt_m / (r32 + a * sqrt_m)

    x = a * sqrt_m / r32
    big_b = 1.0 - 3.0 * m / r_hit + 2.0 * x
    if big_b < 1e-6:
        big_b = 1e-6

    denom = (1.0 + x) * (1.0 - omega * lam)
    if abs(denom) < 1e-6:
        denom = 1e-6 if denom >= 0.0 else -1e-6

    g = np.sqrt(big_b) / denom
    return float(np.clip(g, 0.05, 5.0))


def photon_angular_momentum(
    b: float,
    e1: NDArray[np.float64],
    e2: NDArray[np.float64],
) -> float:
    """L_z/E del foton a partir del parametro de impacto y el plano orbital.

    El vector momento angular del foton es perpendicular a su plano orbital:
    L = b*E*n con n = e1 x e2. Para el rayo trazado HACIA ATRAS desde la
    camara, el foton fisico viaja en sentido opuesto, asi que lam = -b*n_z.

    Args:
        b: Parametro de impacto del rayo.
        e1, e2: Base ortonormal del plano orbital (e2 = n x e1).

    Returns:
        lam = L_z/E del foton fisico.
    """
    n_z = e1[0] * e2[1] - e1[1] * e2[0]
    return -b * n_z


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
    """Factor combinado de frecuencia: gravitacional * Doppler (APROXIMADO).

    nu_obs / nu_emit ~ g * D

    NOTA: aproximacion de campo debil que asume que el foton viaja en linea
    recta hacia la camara y usa el redshift de un emisor ESTATICO. Para el
    render usar cunningham_g_factor, que es exacto para orbitas Keplerianas.

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
