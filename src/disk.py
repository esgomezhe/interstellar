"""
Modelo de disco de acrecion para agujero negro de Schwarzschild.

Disco delgado en el plano ecuatorial (theta = pi/2) entre r_ISCO y r_outer.
Perfil de emision siguiendo la ley de potencia r^(-3/4) de Shakura-Sunyaev.
"""

import numpy as np
from numpy.typing import NDArray

from src.constants import R_ISCO, R_DISK_OUTER


class AccretionDisk:
    """Disco de acrecion geometricamente delgado en el plano ecuatorial.

    Args:
        r_inner: Radio del borde interno (por defecto: ISCO = 6M).
        r_outer: Radio del borde externo (por defecto: 20M).
    """

    def __init__(
        self,
        r_inner: float = R_ISCO,
        r_outer: float = R_DISK_OUTER,
    ) -> None:
        self.r_inner = r_inner
        self.r_outer = r_outer

    def contains(self, r: float) -> bool:
        """Verifica si el radio r esta dentro del disco."""
        return self.r_inner <= r <= self.r_outer

    def emission(self, r: float) -> float:
        """Brillo del disco en el radio r (perfil Shakura-Sunyaev).

        I(r) ~ r^(-3/4), normalizado para que I(r_inner) ~ 1.
        Retorna 0 si esta fuera del disco.
        """
        if not self.contains(r):
            return 0.0
        return (self.r_inner / r) ** 0.75

    def find_equatorial_crossings(
        self,
        psi: NDArray[np.float64],
        r: NDArray[np.float64],
        e1: NDArray[np.float64],
        e2: NDArray[np.float64],
    ) -> list[tuple[float, float]]:
        """Encuentra donde una geodesica cruza el plano ecuatorial.

        La coordenada z en 3D a lo largo de la geodesica es:
            z(psi) = r(psi) * [cos(psi)*e1_z + sin(psi)*e2_z]

        Los cruces ecuatoriales ocurren donde z = 0, es decir donde
            cos(psi)*e1_z + sin(psi)*e2_z = 0

        Detectamos cambios de signo en z a lo largo de la trayectoria
        e interpolamos para encontrar el radio y angulo de cruce.

        Args:
            psi: Arreglo de angulos a lo largo de la geodesica.
            r: Arreglo de radios (1/u).
            e1: Vector base del plano orbital (radial).
            e2: Vector base del plano orbital (perpendicular).

        Returns:
            Lista de tuplas (r_cruce, psi_cruce) para cruces dentro del disco.
        """
        z = r * (np.cos(psi) * e1[2] + np.sin(psi) * e2[2])

        crossings = []
        for k in range(len(z) - 1):
            if z[k] * z[k + 1] < 0:
                # Interpolacion lineal para encontrar el punto de cruce
                frac = abs(z[k]) / (abs(z[k]) + abs(z[k + 1]))
                r_cross = r[k] + frac * (r[k + 1] - r[k])
                psi_cross = psi[k] + frac * (psi[k + 1] - psi[k])
                if self.contains(r_cross):
                    crossings.append((r_cross, psi_cross))
        return crossings
