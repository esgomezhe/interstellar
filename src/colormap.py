"""
Mapeo de intensidad y frecuencia a color RGB tipo cuerpo negro.

Convierte la intensidad del disco y el factor de frecuencia observada
a un color realista:
  - Lado que se acerca (blueshifted): blanco-azulado, mas brillante
  - Centro del disco: naranja-amarillo
  - Lado que se aleja (redshifted): rojo-oscuro, mas tenue
"""

import numpy as np
from numpy.typing import NDArray


# Temperatura base del disco en el borde interno (unidades arbitrarias)
# 2200K da rojo-naranja de base, blueshift lo lleva a dorado-blanco
T_BASE: float = 2200.0


def frequency_to_temperature(base_temp: float, g_d: float) -> float:
    """Temperatura observada dado el shift de frecuencia.

    T_obs = T_emit * g * D (la temperatura escala linealmente con la frecuencia).

    Args:
        base_temp: Temperatura de emision.
        g_d: Factor combinado gravitacional * Doppler.

    Returns:
        Temperatura observada en K.
    """
    return base_temp * g_d


def blackbody_to_rgb(temperature: float) -> tuple[float, float, float]:
    """Convierte temperatura de cuerpo negro a color RGB.

    Usa la aproximacion de Tanner Helland para mapear temperatura (K)
    a valores RGB normalizados [0, 1]. Valida para 1000K - 40000K.

    Args:
        temperature: Temperatura en Kelvin.

    Returns:
        Tupla (r, g, b) con valores en [0, 1].
    """
    temp = np.clip(temperature, 1000.0, 40000.0) / 100.0

    # Componente roja
    if temp <= 66.0:
        red = 1.0
    else:
        red = 329.698727446 * (temp - 60.0) ** (-0.1332047592) / 255.0

    # Componente verde
    if temp <= 66.0:
        green = (99.4708025861 * np.log(temp) - 161.1195681661) / 255.0
    else:
        green = 288.1221695283 * (temp - 60.0) ** (-0.0755148492) / 255.0

    # Componente azul
    if temp >= 66.0:
        blue = 1.0
    elif temp <= 19.0:
        blue = 0.0
    else:
        blue = (138.5177312231 * np.log(temp - 10.0) - 305.0447927307) / 255.0

    return (
        float(np.clip(red, 0.0, 1.0)),
        float(np.clip(green, 0.0, 1.0)),
        float(np.clip(blue, 0.0, 1.0)),
    )


def disk_color(
    intensity: float,
    g_d: float,
    base_temp: float = T_BASE,
) -> tuple[float, float, float]:
    """Calcula el color final de un pixel del disco.

    Combina:
      1. La temperatura observada (shift de frecuencia -> color del cuerpo negro)
      2. La intensidad observada (beaming -> brillo)

    Args:
        intensity: Intensidad observada (ya con beaming aplicado).
        g_d: Factor combinado g * D (para determinar el color).
        base_temp: Temperatura base del disco.

    Returns:
        Tupla (r, g, b) con valores en [0, 1].
    """
    if intensity <= 0:
        return (0.0, 0.0, 0.0)

    # Temperatura observada determina el tono del color
    temp_obs = frequency_to_temperature(base_temp, g_d)
    r, g, b = blackbody_to_rgb(temp_obs)

    # Escalar por intensidad con compresion logaritmica para rango dinamico
    brightness = np.clip(intensity, 0.0, 1.0) ** 0.5

    return (
        float(np.clip(r * brightness, 0.0, 1.0)),
        float(np.clip(g * brightness, 0.0, 1.0)),
        float(np.clip(b * brightness, 0.0, 1.0)),
    )
