"""
Integracion de geodesicas nulas en el espacio-tiempo de Schwarzschild.

Resuelve la ecuacion de orbita del foton:
    u'' + u = (3/2) * rs * u^2

donde u = 1/r y las primas denotan d/dphi.

El termino de correccion relativista (3/2)*rs*u^2 causa la curvatura de la luz.
Sin el, los fotones viajarian en linea recta (limite newtoniano).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from src.constants import RS, B_CRIT


class RayFate(Enum):
    """Destino de un rayo trazado."""
    CAPTURED = "captured"   # cayo al agujero negro (r <= rs)
    ESCAPED = "escaped"     # escapo a r grande
    ORBITING = "orbiting"   # sigue orbitando cuando la integracion termino


@dataclass
class GeodesicResult:
    """Resultado de integrar una geodesica de foton."""
    phi: NDArray[np.float64]       # arreglo de angulo azimutal
    u: NDArray[np.float64]         # arreglo de 1/r
    r: NDArray[np.float64]         # arreglo de coordenada radial
    x: NDArray[np.float64]         # cartesiana x = r*cos(phi)
    y: NDArray[np.float64]         # cartesiana y = r*sin(phi)
    b: float                       # parametro de impacto
    fate: RayFate                  # que le paso al rayo
    n_orbits: float                # numero de orbitas completadas


def geodesic_ode(phi: float, y: NDArray[np.float64], rs: float) -> list[float]:
    """Lado derecho del sistema ODE de la orbita del foton.

    Sistema:  y1 = u,  y2 = du/dphi
        dy1/dphi = y2
        dy2/dphi = -y1 + (3/2) * rs * y1^2
    """
    u, du = y
    return [du, -u + 1.5 * rs * u**2]


def initial_conditions(
    r_cam: float,
    alpha: float,
) -> tuple[float, float, float]:
    """Calcula las condiciones iniciales para un rayo desde la camara.

    Args:
        r_cam: Distancia radial de la camara al centro del agujero negro.
        alpha: Desviacion angular del rayo respecto al eje camara-centro (radianes).

    Returns:
        (b, u0, du0): parametro de impacto, u=1/r inicial, du/dphi inicial.
    """
    b = r_cam * np.sin(alpha)
    u0 = 1.0 / r_cam
    du0 = -np.cos(alpha) / (r_cam * np.sin(alpha))  # -cot(alpha) / r_cam
    return b, u0, du0


def trace_geodesic(
    b: float,
    r_cam: float,
    phi_max: float = 40.0,
    rs: float = RS,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int = 2000,
    dense_output: bool = True,
) -> GeodesicResult:
    """Integra una geodesica de foton dado el parametro de impacto b.

    Usa la ecuacion de orbita u'' + u = (3/2)*rs*u^2 integrada sobre
    el angulo azimutal phi, partiendo desde la posicion de la camara.

    Args:
        b: Parametro de impacto del rayo.
        r_cam: Distancia radial de la camara.
        phi_max: Angulo phi maximo para integrar (radianes).
        rs: Radio de Schwarzschild.
        rtol: Tolerancia relativa para el solver ODE.
        atol: Tolerancia absoluta para el solver ODE.
        max_steps: Numero maximo de pasos de integracion.
        dense_output: Si se solicita salida densa del solver.

    Returns:
        GeodesicResult con datos de la trayectoria y destino del rayo.
    """
    u0 = 1.0 / r_cam
    # De la ecuacion de energia: (du/dphi)^2 = 1/b^2 - u^2 + rs*u^3
    # Signo positivo: el rayo se acerca al agujero negro (u crece, r decrece)
    val = 1.0 / b**2 - u0**2 + rs * u0**3
    if val < 0:
        val = 0.0
    du0 = np.sqrt(val)

    y0 = [u0, du0]

    # Evento: rayo capturado cuando u >= 1/rs (r <= rs)
    def event_captured(phi: float, y: NDArray, rs: float) -> float:
        return 1.0 / rs - y[0]  # negativo cuando u > 1/rs
    event_captured.terminal = True
    event_captured.direction = -1

    # Evento: rayo escapa cuando r > 2*r_cam (u < 1/(2*r_cam))
    def event_escaped(phi: float, y: NDArray, rs: float) -> float:
        return y[0] - 0.5 / r_cam  # negativo cuando u < umbral
    event_escaped.terminal = True
    event_escaped.direction = -1

    phi_span = (0.0, phi_max)
    n_eval = max(500, max_steps)
    phi_eval = np.linspace(0.0, phi_max, n_eval)

    sol = solve_ivp(
        geodesic_ode,
        phi_span,
        y0,
        args=(rs,),
        method="RK45",
        events=[event_captured, event_escaped],
        t_eval=phi_eval,
        rtol=rtol,
        atol=atol,
        max_step=phi_max / max_steps,
        dense_output=dense_output,
    )

    phi = sol.t
    u = sol.y[0]

    # Recortar u para evitar r negativo
    u = np.clip(u, 1e-12, None)
    r = 1.0 / u

    x = r * np.cos(phi)
    y_coord = r * np.sin(phi)

    # Determinar destino
    if len(sol.t_events[0]) > 0:
        fate = RayFate.CAPTURED
    elif len(sol.t_events[1]) > 0:
        fate = RayFate.ESCAPED
    else:
        fate = RayFate.ORBITING

    n_orbits = phi[-1] / (2.0 * np.pi) if len(phi) > 0 else 0.0

    return GeodesicResult(
        phi=phi,
        u=u,
        r=r,
        x=x,
        y=y_coord,
        b=b,
        fate=fate,
        n_orbits=n_orbits,
    )


def trace_rays(
    impact_params: NDArray[np.float64],
    r_cam: float,
    **kwargs,
) -> list[GeodesicResult]:
    """Traza multiples rayos con diferentes parametros de impacto.

    Args:
        impact_params: Arreglo de parametros de impacto b.
        r_cam: Distancia radial de la camara.
        **kwargs: Se pasan a trace_geodesic.

    Returns:
        Lista de GeodesicResult, uno por rayo.
    """
    return [trace_geodesic(b, r_cam, **kwargs) for b in impact_params]


def classical_deflection(b: float, rs: float = RS) -> float:
    """Angulo de deflexion clasico de Einstein en campo debil.

    Valido para b >> b_crit:  dphi ~ 2*rs / b = 4M/b

    Args:
        b: Parametro de impacto.
        rs: Radio de Schwarzschild.

    Returns:
        Angulo de deflexion esperado en radianes.
    """
    return 2.0 * rs / b
