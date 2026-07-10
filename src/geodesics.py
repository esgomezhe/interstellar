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

from src.constants import RS, B_CRIT, M as _M


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
    """
    Lado derecho del sistema ODE de la orbita del foton.

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
    """
    Calcula las condiciones iniciales para un rayo desde la camara.

    Args:
        r_cam: Distancia radial de la camara al centro del agujero negro.
        alpha: Desviacion angular del rayo respecto al eje camara-centro (radianes).

    Returns:
        (b, u0, du0): parametro de impacto, u=1/r inicial, du/dphi inicial.
    """
    # Observador ESTATICO a distancia finita: factor de lapse (Synge 1966)
    #   b = r_cam * sin(alpha) / sqrt(1 - rs/r_cam)
    lapse = np.sqrt(1.0 - RS / r_cam)
    b = r_cam * np.sin(alpha) / lapse
    u0 = 1.0 / r_cam
    # De la ecuacion de energia: (du/dphi)^2 = 1/b^2 - u^2 + rs*u^3
    # Signo POSITIVO: u crece porque el rayo va hacia el agujero
    # (consistente con trace_geodesic)
    val = 1.0 / b**2 - u0**2 + RS * u0**3
    du0 = np.sqrt(max(val, 0.0))
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
    """
    Integra una geodesica de foton dado el parametro de impacto b.

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
    """
    Traza multiples rayos con diferentes parametros de impacto.

    Args:
        impact_params: Arreglo de parametros de impacto b.
        r_cam: Distancia radial de la camara.
        **kwargs: Se pasan a trace_geodesic.

    Returns:
        Lista de GeodesicResult, uno por rayo.
    """
    return [trace_geodesic(b, r_cam, **kwargs) for b in impact_params]


def classical_deflection(b: float, rs: float = RS) -> float:
    """
    Angulo de deflexion clasico de Einstein en campo debil.

    Valido para b >> b_crit:  dphi ~ 2*rs / b = 4M/b

    Args:
        b: Parametro de impacto.
        rs: Radio de Schwarzschild.

    Returns:
        Angulo de deflexion esperado en radianes.
    """
    return 2.0 * rs / b


# =========================================================================
# Kerr: ecuaciones de Carter para geodesicas nulas
# =========================================================================

def kerr_potentials(
    r: float, theta: float, a: float, xi: float, eta: float, m: float = _M,
) -> tuple[float, float, float, float]:
    """
    Potenciales radial R(r) y polar Theta(theta) para geodesicas en Kerr.

    R(r) = (r² + a² - a*xi)² - Delta * [eta + (xi - a)²]
    Theta(theta) = eta + a²*cos²(theta) - xi²*cot²(theta)

    Returns:
        (R, Theta, Sigma, Delta)
    """
    sigma = r * r + a * a * np.cos(theta) ** 2
    delta = r * r - 2.0 * m * r + a * a
    P = r * r + a * a - a * xi
    R = P * P - delta * (eta + (xi - a) ** 2)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    if abs(sin_t) < 1e-12:
        Theta = eta + a * a * cos_t * cos_t
    else:
        Theta = eta + a * a * cos_t * cos_t - xi * xi * (cos_t / sin_t) ** 2
    return R, Theta, sigma, delta


def kerr_geodesic_rhs(
    lam: float,
    y: NDArray[np.float64],
    a: float,
    xi: float,
    eta: float,
    m: float,
) -> list[float]:
    """
    Lado derecho del sistema ODE de Carter para geodesicas nulas en Kerr.

    Variables: y = [r, theta, phi, t]
    Parametro independiente: lambda (parametro afin)

    Las ecuaciones de movimiento (Chandrasekhar 1983):
        Sigma * dr/dlam = ± sqrt(R(r))
        Sigma * dtheta/dlam = ± sqrt(Theta(theta))
        Sigma * dphi/dlam = -a + a*P/Delta + xi/sin²(theta)
        Sigma * dt/dlam = -a²*sin²(theta) + (r²+a²)*P/Delta

    Donde P = r² + a² - a*xi

    El signo de dr/dlam y dtheta/dlam se determina por la evolucion
    (turning points cuando R=0 o Theta=0).
    """
    r, theta, phi, t_coord = y

    sigma = r * r + a * a * np.cos(theta) ** 2
    sigma = max(sigma, 1e-6)
    delta = r * r - 2.0 * m * r + a * a
    if abs(delta) < 1e-6:
        delta = 1e-6
    P = r * r + a * a - a * xi

    # Potencial radial
    R_val = P * P - delta * (eta + (xi - a) ** 2)

    # Potencial polar
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    if abs(sin_t) < 0.01:
        sin_t = 0.01 if sin_t >= 0 else -0.01
    Theta_val = eta + a * a * cos_t * cos_t - xi * xi * (cos_t / sin_t) ** 2

    inv_sigma = 1.0 / sigma

    # dr/dlam (usamos la primera derivada de la forma cuadratica)
    # d(Sigma²*(dr/dlam)²)/dr = dR/dr =>
    # 2*Sigma*dr/dlam * d²r/dlam² = dR/dr - 2*r*(dr/dlam)² ... complejo
    # Mejor: usar la formulacion de segundo orden para evitar problemas de signo

    # Formulacion de segundo orden (Dexter & Agol 2009):
    # d²r/dlam² = (1/2Σ) * dR/dr - (r/Σ)*(dr/dlam)² + ...
    # Pero para SciPy es mas estable usar primera derivada con manejo de signos.

    # Metodo robusto: almacenar dr/dlam y dtheta/dlam como variables auxiliares
    # Usamos formulacion con 6 variables: [r, theta, phi, t, p_r, p_theta]
    # donde p_r = Sigma * dr/dlam, p_theta = Sigma * dtheta/dlam

    # NOTA: esta funcion usa la formulacion de primera derivada.
    # Para el integrador Numba usaremos la formulacion de segunda derivada.
    # Aqui se usa solo para validacion con SciPy.

    if R_val < 0.0:
        R_val = 0.0
    if Theta_val < 0.0:
        Theta_val = 0.0

    # Los signos se determinan externamente (ver trace_kerr_geodesic)
    # Placeholder: no se puede determinar sin estado adicional
    dr_dlam = np.sqrt(R_val) * inv_sigma
    dtheta_dlam = np.sqrt(Theta_val) * inv_sigma
    dphi_dlam = (-a + a * P / delta + xi / (sin_t * sin_t)) * inv_sigma
    dt_dlam = (a * (xi - a * sin_t * sin_t) + (r * r + a * a) * P / delta) * inv_sigma

    return [dr_dlam, dtheta_dlam, dphi_dlam, dt_dlam]


def kerr_geodesic_rhs_2nd_order(
    lam: float,
    y: NDArray[np.float64],
    a: float,
    xi: float,
    eta: float,
    m: float,
) -> list[float]:
    """
    Sistema de segundo orden para geodesicas de Kerr (6 variables).

    Variables: y = [r, theta, p_r, p_theta, phi, t]
    Donde p_r = Sigma * dr/dlam, p_theta = Sigma * dtheta/dlam.

    Las ecuaciones de evolucion de p_r y p_theta se obtienen de:
        p_r² = R(r)
        p_theta² = Theta(theta)

    Derivando: dp_r/dlam = (1/2) * dR/dr * (1/Sigma)
               dp_theta/dlam = (1/2) * dTheta/dtheta * (1/Sigma)
    """
    r, theta, p_r, p_theta, phi, t_coord = y

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    if abs(sin_t) < 0.01:
        sin_t = 0.01 if sin_t >= 0 else -0.01

    sigma = r * r + a * a * cos_t * cos_t
    sigma = max(sigma, 1e-6)  # guard: singularidad del anillo
    delta = r * r - 2.0 * m * r + a * a
    if abs(delta) < 1e-6:
        delta = 1e-6  # guard: horizonte
    P = r * r + a * a - a * xi
    inv_sigma = 1.0 / sigma

    # dr/dlam = p_r / Sigma
    dr = p_r * inv_sigma
    # dtheta/dlam = p_theta / Sigma
    dtheta = p_theta * inv_sigma

    # dphi/dlam
    dphi = (-a + a * P / delta + xi / (sin_t * sin_t)) * inv_sigma

    # dt/dlam
    dt_val = (a * (xi - a * sin_t * sin_t) + (r * r + a * a) * P / delta) * inv_sigma

    # dp_r/dlam = (1/2) * dR/dr / Sigma
    # R(r) = P² - Delta * [eta + (xi-a)²]
    # dR/dr = 2*P*(dP/dr) - (dDelta/dr)*[eta + (xi-a)²]
    #       = 2*P*2r - (2r - 2m)*[eta + (xi-a)²]
    #       = 4*r*P - (2r - 2m)*[eta + (xi-a)²]
    dR_dr = 4.0 * r * P - (2.0 * r - 2.0 * m) * (eta + (xi - a) ** 2)
    dp_r = 0.5 * dR_dr * inv_sigma

    # dp_theta/dlam = (1/2) * dTheta/dtheta / Sigma
    # Theta = eta + a²cos²θ - xi²cos²θ/sin²θ
    # dTheta/dtheta = -2a²cosθ sinθ - xi²*d(cos²θ/sin²θ)/dtheta
    # d(cos²θ/sin²θ)/dtheta = d(cot²θ)/dtheta = -2cosθ/sin³θ
    # dTheta/dtheta = -2a²cosθ sinθ + 2xi²cosθ/sin³θ
    dTheta_dtheta = -2.0 * a * a * cos_t * sin_t + 2.0 * xi * xi * cos_t / (sin_t ** 3)
    dp_theta = 0.5 * dTheta_dtheta * inv_sigma

    return [dr, dtheta, dp_r, dp_theta, dphi, dt_val]


def trace_kerr_geodesic(
    xi: float,
    eta: float,
    r_cam: float,
    theta_cam: float,
    a: float,
    m: float = _M,
    lam_max: float = 200.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 5000,
    beta_B: float = 0.0,
) -> dict:
    """
    Integra una geodesica nula en Kerr con SciPy (para validacion).

    Args:
        xi: parametro de impacto azimutal (Lz/E)
        eta: parametro de Carter (Q/E²)
        r_cam: radio del observador
        theta_cam: angulo polar del observador
        a: parametro de spin
        m: masa
        lam_max: parametro afin maximo
        rtol, atol: tolerancias del solver
        beta_B: coordenada Bardeen vertical (signo de p_theta)

    Returns:
        Diccionario con arrays r, theta, phi, t, fate.
    """
    sigma0 = r_cam * r_cam + a * a * np.cos(theta_cam) ** 2
    delta0 = r_cam * r_cam - 2.0 * m * r_cam + a * a
    P0 = r_cam * r_cam + a * a - a * xi

    R0 = P0 * P0 - delta0 * (eta + (xi - a) ** 2)
    if R0 < 0.0:
        R0 = 0.0
    p_r0 = -np.sqrt(R0)  # negativo: rayo va hacia el agujero

    cos_t0 = np.cos(theta_cam)
    sin_t0 = np.sin(theta_cam)
    if abs(sin_t0) < 1e-12:
        sin_t0 = 1e-12
    Theta0 = eta + a * a * cos_t0 * cos_t0 - xi * xi * (cos_t0 / sin_t0) ** 2
    if Theta0 < 0.0:
        Theta0 = 0.0
    # p_theta = -beta_B para backward ray tracing
    sqrt_Theta0 = np.sqrt(Theta0)
    if beta_B > 0.0:
        p_theta0 = -sqrt_Theta0
    elif beta_B < 0.0:
        p_theta0 = sqrt_Theta0
    else:
        p_theta0 = sqrt_Theta0 if theta_cam < np.pi / 2.0 else -sqrt_Theta0

    y0 = [r_cam, theta_cam, p_r0, p_theta0, 0.0, 0.0]

    r_plus = m + np.sqrt(m * m - a * a)

    def event_captured(lam, y, *args):
        return y[0] - r_plus * 1.01
    event_captured.terminal = True
    event_captured.direction = -1

    def event_escaped(lam, y, *args):
        return y[0] - 2.0 * r_cam
    event_escaped.terminal = True
    event_escaped.direction = 1

    sol = solve_ivp(
        kerr_geodesic_rhs_2nd_order,
        (0.0, lam_max),
        y0,
        args=(a, xi, eta, m),
        method="RK45",
        events=[event_captured, event_escaped],
        rtol=rtol,
        atol=atol,
        max_step=lam_max / max_steps,
        dense_output=True,
    )

    fate = "orbiting"
    if len(sol.t_events[0]) > 0:
        fate = "captured"
    elif len(sol.t_events[1]) > 0:
        fate = "escaped"

    return {
        "lam": sol.t,
        "r": sol.y[0],
        "theta": sol.y[1],
        "p_r": sol.y[2],
        "p_theta": sol.y[3],
        "phi": sol.y[4],
        "t": sol.y[5],
        "fate": fate,
    }
