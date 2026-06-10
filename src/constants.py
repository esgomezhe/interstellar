import numpy as np

# Fundamentales (unidades geometrizadas)
G: float = 1.0
C: float = 1.0
M: float = 1.0

# Radio de Schwarzschild (horizonte de eventos)
RS: float = 2.0 * G * M / C**2  # = 2M

# Radio de la esfera de fotones (orbita circular inestable de fotones)
R_PHOTON: float = 1.5 * RS  # = 3M

# Parametro de impacto critico (frontera de la sombra)
B_CRIT: float = 3.0 * np.sqrt(3.0) * M  # = 3*sqrt(3) M ~ 5.196

# Orbita circular estable mas interna (para particulas masivas / borde interno del disco)
R_ISCO: float = 3.0 * RS  # = 6M

# Radio exterior del disco por defecto
R_DISK_OUTER: float = 10.0 * RS  # = 20M

# Distancia de la camara por defecto
R_CAMERA: float = 30.0 * M


# ---------------------------------------------------------------------------
# Kerr: funciones parametrizadas por el spin a (0 <= a < M)
# ---------------------------------------------------------------------------

# Spin por defecto (0 = Schwarzschild, 0.998 ≈ Gargantua)
A_SPIN: float = 0.0


def kerr_horizons(a: float, m: float = M) -> tuple[float, float]:
    """Horizontes exterior e interior del agujero negro de Kerr.

    r± = M ± sqrt(M² - a²)
    """
    disc = m * m - a * a
    if disc < 0.0:
        disc = 0.0
    sqrt_disc = np.sqrt(disc)
    return m + sqrt_disc, m - sqrt_disc


def kerr_isco(a: float, m: float = M, prograde: bool = True) -> float:
    """
    ISCO en Kerr (Bardeen, Press, Teukolsky 1972).

    r_ISCO = M * [3 + Z2 ∓ sqrt((3-Z1)(3+Z1+2*Z2))]
    - signo para progrado, + para retrogrado.
    """
    a_norm = a / m
    z1 = 1.0 + (1.0 - a_norm**2) ** (1.0 / 3.0) * (
        (1.0 + a_norm) ** (1.0 / 3.0) + (1.0 - a_norm) ** (1.0 / 3.0)
    )
    z2 = np.sqrt(3.0 * a_norm**2 + z1**2)
    inner = (3.0 - z1) * (3.0 + z1 + 2.0 * z2)
    if inner < 0.0:
        inner = 0.0
    if prograde:
        return m * (3.0 + z2 - np.sqrt(inner))
    return m * (3.0 + z2 + np.sqrt(inner))


def kerr_photon_sphere(a: float, m: float = M, prograde: bool = True) -> float:
    """
    Radio de la esfera de fotones en Kerr.

    r_ph = 2M [1 + cos(2/3 * arccos(∓a/M))]
    - para progrado, + para retrogrado.
    """
    a_norm = a / m
    arg = -a_norm if prograde else a_norm
    arg = max(-1.0, min(1.0, arg))
    return 2.0 * m * (1.0 + np.cos(2.0 / 3.0 * np.arccos(arg)))


def kerr_delta(r: float, a: float, m: float = M) -> float:
    """Δ = r² - 2Mr + a²"""
    return r * r - 2.0 * m * r + a * a


def kerr_sigma(r: float, theta: float, a: float) -> float:
    """Σ = r² + a²cos²θ"""
    return r * r + a * a * np.cos(theta) ** 2


def kerr_omega_kepler(r: float, a: float, m: float = M, prograde: bool = True) -> float:
    """
    Velocidad angular Kepleriana en Kerr.

    Ω± = ±M^(1/2) / (r^(3/2) ± a*M^(1/2))
    """
    sqrt_m = np.sqrt(m)
    r32 = r ** 1.5
    if prograde:
        return sqrt_m / (r32 + a * sqrt_m)
    return -sqrt_m / (r32 - a * sqrt_m)


# ---------------------------------------------------------------------------
# Page-Thorne (1974): flujo exacto del disco delgado relativista
# ---------------------------------------------------------------------------

def page_thorne_flux(r: float, r_isco: float, a: float = 0.0, m: float = M) -> float:
    """
    Flujo radiado por unidad de area del disco delgado relativista
    (Page & Thorne 1974, ec. 15n), sin normalizar.

    Con x = sqrt(r/M) y x0 = sqrt(r_isco/M), y x1, x2, x3 las raices de
    x^3 - 3x + 2a* = 0:

        F(x) ∝ [ x - x0 - (3a*/2) ln(x/x0) - Σᵢ cᵢ ln((x-xᵢ)/(x0-xᵢ)) ]
               / [ x^4 (x^3 - 3x + 2a*) ]

        cᵢ = 3(xᵢ - a*)² / [xᵢ ∏_{j≠i}(xᵢ - xⱼ)]

    Resuelve la conservacion de energia y momento angular del gas en
    orbitas Keplerianas con torque cero en el ISCO. Es la version EXACTA
    del perfil de Novikov-Thorne aproximado (r_isco/r)^(3/4)(1-sqrt())^(1/4)
    que se usaba antes, y depende del spin a traves de las raices xᵢ.

    Asintoticamente F ~ r^(-3) (T ~ r^(-3/4), Shakura-Sunyaev).

    Returns:
        Flujo F >= 0 (0 si r <= r_isco). Normalizar con page_thorne_norm_inv.
    """
    if r <= r_isco:
        return 0.0
    a_star = a / m
    x = np.sqrt(r / m)
    x0 = np.sqrt(r_isco / m)

    acs = np.arccos(min(max(a_star, 0.0), 1.0))
    x1 = 2.0 * np.cos((acs - np.pi) / 3.0)
    x2 = 2.0 * np.cos((acs + np.pi) / 3.0)
    x3 = -2.0 * np.cos(acs / 3.0)

    d1 = x1 * (x1 - x2) * (x1 - x3)
    d2 = x2 * (x2 - x1) * (x2 - x3)
    d3 = x3 * (x3 - x1) * (x3 - x2)
    c1 = 3.0 * (x1 - a_star) ** 2 / d1
    # a* = 0: x2 -> 0 y el limite de c2 es 0 (numerador ~ x2²)
    c2 = 0.0 if abs(d2) < 1e-9 else 3.0 * (x2 - a_star) ** 2 / d2
    c3 = 3.0 * (x3 - a_star) ** 2 / d3

    eps = 1e-9
    bracket = (
        x - x0
        - 1.5 * a_star * np.log(x / x0)
        - c1 * np.log((x - x1) / (x0 - x1))
        - c2 * np.log(max(x - x2, eps) / max(x0 - x2, eps))
        - c3 * np.log((x - x3) / (x0 - x3))
    )
    flux = bracket / (x ** 4 * (x ** 3 - 3.0 * x + 2.0 * a_star))
    return max(flux, 0.0)


def page_thorne_norm_inv(
    r_isco: float, r_outer: float, a: float = 0.0, m: float = M,
) -> float:
    """
    Normalizacion de exposicion: 1/F en el radio medio geometrico del disco.

    Normalizar por el PICO de F aplasta la imagen cuando el rango dinamico
    es grande (en a=0.998 el pico junto al ISCO es ~10^3 veces el borde
    exterior). Referenciar al radio medio expone bien el cuerpo del disco
    y deja que el nucleo supere 1: el tone mapping Reinhard absorbe el
    exceso y el color de cuerpo negro se corre hacia el blanco (nucleo HDR),
    como una camara fotografiando una fuente sobre-expuesta.
    """
    r_ref = np.sqrt(r_isco * r_outer)
    f_ref = page_thorne_flux(r_ref, r_isco, a, m)
    return 1.0 / f_ref if f_ref > 0.0 else 0.0
