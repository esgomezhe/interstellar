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
