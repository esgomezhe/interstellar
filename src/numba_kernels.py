"""
Kernels optimizados con Numba para el ray tracer de agujero negro.

Reemplaza la cadena scipy.solve_ivp -> disk -> effects -> colormap
por funciones compiladas a codigo nativo con @njit.

Incluye:
  - Integrador RK4 manual para la geodesica (Numba no soporta scipy)
  - Deteccion de cruces ecuatoriales
  - Efectos relativistas (redshift, Doppler, beaming)
  - Colormap de cuerpo negro (Tanner Helland)
  - Pipeline completo por pixel
  - Render paralelo con numba.prange + cache por simetria
"""

import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------
# Constantes (duplicadas como literales para que Numba las resuelva en compilacion)
# ---------------------------------------------------------------------------
_RS: float = 2.0
_M: float = 1.0
_R_ISCO: float = 6.0
_R_DISK_OUTER: float = 20.0
_T_BASE: float = 4000.0
_BEAMING_POWER: float = 3.0
_MAX_CROSSINGS: int = 5


# ---------------------------------------------------------------------------
# Integrador RK4 para geodesicas nulas
# ---------------------------------------------------------------------------

@njit(cache=True)
def _rk4_step(u, du, rs, dphi):
    """Un paso RK4 para u'' + u = (3/2)*rs*u^2.

    Sistema: dy1/dphi = y2, dy2/dphi = -y1 + 1.5*rs*y1^2
    """
    # k1
    k1_u = du
    k1_du = -u + 1.5 * rs * u * u

    # k2
    u2 = u + 0.5 * dphi * k1_u
    du2 = du + 0.5 * dphi * k1_du
    k2_u = du2
    k2_du = -u2 + 1.5 * rs * u2 * u2

    # k3
    u3 = u + 0.5 * dphi * k2_u
    du3 = du + 0.5 * dphi * k2_du
    k3_u = du3
    k3_du = -u3 + 1.5 * rs * u3 * u3

    # k4
    u4 = u + dphi * k3_u
    du4 = du + dphi * k3_du
    k4_u = du4
    k4_du = -u4 + 1.5 * rs * u4 * u4

    u_new = u + (dphi / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u)
    du_new = du + (dphi / 6.0) * (k1_du + 2.0 * k2_du + 2.0 * k3_du + k4_du)
    return u_new, du_new


@njit(cache=True)
def trace_geodesic_rk4(b, r_cam, rs, phi_max, n_steps):
    """
    Integra una geodesica de foton con RK4 de paso fijo.

    Args:
        b: Parametro de impacto.
        r_cam: Distancia radial de la camara.
        rs: Radio de Schwarzschild.
        phi_max: Angulo maximo de integracion.
        n_steps: Numero de pasos RK4.

    Returns:
        (phi_arr, r_arr, n_valid, fate)
        fate: 0=capturado, 1=escapo, 2=orbitando
    """
    u0 = 1.0 / r_cam
    val = 1.0 / (b * b) - u0 * u0 + rs * u0 * u0 * u0
    if val < 0.0:
        val = 0.0
    du0 = np.sqrt(val)

    dphi = phi_max / n_steps
    u_capture = 1.0 / rs
    u_escape = 0.5 / r_cam

    phi_arr = np.empty(n_steps + 1)
    r_arr = np.empty(n_steps + 1)

    phi_arr[0] = 0.0
    r_arr[0] = r_cam

    u = u0
    du = du0
    fate = 2  # orbitando
    n_valid = n_steps + 1

    for k in range(n_steps):
        u_new, du_new = _rk4_step(u, du, rs, dphi)
        phi_arr[k + 1] = (k + 1) * dphi

        if u_new <= 1e-12:
            u_new = 1e-12
        r_arr[k + 1] = 1.0 / u_new

        if u_new >= u_capture:
            fate = 0  # capturado
            n_valid = k + 2
            break

        if u_new < u_escape:
            fate = 1  # escapo
            n_valid = k + 2
            break

        u = u_new
        du = du_new

    return phi_arr, r_arr, n_valid, fate


# ---------------------------------------------------------------------------
# Deteccion de cruces ecuatoriales
# ---------------------------------------------------------------------------

@njit(cache=True)
def find_crossings(phi, r, n_pts, e1_z, e2_z, r_inner, r_outer):
    """
    Encuentra cruces con el plano ecuatorial (z=0).

    z(psi) = r(psi) * [cos(psi)*e1_z + sin(psi)*e2_z]
    Los cruces ocurren donde z cambia de signo.

    Returns:
        (r_cross, psi_cross, count) con arreglos de tamanio fijo.
    """
    r_cross = np.empty(_MAX_CROSSINGS)
    psi_cross = np.empty(_MAX_CROSSINGS)
    cross_idx = np.empty(_MAX_CROSSINGS, dtype=np.int64)
    count = 0

    for k in range(n_pts - 1):
        z_k = r[k] * (np.cos(phi[k]) * e1_z + np.sin(phi[k]) * e2_z)
        z_k1 = r[k + 1] * (np.cos(phi[k + 1]) * e1_z + np.sin(phi[k + 1]) * e2_z)

        if z_k * z_k1 < 0.0:
            az_k = abs(z_k)
            az_k1 = abs(z_k1)
            frac = az_k / (az_k + az_k1)
            rc = r[k] + frac * (r[k + 1] - r[k])
            pc = phi[k] + frac * (phi[k + 1] - phi[k])

            if r_inner <= rc <= r_outer:
                r_cross[count] = rc
                psi_cross[count] = pc
                cross_idx[count] = k
                count += 1
                if count >= _MAX_CROSSINGS:
                    break

    return r_cross, psi_cross, cross_idx, count


# ---------------------------------------------------------------------------
# Efectos relativistas
# ---------------------------------------------------------------------------

@njit(cache=True)
def disk_g_factor(r_hit, lam, a, m):
    """
    Factor g = nu_obs / nu_em exacto (Cunningham 1975) para un emisor en
    orbita circular Kepleriana prograda en el plano ecuatorial.

        g = sqrt(1 - 3M/r + 2a*sqrt(M)/r^(3/2))
            / [(1 + a*sqrt(M)/r^(3/2)) * (1 - Omega*lam)]

    - El numerador y (1 + a*sqrt(M)/r^(3/2)) forman 1/u^t del emisor
      (Bardeen-Press-Teukolsky 1972): incluye redshift gravitacional
      Y Doppler transversal de la orbita en un solo termino.
    - (1 - Omega*lam) es el Doppler azimutal, con lam = L_z/E del foton
      (cantidad CONSERVADA a lo largo de la geodesica, no requiere conocer
      la direccion local de emision).
    - a = 0 reduce a g = sqrt(1 - 3M/r) / (1 - Omega*lam): Schwarzschild exacto.

    Args:
        r_hit: radio Boyer-Lindquist del punto de emision.
        lam: momento angular especifico del foton (xi en Kerr, -b*n_z en
             Schwarzschild, donde n es la normal al plano orbital del rayo).
        a: spin del agujero negro.
        m: masa.
    """
    sqrt_m = np.sqrt(m)
    r32 = r_hit ** 1.5
    omega = sqrt_m / (r32 + a * sqrt_m)

    x = a * sqrt_m / r32
    B = 1.0 - 3.0 * m / r_hit + 2.0 * x
    if B < 1e-6:
        B = 1e-6  # bajo la orbita circular de fotones: sin orbita Kepleriana

    denom = (1.0 + x) * (1.0 - omega * lam)
    if abs(denom) < 1e-6:
        denom = 1e-6 if denom >= 0.0 else -1e-6

    g = np.sqrt(B) / denom

    if g < 0.05:
        g = 0.05
    elif g > 5.0:
        g = 5.0
    return g


# ---------------------------------------------------------------------------
# Colormap de cuerpo negro (Tanner Helland)
# ---------------------------------------------------------------------------

@njit(cache=True)
def blackbody_rgb(temperature):
    """Temperatura de cuerpo negro a RGB [0, 1]. Valida 1000K - 40000K."""
    temp = temperature
    if temp < 1000.0:
        temp = 1000.0
    elif temp > 40000.0:
        temp = 40000.0
    temp = temp / 100.0

    # Rojo
    if temp <= 66.0:
        red = 1.0
    else:
        red = 329.698727446 * (temp - 60.0) ** (-0.1332047592) / 255.0

    # Verde
    if temp <= 66.0:
        green = (99.4708025861 * np.log(temp) - 161.1195681661) / 255.0
    else:
        green = 288.1221695283 * (temp - 60.0) ** (-0.0755148492) / 255.0

    # Azul
    if temp >= 66.0:
        blue = 1.0
    elif temp <= 19.0:
        blue = 0.0
    else:
        blue = (138.5177312231 * np.log(temp - 10.0) - 305.0447927307) / 255.0

    # Clamp
    if red < 0.0:
        red = 0.0
    elif red > 1.0:
        red = 1.0
    if green < 0.0:
        green = 0.0
    elif green > 1.0:
        green = 1.0
    if blue < 0.0:
        blue = 0.0
    elif blue > 1.0:
        blue = 1.0

    return red, green, blue


# ---------------------------------------------------------------------------
# Pipeline completo por pixel (a partir de geodesica ya trazada)
# ---------------------------------------------------------------------------

@njit(cache=True)
def limb_darkening(cos_theta_e, mu=0.5):
    """
    Limb darkening por scattering de electrones en la atmosfera del disco.

    I_obs = I_0 * (1 + mu * cos_theta_e) / (1 + mu)

    cos_theta_e: coseno del angulo entre la normal al disco (eje z) y la
                 direccion del foton en el punto de cruce.
    mu: coeficiente de limb darkening (~0.5 para scattering por electrones).
    """
    if cos_theta_e < 0.0:
        cos_theta_e = -cos_theta_e
    return (1.0 + mu * cos_theta_e) / (1.0 + mu)


@njit(cache=True)
def _compute_limb_factor(psi_hit, e1, e2, phi_arr, r_arr, cross_idx, n_valid):
    """
    Calcula el factor de limb darkening para un cruce ecuatorial.

    Estima la direccion del rayo en el punto de cruce a partir de la
    tangente a la trayectoria, y computa cos(theta_e) con la normal del disco.
    """
    # Direccion del rayo: tangente a r(psi)*[cos(psi)*e1 + sin(psi)*e2]
    # Aproximacion: usar la diferencia finita entre puntos adyacentes
    k = cross_idx
    if k <= 0 or k >= n_valid - 1:
        return 1.0

    # Posicion en puntos k y k+1
    cp0, sp0 = np.cos(phi_arr[k]), np.sin(phi_arr[k])
    cp1, sp1 = np.cos(phi_arr[k + 1]), np.sin(phi_arr[k + 1])
    r0, r1 = r_arr[k], r_arr[k + 1]

    dx = r1 * (cp1 * e1[0] + sp1 * e2[0]) - r0 * (cp0 * e1[0] + sp0 * e2[0])
    dy = r1 * (cp1 * e1[1] + sp1 * e2[1]) - r0 * (cp0 * e1[1] + sp0 * e2[1])
    dz = r1 * (cp1 * e1[2] + sp1 * e2[2]) - r0 * (cp0 * e1[2] + sp0 * e2[2])

    d_len = np.sqrt(dx * dx + dy * dy + dz * dz)
    if d_len < 1e-12:
        return 1.0

    # cos(theta_e) = |dz / d_len| (normal del disco es eje z)
    cos_theta_e = abs(dz / d_len)
    return limb_darkening(cos_theta_e)


@njit(cache=True)
def _hash21(px, py):
    """Hash 2D -> 1D para ruido procedural."""
    px = (px * 123.34) % 1.0
    py = (py * 456.21) % 1.0
    dot = px * (px + 45.32) + py * (py + 45.32)
    return (px * py + dot) % 1.0


@njit(cache=True)
def _noise2d(px, py):
    """Ruido 2D basado en hash con interpolacion suave."""
    ix = int(np.floor(px))
    iy = int(np.floor(py))
    fx = px - ix
    fy = py - iy
    # smoothstep
    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)

    a = _hash21(ix % 1000 * 0.001, iy % 1000 * 0.001)
    b = _hash21((ix + 1) % 1000 * 0.001, iy % 1000 * 0.001)
    c = _hash21(ix % 1000 * 0.001, (iy + 1) % 1000 * 0.001)
    d = _hash21((ix + 1) % 1000 * 0.001, (iy + 1) % 1000 * 0.001)

    ab = a + fx * (b - a)
    cd = c + fx * (d - c)
    return ab + fy * (cd - ab)


@njit(cache=True)
def _fbm(px, py):
    """Fractional Brownian Motion con 4 octavas."""
    val = 0.0
    amp = 0.5
    for _ in range(4):
        val += amp * _noise2d(px, py)
        px *= 2.0
        py *= 2.0
        amp *= 0.5
    return val


@njit(cache=True)
def disk_turbulence(r, psi, e1, e2, m, t):
    """
    Textura de turbulencia del disco con rotacion diferencial.

    Modula la emision con ruido procedural que rota a velocidad kepleriana.
    """
    _TURB_AMPLITUDE = 0.35

    # Posicion 3D del punto para calcular angulo azimutal
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    hit_x = r * (cos_psi * e1[0] + sin_psi * e2[0])
    hit_y = r * (cos_psi * e1[1] + sin_psi * e2[1])
    azimuth = np.arctan2(hit_y, hit_x)

    # Rotacion diferencial kepleriana + shear espiral (trailing arms)
    omega = np.sqrt(m / (r * r * r))
    phi_s = azimuth - omega * t + 1.5 * np.log(r)

    # Muestrear el ruido sobre un circulo lo hace periodico en azimut
    # (sin costura en phi = +-pi); log(r) desplaza el circulo radialmente
    noise_x = np.cos(phi_s) * 2.5 + np.log(r) * 4.0
    noise_y = np.sin(phi_s) * 2.5
    turb = _fbm(noise_x, noise_y) * 2.0 - 1.0

    return 1.0 + _TURB_AMPLITUDE * turb


@njit(cache=True)
def page_thorne_flux(r, r_isco, a, m):
    """
    Flujo del disco delgado relativista (Page & Thorne 1974, ec. 15n).

    Version exacta (dependiente del spin) del perfil Novikov-Thorne
    aproximado que se usaba antes. Copia njit de la referencia en
    src/constants.py (mantener sincronizadas — hay test de consistencia).
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
    x_m_x2 = x - x2
    if x_m_x2 < eps:
        x_m_x2 = eps
    x0_m_x2 = x0 - x2
    if x0_m_x2 < eps:
        x0_m_x2 = eps

    bracket = (
        x - x0
        - 1.5 * a_star * np.log(x / x0)
        - c1 * np.log((x - x1) / (x0 - x1))
        - c2 * np.log(x_m_x2 / x0_m_x2)
        - c3 * np.log((x - x3) / (x0 - x3))
    )
    flux = bracket / (x ** 4 * (x ** 3 - 3.0 * x + 2.0 * a_star))
    if flux < 0.0:
        flux = 0.0
    return flux


@njit(cache=True)
def page_thorne_norm_inv(r_isco, r_outer, a, m):
    """
    Normalizacion de exposicion: 1/F en el radio medio geometrico.

    El nucleo del disco queda > 1 (HDR): Reinhard absorbe el exceso de
    brillo y el cuerpo negro se corre al blanco. Ver src/constants.py.
    """
    r_ref = np.sqrt(r_isco * r_outer)
    f_ref = page_thorne_flux(r_ref, r_isco, a, m)
    if f_ref <= 0.0:
        return 0.0
    return 1.0 / f_ref


@njit(cache=True)
def page_thorne_emission(r, r_isco, a, m, f_norm_inv):
    """
    Perfil de temperatura relativo: T_norm = (F/F_ref)^(1/4).

    Stefan-Boltzmann: F = sigma*T^4, asi que la temperatura local del disco
    es T(r) = T_ref * T_norm(r) y el brillo es T_norm^4. T_norm > 1 cerca
    del pico de emision (nucleo HDR).
    """
    f_norm = page_thorne_flux(r, r_isco, a, m) * f_norm_inv
    if f_norm <= 0.0:
        return 0.0
    return f_norm ** 0.25


@njit(cache=True)
def _starfield(direction_x, direction_y, direction_z):
    """
    Campo estelar procedural para rayos escapados.

    Genera estrellas pseudo-aleatorias basadas en la direccion del rayo.
    """
    # Coordenadas esfericas
    theta_sky = np.arccos(max(-1.0, min(1.0, direction_z)))
    phi_sky = np.arctan2(direction_y, direction_x)

    cr, cg, cb = 0.0, 0.0, 0.0

    for layer in range(3):
        scale = 80.0 + layer * 60.0
        # Numero ENTERO de celdas en phi: grid periodico, sin costura en
        # el corte de rama de arctan2 (phi = +-pi)
        n_phi = np.floor(2.0 * np.pi * scale)
        u = (phi_sky + np.pi) / (2.0 * np.pi) * n_phi
        cy = theta_sky * scale
        ix = int(np.floor(u)) % int(n_phi)
        iy = int(np.floor(cy))
        fx = u - np.floor(u)
        fy = cy - iy

        seed1 = (0.13 + layer * 0.07)
        seed2 = (0.27 + layer * 0.11)
        seed3 = (0.41 + layer * 0.03)
        h1 = ((ix * seed1 + iy * 0.73) * 12345.6789) % 1.0
        if h1 < 0:
            h1 += 1.0
        h2 = ((ix * seed2 + iy * 0.31) * 67890.1234) % 1.0
        if h2 < 0:
            h2 += 1.0
        h3 = ((ix * seed3 + iy * 0.57) * 13579.2468) % 1.0
        if h3 < 0:
            h3 += 1.0

        if h1 > 0.85:
            dx = fx - h2
            dy = fy - h3
            dist = np.sqrt(dx * dx + dy * dy)

            if dist < 0.05:
                brightness = max(0.0, 1.0 - dist / 0.05)
                brightness = brightness * brightness  # smoothstep-like

                temp_star = 3000.0 + h2 * 20000.0
                sr, sg, sb = blackbody_rgb(temp_star)
                magnitude = 0.3 + h3 * 0.7

                cr += sr * brightness * magnitude
                cg += sg * brightness * magnitude
                cb += sb * brightness * magnitude

    return min(cr, 1.0), min(cg, 1.0), min(cb, 1.0)


@njit(cache=True)
def _color_from_geodesic(
    phi, r, n_pts, e1, e2, lam,
    rs, m, r_inner, r_outer, base_temp, beaming_power, t, fate, f_pt_inv,
):
    """
    Calcula el color RGB de un pixel dada su geodesica pre-computada.

    Busca cruces con el disco, aplica el g-factor exacto de Cunningham por
    cruce (cada imagen secundaria con su propio corrimiento), y tone mapping
    Reinhard. Si el rayo escapa sin cruzar el disco, muestra el campo estelar.

    Args:
        lam: L_z/E del foton (conservada): lam = -b * n_z, con n = e1 x e2.
    """
    r_cross, psi_cross, c_idx, n_cross = find_crossings(
        phi, r, n_pts, e1[2], e2[2], r_inner, r_outer,
    )

    if n_cross == 0:
        if fate == 1 and n_pts >= 2:  # rayo escapado -> estrellas
            # Direccion de propagacion: tangente entre los dos ultimos puntos
            k = n_pts - 1
            cp0, sp0 = np.cos(phi[k - 1]), np.sin(phi[k - 1])
            cp1, sp1 = np.cos(phi[k]), np.sin(phi[k])
            dx = r[k] * (cp1 * e1[0] + sp1 * e2[0]) - r[k - 1] * (cp0 * e1[0] + sp0 * e2[0])
            dy = r[k] * (cp1 * e1[1] + sp1 * e2[1]) - r[k - 1] * (cp0 * e1[1] + sp0 * e2[1])
            dz = r[k] * (cp1 * e1[2] + sp1 * e2[2]) - r[k - 1] * (cp0 * e1[2] + sp0 * e2[2])
            d_len = np.sqrt(dx * dx + dy * dy + dz * dz)
            if d_len < 1e-12:
                return 0.0, 0.0, 0.0
            return _starfield(dx / d_len, dy / d_len, dz / d_len)
        return 0.0, 0.0, 0.0

    # Acumular cada cruce con su propio g-factor y color de cuerpo negro
    total_i = 0.0
    acc_r = 0.0
    acc_g = 0.0
    acc_b = 0.0
    for c in range(n_cross):
        weight = 1.0 if c == 0 else 0.5  # imagenes secundarias atenuadas
        g_c = disk_g_factor(r_cross[c], lam, 0.0, m)
        # T_norm para el color; el BRILLO sigue al flujo F = T_norm^4
        # (Stefan-Boltzmann: la potencia radiada va como T^4)
        t_norm = page_thorne_emission(r_cross[c], r_inner, 0.0, m, f_pt_inv)
        f_norm = t_norm ** 4
        limb = _compute_limb_factor(psi_cross[c], e1, e2, phi, r, c_idx[c], n_pts)
        turb = disk_turbulence(r_cross[c], psi_cross[c], e1, e2, m, t)
        i_c = f_norm * limb * turb * g_c ** beaming_power * weight
        if i_c <= 0.0:
            continue
        cr, cg, cb = blackbody_rgb(base_temp * t_norm * g_c)
        total_i += i_c
        acc_r += i_c * cr
        acc_g += i_c * cg
        acc_b += i_c * cb

    if total_i <= 0.0:
        return 0.0, 0.0, 0.0

    # Color promedio ponderado por intensidad de cada cruce
    inv_i = 1.0 / total_i
    cr = acc_r * inv_i
    cg = acc_g * inv_i
    cb = acc_b * inv_i

    # Tone mapping Reinhard; la compresion perceptual la hace el gamma del
    # display (una sola compresion — dos aplastarian el rango del flujo fisico)
    mapped = total_i / (1.0 + total_i)
    brightness = mapped

    return min(cr * brightness, 1.0), min(cg * brightness, 1.0), min(cb * brightness, 1.0)


# ---------------------------------------------------------------------------
# Render paralelo con simetria horizontal
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def render_frame_parallel(
    b_arr, e1_arr, e2_arr, cam_pos,
    r_cam, rs, m, phi_max, n_steps,
    r_inner, r_outer, base_temp, beaming_power, t=0.0,
):
    """
    Renderiza el frame completo en paralelo con Numba.

    Aprovecha la simetria horizontal: los pixeles (i, j) y (w-1-i, j)
    comparten el mismo parametro de impacto y la misma geodesica.
    Solo se traza la geodesica una vez por par simetrico, ahorrando ~50%
    del costo de integracion.

    Los efectos Doppler se calculan independientemente para cada pixel
    porque la asimetria de brillo es fisica (lado que se acerca vs aleja).

    Args:
        b_arr: Parametros de impacto, forma (h, w).
        e1_arr: Vectores e1 del plano orbital, forma (h, w, 3).
        e2_arr: Vectores e2 del plano orbital, forma (h, w, 3).
        cam_pos: Posicion 3D de la camara, forma (3,).
        r_cam: Distancia de la camara.
        rs: Radio de Schwarzschild.
        m: Masa del agujero negro.
        phi_max: Angulo maximo de integracion.
        n_steps: Pasos RK4 por geodesica.
        r_inner: Borde interno del disco.
        r_outer: Borde externo del disco.
        base_temp: Temperatura base del disco (K).
        beaming_power: Exponente del beaming relativista.

    Returns:
        Imagen RGB, forma (h, w, 3), valores en [0, 1].
    """
    h = b_arr.shape[0]
    w = b_arr.shape[1]
    image = np.zeros((h, w, 3))
    half_w = (w + 1) // 2  # mitad redondeada arriba (para anchos impares)

    # Normalizacion del perfil Page-Thorne (una vez por frame)
    f_pt_inv = page_thorne_norm_inv(r_inner, r_outer, 0.0, m)

    for idx in prange(h * half_w):
        j = idx // half_w
        i = idx % half_w
        i_mirror = w - 1 - i

        b = b_arr[j, i]
        if b < 1e-6:
            continue

        # Trazar geodesica UNA sola vez para el par simetrico
        phi, r, n_valid, fate = trace_geodesic_rk4(b, r_cam, rs, phi_max, n_steps)

        # Pixel izquierdo
        e1_left = e1_arr[j, i]
        e2_left = e2_arr[j, i]
        # L_z/E del foton: lam = -b * n_z, con n = e1 x e2 (normal al plano orbital)
        n_z_left = e1_left[0] * e2_left[1] - e1_left[1] * e2_left[0]
        lam_left = -b * n_z_left
        cr, cg, cb = _color_from_geodesic(
            phi, r, n_valid, e1_left, e2_left, lam_left,
            rs, m, r_inner, r_outer, base_temp, beaming_power, t, fate, f_pt_inv,
        )
        image[j, i, 0] = cr
        image[j, i, 1] = cg
        image[j, i, 2] = cb

        # Pixel derecho (espejo) — solo si es diferente del izquierdo
        if i != i_mirror:
            e1_right = e1_arr[j, i_mirror]
            e2_right = e2_arr[j, i_mirror]
            n_z_right = e1_right[0] * e2_right[1] - e1_right[1] * e2_right[0]
            lam_right = -b * n_z_right
            cr, cg, cb = _color_from_geodesic(
                phi, r, n_valid, e1_right, e2_right, lam_right,
                rs, m, r_inner, r_outer, base_temp, beaming_power, t, fate, f_pt_inv,
            )
            image[j, i_mirror, 0] = cr
            image[j, i_mirror, 1] = cg
            image[j, i_mirror, 2] = cb

    return image


# ---------------------------------------------------------------------------
# Warmup: compila todas las funciones JIT antes del render real
# ---------------------------------------------------------------------------

def warmup():
    """
    Ejecuta un render minimo para forzar la compilacion JIT.

    La primera invocacion de funciones @njit es lenta (compilacion LLVM).
    Con cache=True el resultado se guarda en disco para futuras ejecuciones.
    """
    b = np.array([[5.0]])
    e1 = np.array([[[0.259, 0.0, 0.966]]])
    e2 = np.array([[[0.966, 0.0, -0.259]]])
    cam = np.array([7.76, 0.0, 29.0])

    render_frame_parallel(
        b, e1, e2, cam,
        30.0, 2.0, 1.0, 10.0, 100,
        6.0, 20.0, 4000.0, 3.0,
    )


# ===========================================================================
# KERR: integrador y render para agujero negro rotante
# ===========================================================================

@njit(cache=True)
def _kerr_rk4_step(r, theta, p_r, p_theta, phi, t_coord,
                    a, xi, eta, m, dlam):
    """
    Un paso RK4 para el sistema de Carter de 6 variables.

    Variables: [r, theta, p_r, p_theta, phi, t]
    Donde p_r = Sigma * dr/dlam, p_theta = Sigma * dtheta/dlam
    """
    def derivs(r_, theta_, p_r_, p_theta_, phi_, t_):
        cos_t = np.cos(theta_)
        sin_t = np.sin(theta_)
        if abs(sin_t) < 0.01:
            sin_t = 0.01 if sin_t >= 0 else -0.01

        sigma = r_ * r_ + a * a * cos_t * cos_t
        if sigma < 1e-6:
            sigma = 1e-6
        delta = r_ * r_ - 2.0 * m * r_ + a * a
        if abs(delta) < 1e-6:
            delta = 1e-6
        P = r_ * r_ + a * a - a * xi
        inv_sigma = 1.0 / sigma

        dr = p_r_ * inv_sigma
        dtheta = p_theta_ * inv_sigma

        dphi = (-a + a * P / delta + xi / (sin_t * sin_t)) * inv_sigma
        dt = (-a * a * sin_t * sin_t + (r_ * r_ + a * a) * P / delta) * inv_sigma

        # dp_r = (1/2) * dR/dr / Sigma
        dR_dr = 4.0 * r_ * P - (2.0 * r_ - 2.0 * m) * (eta + (xi - a) ** 2)
        dp_r = 0.5 * dR_dr * inv_sigma

        # dp_theta = (1/2) * dTheta/dtheta / Sigma
        dTheta_dtheta = -2.0 * a * a * cos_t * sin_t + 2.0 * xi * xi * cos_t / (sin_t ** 3)
        dp_theta = 0.5 * dTheta_dtheta * inv_sigma

        return dr, dtheta, dp_r, dp_theta, dphi, dt

    # k1
    k1 = derivs(r, theta, p_r, p_theta, phi, t_coord)

    # k2
    r2 = r + 0.5 * dlam * k1[0]
    th2 = theta + 0.5 * dlam * k1[1]
    pr2 = p_r + 0.5 * dlam * k1[2]
    pt2 = p_theta + 0.5 * dlam * k1[3]
    ph2 = phi + 0.5 * dlam * k1[4]
    t2 = t_coord + 0.5 * dlam * k1[5]
    k2 = derivs(r2, th2, pr2, pt2, ph2, t2)

    # k3
    r3 = r + 0.5 * dlam * k2[0]
    th3 = theta + 0.5 * dlam * k2[1]
    pr3 = p_r + 0.5 * dlam * k2[2]
    pt3 = p_theta + 0.5 * dlam * k2[3]
    ph3 = phi + 0.5 * dlam * k2[4]
    t3 = t_coord + 0.5 * dlam * k2[5]
    k3 = derivs(r3, th3, pr3, pt3, ph3, t3)

    # k4
    r4 = r + dlam * k3[0]
    th4 = theta + dlam * k3[1]
    pr4 = p_r + dlam * k3[2]
    pt4 = p_theta + dlam * k3[3]
    ph4 = phi + dlam * k3[4]
    t4 = t_coord + dlam * k3[5]
    k4 = derivs(r4, th4, pr4, pt4, ph4, t4)

    r_new = r + (dlam / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
    theta_new = theta + (dlam / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
    pr_new = p_r + (dlam / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
    pt_new = p_theta + (dlam / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
    phi_new = phi + (dlam / 6.0) * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4])
    t_new = t_coord + (dlam / 6.0) * (k1[5] + 2.0 * k2[5] + 2.0 * k3[5] + k4[5])

    return r_new, theta_new, pr_new, pt_new, phi_new, t_new


@njit(cache=True)
def trace_kerr_geodesic_rk4(xi, eta, r_cam, theta_cam, a, m, lam_max, n_steps,
                             beta_B=0.0):
    """
    Integra una geodesica nula en Kerr con RK4 de paso fijo.

    Args:
        beta_B: coordenada Bardeen vertical (para determinar signo de p_theta).

    Returns:
        (r_arr, theta_arr, phi_arr, n_valid, fate)
        fate: 0=capturado, 1=escapo, 2=orbitando
    """
    sigma0 = r_cam * r_cam + a * a * np.cos(theta_cam) ** 2
    delta0 = r_cam * r_cam - 2.0 * m * r_cam + a * a
    P0 = r_cam * r_cam + a * a - a * xi

    R0 = P0 * P0 - delta0 * (eta + (xi - a) ** 2)
    if R0 < 0.0:
        R0 = 0.0
    p_r0 = -np.sqrt(R0)  # negativo: rayo va hacia adentro

    cos_t0 = np.cos(theta_cam)
    sin_t0 = np.sin(theta_cam)
    if abs(sin_t0) < 1e-12:
        sin_t0 = 1e-12
    Theta0 = eta + a * a * cos_t0 * cos_t0 - xi * xi * (cos_t0 / sin_t0) ** 2
    if Theta0 < 0.0:
        Theta0 = 0.0
    # p_theta = -beta_B para backward ray tracing
    # (el rayo retrocede en la direccion opuesta a la llegada)
    sqrt_Theta0 = np.sqrt(Theta0)
    if beta_B > 0.0:
        p_theta0 = -sqrt_Theta0
    elif beta_B < 0.0:
        p_theta0 = sqrt_Theta0
    else:
        # beta_B = 0: rayo ecuatorial, p_theta hacia el ecuador
        p_theta0 = sqrt_Theta0 if theta_cam < np.pi / 2.0 else -sqrt_Theta0

    # Horizonte exterior
    disc = m * m - a * a
    if disc < 0.0:
        disc = 0.0
    r_plus = m + np.sqrt(disc)

    dlam = lam_max / n_steps

    r_arr = np.empty(n_steps + 1)
    theta_arr = np.empty(n_steps + 1)
    phi_arr = np.empty(n_steps + 1)

    r_arr[0] = r_cam
    theta_arr[0] = theta_cam
    phi_arr[0] = 0.0

    r_ = r_cam
    theta_ = theta_cam
    p_r_ = p_r0
    p_theta_ = p_theta0
    phi_ = 0.0
    t_ = 0.0

    fate = 2  # orbitando
    n_valid = n_steps + 1

    for k in range(n_steps):
        r_new, theta_new, pr_new, pt_new, phi_new, t_new = _kerr_rk4_step(
            r_, theta_, p_r_, p_theta_, phi_, t_,
            a, xi, eta, m, dlam,
        )

        # Paso polar: reflexion de momento + salto de pi en phi (el rayo
        # pasa sobre el polo y continua en el meridiano opuesto)
        if theta_new < 0.01:
            theta_new = 0.01
            pt_new = abs(pt_new)
            phi_new += np.pi
        elif theta_new > np.pi - 0.01:
            theta_new = np.pi - 0.01
            pt_new = -abs(pt_new)
            phi_new += np.pi

        r_arr[k + 1] = r_new
        theta_arr[k + 1] = theta_new
        phi_arr[k + 1] = phi_new

        # Captura: r <= r+ * 1.01
        if r_new <= r_plus * 1.01:
            fate = 0
            n_valid = k + 2
            break

        # Escape: r > r_cam y alejandose (p_r > 0)
        if r_new > r_cam and pr_new > 0.0:
            fate = 1
            n_valid = k + 2
            break

        r_ = r_new
        theta_ = theta_new
        p_r_ = pr_new
        p_theta_ = pt_new
        phi_ = phi_new
        t_ = t_new

    return r_arr, theta_arr, phi_arr, n_valid, fate


@njit(cache=True)
def kerr_find_crossings(r_arr, theta_arr, phi_arr, n_pts, r_inner, r_outer):
    """
    Detecta cruces con el plano ecuatorial (theta = pi/2) en Kerr.

    A diferencia de Schwarzschild donde usamos z(psi) con plano orbital,
    aqui usamos directamente theta(lambda) porque el movimiento es 3D.
    """
    r_cross = np.empty(_MAX_CROSSINGS)
    theta_cross_dir = np.empty(_MAX_CROSSINGS)  # +1: arriba->abajo, -1: abajo->arriba
    phi_cross = np.empty(_MAX_CROSSINGS)
    cross_idx = np.empty(_MAX_CROSSINGS, dtype=np.int64)
    count = 0
    half_pi = np.pi / 2.0

    for k in range(n_pts - 1):
        d_prev = theta_arr[k] - half_pi
        d_next = theta_arr[k + 1] - half_pi

        if d_prev * d_next < 0.0:
            # Interpolacion lineal para encontrar el punto de cruce
            frac = abs(d_prev) / (abs(d_prev) + abs(d_next))
            rc = r_arr[k] + frac * (r_arr[k + 1] - r_arr[k])
            pc = phi_arr[k] + frac * (phi_arr[k + 1] - phi_arr[k])

            if r_inner <= rc <= r_outer:
                r_cross[count] = rc
                phi_cross[count] = pc
                theta_cross_dir[count] = 1.0 if d_prev > 0.0 else -1.0
                cross_idx[count] = k
                count += 1
                if count >= _MAX_CROSSINGS:
                    break

    return r_cross, phi_cross, cross_idx, count


@njit(cache=True)
def kerr_g_factor(r_hit, xi, a, m):
    """
    Factor g de Cunningham (1975) en Kerr: alias de disk_g_factor.

    xi = L_z/E es la constante conservada del foton (Bardeen), por lo que
    el g-factor es exacto sin necesitar la direccion local de emision.
    """
    return disk_g_factor(r_hit, xi, a, m)


@njit(cache=True)
def _kerr_limb_factor(r_arr, theta_arr, phi_arr, cross_idx, n_valid):
    """
    Limb darkening para un cruce ecuatorial en Kerr.

    Calcula la tangente del rayo en 3D esfericas y computa cos(theta_e).
    """
    k = cross_idx
    if k <= 0 or k >= n_valid - 1:
        return 1.0

    # Posicion en cartesianas para puntos k y k+1
    r0, th0, ph0 = r_arr[k], theta_arr[k], phi_arr[k]
    r1, th1, ph1 = r_arr[k + 1], theta_arr[k + 1], phi_arr[k + 1]

    x0 = r0 * np.sin(th0) * np.cos(ph0)
    y0 = r0 * np.sin(th0) * np.sin(ph0)
    z0 = r0 * np.cos(th0)

    x1 = r1 * np.sin(th1) * np.cos(ph1)
    y1 = r1 * np.sin(th1) * np.sin(ph1)
    z1 = r1 * np.cos(th1)

    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    d_len = np.sqrt(dx * dx + dy * dy + dz * dz)
    if d_len < 1e-12:
        return 1.0

    cos_theta_e = abs(dz / d_len)
    return limb_darkening(cos_theta_e)


@njit(cache=True)
def _kerr_disk_turbulence(r_hit, phi_hit, m, a, t):
    """Turbulencia del disco en Kerr con rotacion Kepleriana."""
    _TURB_AMPLITUDE = 0.35

    sqrt_m = np.sqrt(m)
    r32 = r_hit ** 1.5
    omega = sqrt_m / (r32 + a * sqrt_m)
    # Rotacion diferencial + shear espiral, periodico en azimut
    # (consistente con disk_turbulence y el shader)
    phi_s = phi_hit - omega * t + 1.5 * np.log(r_hit)

    noise_x = np.cos(phi_s) * 2.5 + np.log(r_hit) * 4.0
    noise_y = np.sin(phi_s) * 2.5
    turb = _fbm(noise_x, noise_y) * 2.0 - 1.0

    return 1.0 + _TURB_AMPLITUDE * turb


@njit(cache=True)
def _kerr_color_from_geodesic(
    r_arr, theta_arr, phi_arr, n_pts,
    r_cam, theta_cam, xi, a, m,
    r_inner, r_outer, base_temp, beaming_power, t, fate, f_pt_inv,
):
    """Calcula el color RGB para un pixel en Kerr."""
    r_cross, phi_cross, c_idx, n_cross = kerr_find_crossings(
        r_arr, theta_arr, phi_arr, n_pts, r_inner, r_outer,
    )

    if n_cross == 0:
        if fate == 1 and n_pts >= 2:  # rayo escapado -> estrellas
            # Direccion de propagacion: tangente entre los dos ultimos puntos
            k = n_pts - 1
            x0 = r_arr[k - 1] * np.sin(theta_arr[k - 1]) * np.cos(phi_arr[k - 1])
            y0 = r_arr[k - 1] * np.sin(theta_arr[k - 1]) * np.sin(phi_arr[k - 1])
            z0 = r_arr[k - 1] * np.cos(theta_arr[k - 1])
            x1 = r_arr[k] * np.sin(theta_arr[k]) * np.cos(phi_arr[k])
            y1 = r_arr[k] * np.sin(theta_arr[k]) * np.sin(phi_arr[k])
            z1 = r_arr[k] * np.cos(theta_arr[k])
            dx = x1 - x0
            dy = y1 - y0
            dz = z1 - z0
            d_len = np.sqrt(dx * dx + dy * dy + dz * dz)
            if d_len < 1e-12:
                return 0.0, 0.0, 0.0
            return _starfield(dx / d_len, dy / d_len, dz / d_len)
        return 0.0, 0.0, 0.0

    # Acumular cada cruce con su propio g-factor y color de cuerpo negro
    total_i = 0.0
    acc_r = 0.0
    acc_g = 0.0
    acc_b = 0.0
    for c in range(n_cross):
        weight = 1.0 if c == 0 else 0.5  # imagenes secundarias atenuadas
        g_c = disk_g_factor(r_cross[c], xi, a, m)
        # T_norm para el color; el BRILLO sigue al flujo F = T_norm^4
        # (Stefan-Boltzmann: la potencia radiada va como T^4)
        t_norm = page_thorne_emission(r_cross[c], r_inner, a, m, f_pt_inv)
        f_norm = t_norm ** 4
        limb = _kerr_limb_factor(r_arr, theta_arr, phi_arr, c_idx[c], n_pts)
        turb = _kerr_disk_turbulence(r_cross[c], phi_cross[c], m, a, t)
        i_c = f_norm * limb * turb * g_c ** beaming_power * weight
        if i_c <= 0.0:
            continue
        cr, cg, cb = blackbody_rgb(base_temp * t_norm * g_c)
        total_i += i_c
        acc_r += i_c * cr
        acc_g += i_c * cg
        acc_b += i_c * cb

    if total_i <= 0.0:
        return 0.0, 0.0, 0.0

    # Color promedio ponderado por intensidad de cada cruce
    inv_i = 1.0 / total_i
    cr = acc_r * inv_i
    cg = acc_g * inv_i
    cb = acc_b * inv_i

    # Tone mapping Reinhard; la compresion perceptual la hace el gamma del
    # display (una sola compresion — dos aplastarian el rango del flujo fisico)
    mapped = total_i / (1.0 + total_i)
    brightness = mapped

    return min(cr * brightness, 1.0), min(cg * brightness, 1.0), min(cb * brightness, 1.0)


@njit(parallel=True, cache=True)
def kerr_render_frame_parallel(
    xi_arr, eta_arr, beta_B_arr,
    r_cam, theta_cam, a, m,
    lam_max, n_steps,
    r_inner, r_outer, base_temp, beaming_power, t=0.0,
):
    """
    Renderiza un frame completo con el integrador Kerr en paralelo.

    A diferencia de Schwarzschild, NO hay simetria horizontal para explotar
    porque el spin rompe la simetria esferica. Cada pixel se integra
    independientemente.

    Args:
        xi_arr: constantes de Carter xi, forma (h, w)
        eta_arr: constantes de Carter eta, forma (h, w)
        beta_B_arr: coordenadas Bardeen verticales, forma (h, w)
        r_cam, theta_cam: posicion del observador
        a: parametro de spin
        m: masa
        lam_max: parametro afin maximo
        n_steps: pasos RK4
        r_inner, r_outer: limites del disco
        base_temp: temperatura base (K)
        beaming_power: exponente de beaming
        t: tiempo para turbulencia

    Returns:
        Imagen RGB, forma (h, w, 3), valores en [0, 1].
    """
    h = xi_arr.shape[0]
    w = xi_arr.shape[1]
    image = np.zeros((h, w, 3))

    # Normalizacion del perfil Page-Thorne (una vez por frame)
    f_pt_inv = page_thorne_norm_inv(r_inner, r_outer, a, m)

    for idx in prange(h * w):
        j = idx // w
        i = idx % w

        xi = xi_arr[j, i]
        eta = eta_arr[j, i]
        beta_B = beta_B_arr[j, i]

        # Integrar geodesica
        r_arr, theta_arr, phi_arr, n_valid, fate = trace_kerr_geodesic_rk4(
            xi, eta, r_cam, theta_cam, a, m, lam_max, n_steps, beta_B,
        )

        # Colorear
        cr, cg, cb = _kerr_color_from_geodesic(
            r_arr, theta_arr, phi_arr, n_valid,
            r_cam, theta_cam, xi, a, m,
            r_inner, r_outer, base_temp, beaming_power, t, fate, f_pt_inv,
        )
        image[j, i, 0] = cr
        image[j, i, 1] = cg
        image[j, i, 2] = cb

    return image


def warmup_kerr():
    """Compila las funciones JIT de Kerr con un render minimo."""
    xi = np.array([[0.1]])
    eta = np.array([[0.01]])
    beta_B = np.array([[0.1]])
    kerr_render_frame_parallel(
        xi, eta, beta_B,
        30.0, np.pi / 2.0 - 0.2, 0.0, 1.0,
        50.0, 100,
        6.0, 20.0, 4000.0, 3.0,
    )
