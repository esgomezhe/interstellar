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
_T_BASE: float = 2200.0
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
                count += 1
                if count >= _MAX_CROSSINGS:
                    break

    return r_cross, psi_cross, count


# ---------------------------------------------------------------------------
# Efectos relativistas
# ---------------------------------------------------------------------------

@njit(cache=True)
def gravitational_redshift(r, rs):
    """g = sqrt(1 - rs/r). Valores cercanos al horizonte -> g ~ 0."""
    if r <= rs:
        return 0.0
    return np.sqrt(1.0 - rs / r)


@njit(cache=True)
def doppler_factor(r_hit, psi_hit, e1, e2, cam_pos, m):
    """
    Factor Doppler relativista en el punto de interseccion con el disco.

    D = 1 / (gamma * (1 - v . n_hat))
    D > 1: lado que se acerca (blueshift)
    D < 1: lado que se aleja (redshift)
    """
    cos_psi = np.cos(psi_hit)
    sin_psi = np.sin(psi_hit)

    # Posicion 3D del punto de impacto
    hit_x = r_hit * (cos_psi * e1[0] + sin_psi * e2[0])
    hit_y = r_hit * (cos_psi * e1[1] + sin_psi * e2[1])
    hit_z = r_hit * (cos_psi * e1[2] + sin_psi * e2[2])

    # Direccion hacia el observador
    to_x = cam_pos[0] - hit_x
    to_y = cam_pos[1] - hit_y
    to_z = cam_pos[2] - hit_z
    dist = np.sqrt(to_x * to_x + to_y * to_y + to_z * to_z)
    if dist < 1e-12:
        return 1.0
    nx = to_x / dist
    ny = to_y / dist
    nz = to_z / dist

    # Distancia en el plano x-y para la direccion azimutal
    r_xy = np.sqrt(hit_x * hit_x + hit_y * hit_y)
    if r_xy < 1e-12:
        return 1.0

    # Velocidad kepleriana en direccion azimutal (rotacion prograda)
    v_mag = np.sqrt(m / r_hit)
    phi_hat_x = -hit_y / r_xy
    phi_hat_y = hit_x / r_xy

    v_dot_n = v_mag * (phi_hat_x * nx + phi_hat_y * ny + 0.0 * nz)
    gamma = 1.0 / np.sqrt(1.0 - v_mag * v_mag)

    return 1.0 / (gamma * (1.0 - v_dot_n))


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
def _color_from_geodesic(
    phi, r, n_pts, e1, e2, cam_pos,
    rs, m, r_inner, r_outer, base_temp, beaming_power,
):
    """
    Calcula el color RGB de un pixel dada su geodesica pre-computada.

    Busca cruces con el disco, aplica efectos relativistas y colormap.
    """
    r_cross, psi_cross, n_cross = find_crossings(
        phi, r, n_pts, e1[2], e2[2], r_inner, r_outer,
    )

    if n_cross == 0:
        return 0.0, 0.0, 0.0

    # Primer cruce (mas cercano a la camara)
    r_hit = r_cross[0]
    psi_hit = psi_cross[0]
    base_emission = (r_inner / r_hit) ** 0.75

    g = gravitational_redshift(r_hit, rs)
    D = doppler_factor(r_hit, psi_hit, e1, e2, cam_pos, m)
    g_d = g * D

    intensity = base_emission * g_d ** beaming_power

    # Contribucion de cruces adicionales (imagenes secundarias)
    for c in range(1, n_cross):
        g_extra = gravitational_redshift(r_cross[c], rs)
        D_extra = doppler_factor(r_cross[c], psi_cross[c], e1, e2, cam_pos, m)
        g_d_extra = g_extra * D_extra
        extra_emission = (r_inner / r_cross[c]) ** 0.75
        intensity += extra_emission * g_d_extra ** beaming_power * 0.5

    if intensity > 1.0:
        intensity = 1.0
    if intensity <= 0.0:
        return 0.0, 0.0, 0.0

    # Color: temperatura observada -> RGB de cuerpo negro
    temp_obs = base_temp * g_d
    cr, cg, cb = blackbody_rgb(temp_obs)

    # Brillo con compresion sqrt para rango dinamico
    brightness = intensity ** 0.5

    cr = cr * brightness
    cg = cg * brightness
    cb = cb * brightness

    if cr > 1.0:
        cr = 1.0
    if cg > 1.0:
        cg = 1.0
    if cb > 1.0:
        cb = 1.0

    return cr, cg, cb


# ---------------------------------------------------------------------------
# Render paralelo con simetria horizontal
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def render_frame_parallel(
    b_arr, e1_arr, e2_arr, cam_pos,
    r_cam, rs, m, phi_max, n_steps,
    r_inner, r_outer, base_temp, beaming_power,
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
        cr, cg, cb = _color_from_geodesic(
            phi, r, n_valid, e1_left, e2_left, cam_pos,
            rs, m, r_inner, r_outer, base_temp, beaming_power,
        )
        image[j, i, 0] = cr
        image[j, i, 1] = cg
        image[j, i, 2] = cb

        # Pixel derecho (espejo) — solo si es diferente del izquierdo
        if i != i_mirror:
            e1_right = e1_arr[j, i_mirror]
            e2_right = e2_arr[j, i_mirror]
            cr, cg, cb = _color_from_geodesic(
                phi, r, n_valid, e1_right, e2_right, cam_pos,
                rs, m, r_inner, r_outer, base_temp, beaming_power,
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
        6.0, 20.0, 2200.0, 3.0,
    )
