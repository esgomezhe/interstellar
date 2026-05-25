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
def limb_darkening(cos_theta_e, mu=0.5):
    """Limb darkening por scattering de electrones en la atmosfera del disco.

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
    """Calcula el factor de limb darkening para un cruce ecuatorial.

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
    """Textura de turbulencia del disco con rotacion diferencial.

    Modula la emision con ruido procedural que rota a velocidad kepleriana.
    """
    _TURB_AMPLITUDE = 0.2

    # Posicion 3D del punto para calcular angulo azimutal
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    hit_x = r * (cos_psi * e1[0] + sin_psi * e2[0])
    hit_y = r * (cos_psi * e1[1] + sin_psi * e2[1])
    azimuth = np.arctan2(hit_y, hit_x)

    # Rotacion diferencial kepleriana
    omega = np.sqrt(m / (r * r * r))
    phi_rot = azimuth - omega * t

    # Coordenadas de ruido
    noise_x = np.log(r) * 4.0
    noise_y = phi_rot * 3.0
    turb = _fbm(noise_x, noise_y) * 2.0 - 1.0

    return 1.0 + _TURB_AMPLITUDE * turb


@njit(cache=True)
def novikov_thorne_emission(r, r_isco):
    """Perfil de emision Novikov-Thorne para disco delgado en Schwarzschild.

    T(r) = (r_isco/r)^(3/4) * (1 - sqrt(r_isco/r))^(1/4)

    La condicion de torque cero en el ISCO hace que la emision caiga a cero
    exactamente en r = r_isco, creando un borde interior oscuro realista.
    """
    ratio = r_isco / r
    factor = 1.0 - np.sqrt(ratio)
    if factor <= 0.0:
        return 0.0
    return ratio ** 0.75 * factor ** 0.25


@njit(cache=True)
def _starfield(direction_x, direction_y, direction_z):
    """Campo estelar procedural para rayos escapados.

    Genera estrellas pseudo-aleatorias basadas en la direccion del rayo.
    """
    # Coordenadas esfericas
    theta_sky = np.arccos(max(-1.0, min(1.0, direction_z)))
    phi_sky = np.arctan2(direction_y, direction_x)

    cr, cg, cb = 0.0, 0.0, 0.0

    for layer in range(3):
        scale = 80.0 + layer * 60.0
        cx = phi_sky * scale
        cy = theta_sky * scale
        ix = int(np.floor(cx))
        iy = int(np.floor(cy))
        fx = cx - ix
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
    phi, r, n_pts, e1, e2, cam_pos,
    rs, m, r_inner, r_outer, base_temp, beaming_power, t, fate,
):
    """
    Calcula el color RGB de un pixel dada su geodesica pre-computada.

    Busca cruces con el disco, aplica efectos relativistas y colormap.
    Si el rayo escapa sin cruzar el disco, muestra campo estelar de fondo.
    """
    r_cross, psi_cross, c_idx, n_cross = find_crossings(
        phi, r, n_pts, e1[2], e2[2], r_inner, r_outer,
    )

    if n_cross == 0:
        if fate == 1:  # rayo escapado -> estrellas
            phi_final = phi[n_pts - 1]
            dx = np.cos(phi_final) * e1[0] + np.sin(phi_final) * e2[0]
            dy = np.cos(phi_final) * e1[1] + np.sin(phi_final) * e2[1]
            dz = np.cos(phi_final) * e1[2] + np.sin(phi_final) * e2[2]
            return _starfield(dx, dy, dz)
        return 0.0, 0.0, 0.0

    # Primer cruce (mas cercano a la camara)
    r_hit = r_cross[0]
    psi_hit = psi_cross[0]
    base_emission = novikov_thorne_emission(r_hit, r_inner)
    limb = _compute_limb_factor(psi_hit, e1, e2, phi, r, c_idx[0], n_pts)

    g = gravitational_redshift(r_hit, rs)
    D = doppler_factor(r_hit, psi_hit, e1, e2, cam_pos, m)
    g_d = g * D

    turb = disk_turbulence(r_hit, psi_hit, e1, e2, m, t)
    intensity = base_emission * limb * turb * g_d ** beaming_power

    # Contribucion de cruces adicionales (imagenes secundarias)
    for c in range(1, n_cross):
        g_extra = gravitational_redshift(r_cross[c], rs)
        D_extra = doppler_factor(r_cross[c], psi_cross[c], e1, e2, cam_pos, m)
        g_d_extra = g_extra * D_extra
        extra_emission = novikov_thorne_emission(r_cross[c], r_inner)
        limb_extra = _compute_limb_factor(psi_cross[c], e1, e2, phi, r, c_idx[c], n_pts)
        turb_extra = disk_turbulence(r_cross[c], psi_cross[c], e1, e2, m, t)
        intensity += extra_emission * limb_extra * turb_extra * g_d_extra ** beaming_power * 0.5

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
            rs, m, r_inner, r_outer, base_temp, beaming_power, t, fate,
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
                rs, m, r_inner, r_outer, base_temp, beaming_power, t, fate,
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
