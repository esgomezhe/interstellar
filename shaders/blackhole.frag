#version 330 core

/*
 * Fragment shader: ray tracer de agujero negro de Schwarzschild.
 *
 * Cada pixel traza una geodesica nula hacia atras desde la camara,
 * detecta cruces con el disco de acrecion, y aplica efectos relativistas.
 *
 * Port directo de numba_kernels.py a GLSL para ejecucion en GPU.
 */

in vec2 v_uv;
out vec4 frag_color;

// --- Uniforms de camara ---
uniform float u_r_cam;          // distancia radial de la camara
uniform float u_theta_cam;      // angulo polar (radianes)
uniform float u_phi_cam;        // angulo azimutal (radianes)
uniform float u_fov;            // campo de vision vertical (grados)
uniform vec2  u_resolution;     // (width, height)

// --- Uniforms fisicos ---
uniform float u_rs;             // radio de Schwarzschild (2.0)
uniform float u_m;              // masa (1.0)
uniform float u_r_inner;        // borde interno del disco (ISCO = 6.0)
uniform float u_r_outer;        // borde externo del disco (20.0)
uniform float u_base_temp;      // temperatura base (2200 K)
uniform float u_beaming_power;  // exponente beaming (3.0)

// --- Uniforms de render ---
uniform int   u_n_steps;        // pasos RK4 por geodesica
uniform float u_phi_max;        // angulo maximo de integracion
uniform float u_gamma;          // correccion gamma

// --- Constantes ---
const int MAX_CROSSINGS = 5;
const float PI = 3.14159265359;


// =========================================================================
// Base ortonormal local de la camara (coordenadas esfericas -> cartesianas)
// =========================================================================

vec3 cam_e_r(float theta, float phi) {
    float st = sin(theta), ct = cos(theta);
    float sp = sin(phi),   cp = cos(phi);
    return vec3(st * cp, st * sp, ct);
}

vec3 cam_e_theta(float theta, float phi) {
    float st = sin(theta), ct = cos(theta);
    float sp = sin(phi),   cp = cos(phi);
    return vec3(ct * cp, ct * sp, -st);
}

vec3 cam_e_phi(float theta, float phi) {
    float sp = sin(phi), cp = cos(phi);
    return vec3(-sp, cp, 0.0);
}


// =========================================================================
// RK4 step para u'' + u = (3/2) * rs * u^2
// =========================================================================

vec2 rk4_step(float u, float du, float rs, float dphi) {
    // k1
    float k1_u  = du;
    float k1_du = -u + 1.5 * rs * u * u;

    // k2
    float u2  = u  + 0.5 * dphi * k1_u;
    float du2 = du + 0.5 * dphi * k1_du;
    float k2_u  = du2;
    float k2_du = -u2 + 1.5 * rs * u2 * u2;

    // k3
    float u3  = u  + 0.5 * dphi * k2_u;
    float du3 = du + 0.5 * dphi * k2_du;
    float k3_u  = du3;
    float k3_du = -u3 + 1.5 * rs * u3 * u3;

    // k4
    float u4  = u  + dphi * k3_u;
    float du4 = du + dphi * k3_du;
    float k4_u  = du4;
    float k4_du = -u4 + 1.5 * rs * u4 * u4;

    float u_new  = u  + (dphi / 6.0) * (k1_u  + 2.0*k2_u  + 2.0*k3_u  + k4_u);
    float du_new = du + (dphi / 6.0) * (k1_du + 2.0*k2_du + 2.0*k3_du + k4_du);

    return vec2(u_new, du_new);
}


// =========================================================================
// Colormap de cuerpo negro (Tanner Helland)
// =========================================================================

vec3 blackbody_rgb(float temperature) {
    float temp = clamp(temperature, 1000.0, 40000.0) / 100.0;

    float red, green, blue;

    // Rojo
    if (temp <= 66.0) {
        red = 1.0;
    } else {
        red = 329.698727446 * pow(temp - 60.0, -0.1332047592) / 255.0;
    }

    // Verde
    if (temp <= 66.0) {
        green = (99.4708025861 * log(temp) - 161.1195681661) / 255.0;
    } else {
        green = 288.1221695283 * pow(temp - 60.0, -0.0755148492) / 255.0;
    }

    // Azul
    if (temp >= 66.0) {
        blue = 1.0;
    } else if (temp <= 19.0) {
        blue = 0.0;
    } else {
        blue = (138.5177312231 * log(temp - 10.0) - 305.0447927307) / 255.0;
    }

    return clamp(vec3(red, green, blue), 0.0, 1.0);
}


// =========================================================================
// Redshift gravitacional
// =========================================================================

float gravitational_redshift(float r, float rs) {
    if (r <= rs) return 0.0;
    return sqrt(1.0 - rs / r);
}


// =========================================================================
// Factor Doppler relativista
// =========================================================================

float doppler_factor(float r_hit, float psi_hit, vec3 e1, vec3 e2, vec3 cam_pos, float m) {
    float cos_psi = cos(psi_hit);
    float sin_psi = sin(psi_hit);

    // Posicion 3D del punto de impacto
    vec3 hit_pos = r_hit * (cos_psi * e1 + sin_psi * e2);

    // Direccion hacia el observador
    vec3 to_cam = cam_pos - hit_pos;
    float dist = length(to_cam);
    if (dist < 1e-12) return 1.0;
    vec3 n_hat = to_cam / dist;

    // Distancia en el plano x-y
    float r_xy = length(hit_pos.xy);
    if (r_xy < 1e-12) return 1.0;

    // Velocidad kepleriana azimutal (rotacion prograda)
    float v_mag = sqrt(m / r_hit);
    vec2 phi_hat = vec2(-hit_pos.y / r_xy, hit_pos.x / r_xy);

    float v_dot_n = v_mag * (phi_hat.x * n_hat.x + phi_hat.y * n_hat.y);
    float gamma_lorentz = 1.0 / sqrt(1.0 - v_mag * v_mag);

    return 1.0 / (gamma_lorentz * (1.0 - v_dot_n));
}


// =========================================================================
// Trazar geodesica y colorear pixel
// =========================================================================

vec3 trace_and_color(float b, vec3 e1, vec3 e2, vec3 cam_pos,
                     float r_cam, float rs, float m, float phi_max, int n_steps,
                     float r_inner, float r_outer, float base_temp, float beaming_power)
{
    // Condiciones iniciales
    float u0 = 1.0 / r_cam;
    float val = 1.0 / (b * b) - u0 * u0 + rs * u0 * u0 * u0;
    if (val < 0.0) val = 0.0;
    float du0 = sqrt(val);

    float dphi = phi_max / float(n_steps);
    float u_capture = 1.0 / rs;
    float u_escape = 0.5 / r_cam;

    float u  = u0;
    float du = du0;

    // Arrays para cruces ecuatoriales (tamano fijo en GLSL)
    float r_cross[MAX_CROSSINGS];
    float psi_cross[MAX_CROSSINGS];
    int n_cross = 0;

    // z previa para deteccion de cruce
    float phi_prev = 0.0;
    float r_prev = r_cam;
    float z_prev = r_prev * (cos(phi_prev) * e1.z + sin(phi_prev) * e2.z);

    // Integrar geodesica con RK4
    for (int k = 0; k < n_steps; k++) {
        vec2 result = rk4_step(u, du, rs, dphi);
        float u_new = result.x;
        float du_new = result.y;

        if (u_new <= 1e-12) u_new = 1e-12;
        float r_new = 1.0 / u_new;
        float phi_new = float(k + 1) * dphi;

        // Verificar captura
        if (u_new >= u_capture) break;

        // Verificar escape
        if (u_new < u_escape) break;

        // Deteccion de cruce ecuatorial (cambio de signo en z)
        float z_new = r_new * (cos(phi_new) * e1.z + sin(phi_new) * e2.z);

        if (z_prev * z_new < 0.0 && n_cross < MAX_CROSSINGS) {
            float az_prev = abs(z_prev);
            float az_new  = abs(z_new);
            float frac = az_prev / (az_prev + az_new);
            float rc = r_prev + frac * (r_new - r_prev);
            float pc = phi_prev + frac * (phi_new - phi_prev);

            if (rc >= r_inner && rc <= r_outer) {
                r_cross[n_cross] = rc;
                psi_cross[n_cross] = pc;
                n_cross++;
            }
        }

        phi_prev = phi_new;
        r_prev = r_new;
        z_prev = z_new;
        u = u_new;
        du = du_new;
    }

    // Sin cruces -> pixel negro
    if (n_cross == 0) return vec3(0.0);

    // Primer cruce (mas cercano a la camara)
    float r_hit = r_cross[0];
    float psi_hit = psi_cross[0];
    float base_emission = pow(r_inner / r_hit, 0.75);

    float g = gravitational_redshift(r_hit, rs);
    float D = doppler_factor(r_hit, psi_hit, e1, e2, cam_pos, m);
    float g_d = g * D;

    float intensity = base_emission * pow(g_d, beaming_power);

    // Cruces adicionales (imagenes secundarias)
    for (int c = 1; c < n_cross; c++) {
        float g_extra = gravitational_redshift(r_cross[c], rs);
        float D_extra = doppler_factor(r_cross[c], psi_cross[c], e1, e2, cam_pos, m);
        float g_d_extra = g_extra * D_extra;
        float extra_emission = pow(r_inner / r_cross[c], 0.75);
        intensity += extra_emission * pow(g_d_extra, beaming_power) * 0.5;
    }

    intensity = clamp(intensity, 0.0, 1.0);
    if (intensity <= 0.0) return vec3(0.0);

    // Color de cuerpo negro
    float temp_obs = base_temp * g_d;
    vec3 color = blackbody_rgb(temp_obs);

    // Brillo con compresion sqrt
    float brightness = sqrt(intensity);

    return clamp(color * brightness, 0.0, 1.0);
}


// =========================================================================
// Main
// =========================================================================

void main() {
    float width  = u_resolution.x;
    float height = u_resolution.y;

    // Indice de pixel desde UV
    float i = v_uv.x * width;
    float j = (1.0 - v_uv.y) * height;  // flip Y (OpenGL origin bottom-left)

    // Base ortonormal de la camara
    vec3 e_r     = cam_e_r(u_theta_cam, u_phi_cam);
    vec3 e_theta = cam_e_theta(u_theta_cam, u_phi_cam);
    vec3 e_phi   = cam_e_phi(u_theta_cam, u_phi_cam);

    // Posicion de la camara
    vec3 cam_pos = u_r_cam * e_r;

    // Direccion del rayo
    float fov_rad = radians(u_fov);
    float pixel_size = fov_rad / height;

    float alpha = (i - width / 2.0 + 0.5) * pixel_size;
    float beta  = (height / 2.0 - j - 0.5) * pixel_size;

    vec3 d = -e_r + tan(alpha) * e_phi + tan(beta) * (-e_theta);
    d = normalize(d);

    // Parametro de impacto: b = r_cam * |e_r x d|
    vec3 cross_rd = cross(e_r, d);
    float sin_psi = length(cross_rd);
    float b = u_r_cam * sin_psi;

    if (b < 1e-6) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Plano orbital: e1 = e_r, e2 = (n x e1) normalizado
    vec3 e1 = e_r;
    vec3 e2;

    if (sin_psi < 1e-12) {
        e2 = e_theta;
    } else {
        vec3 n = cross_rd / sin_psi;
        e2 = cross(n, e1);
        e2 = normalize(e2);
    }

    // Trazar geodesica y obtener color
    vec3 color = trace_and_color(
        b, e1, e2, cam_pos,
        u_r_cam, u_rs, u_m, u_phi_max, u_n_steps,
        u_r_inner, u_r_outer, u_base_temp, u_beaming_power
    );

    // Correccion gamma
    color = pow(color, vec3(u_gamma));

    frag_color = vec4(color, 1.0);
}
