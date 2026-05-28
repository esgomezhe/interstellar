#version 330 core

/*
 * Fragment shader: ray tracer de agujero negro (Schwarzschild + Kerr).
 *
 * Cada pixel traza una geodesica nula hacia atras desde la camara,
 * detecta cruces con el disco de acrecion, y aplica efectos relativistas.
 *
 * Cuando u_spin = 0: usa la ODE de Schwarzschild u'' + u = (3/2)*rs*u^2
 * Cuando u_spin > 0: usa las ecuaciones de Carter para Kerr (4 variables)
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
uniform float u_spin;           // parametro de spin a (0 = Schwarzschild)
uniform float u_r_inner;        // borde interno del disco (ISCO)
uniform float u_r_outer;        // borde externo del disco (20.0)
uniform float u_base_temp;      // temperatura base (2200 K)
uniform float u_beaming_power;  // exponente beaming (3.0)

// --- Uniforms de render ---
uniform int   u_n_steps;        // pasos RK4 por geodesica
uniform float u_phi_max;        // angulo maximo de integracion (Schwarzschild)
uniform float u_lam_max;        // parametro afin maximo (Kerr)
uniform float u_gamma;          // correccion gamma
uniform float u_time;           // tiempo para animacion del disco

// --- Constantes ---
const int MAX_CROSSINGS = 5;
const float PI = 3.14159265359;
const float TURB_AMPLITUDE = 0.2;

// Guardas numericas para Kerr (float32)
const float THETA_GUARD = 0.01;       // ~0.57 deg del polo, smoother pole handling
const float DELTA_GUARD = 1e-6;       // evita division por delta->0 en horizonte


// =========================================================================
// Ruido procedural (hash-based)
// =========================================================================

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise2d(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float val = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 4; i++) {
        val += amp * noise2d(p);
        p *= 2.0;
        amp *= 0.5;
    }
    return val;
}

float disk_turbulence(float r, float azimuth, float m, float a, float time) {
    float sqrt_m = sqrt(m);
    float omega = sqrt_m / (pow(r, 1.5) + a * sqrt_m);
    float phi_rot = azimuth - omega * time;
    vec2 noise_coord = vec2(log(r) * 4.0, phi_rot * 3.0);
    float turb = fbm(noise_coord) * 2.0 - 1.0;
    return 1.0 + TURB_AMPLITUDE * turb;
}


// =========================================================================
// Base ortonormal local de la camara
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
// Colormap de cuerpo negro (Tanner Helland)
// =========================================================================

vec3 blackbody_rgb(float temperature) {
    float temp = clamp(temperature, 1000.0, 40000.0) / 100.0;
    float red, green, blue;

    if (temp <= 66.0) { red = 1.0; }
    else { red = 329.698727446 * pow(temp - 60.0, -0.1332047592) / 255.0; }

    if (temp <= 66.0) { green = (99.4708025861 * log(temp) - 161.1195681661) / 255.0; }
    else { green = 288.1221695283 * pow(temp - 60.0, -0.0755148492) / 255.0; }

    if (temp >= 66.0) { blue = 1.0; }
    else if (temp <= 19.0) { blue = 0.0; }
    else { blue = (138.5177312231 * log(temp - 10.0) - 305.0447927307) / 255.0; }

    return clamp(vec3(red, green, blue), 0.0, 1.0);
}


// =========================================================================
// Funciones compartidas
// =========================================================================

float limb_darkening(float cos_theta_e) {
    float mu = 0.5;
    return (1.0 + mu * abs(cos_theta_e)) / (1.0 + mu);
}

float novikov_thorne_emission(float r, float r_isco) {
    float ratio = r_isco / r;
    float factor = 1.0 - sqrt(ratio);
    if (factor <= 0.0) return 0.0;
    return pow(ratio, 0.75) * pow(factor, 0.25);
}

vec3 starfield(vec3 direction) {
    float theta_sky = acos(clamp(direction.z, -1.0, 1.0));
    float phi_sky = atan(direction.y, direction.x);
    vec3 color = vec3(0.0);

    for (int layer = 0; layer < 3; layer++) {
        float scale = 80.0 + float(layer) * 60.0;
        vec2 cell = vec2(phi_sky, theta_sky) * scale;
        vec2 cell_id = floor(cell);
        vec2 cell_frac = fract(cell);

        float h1 = hash21(cell_id * (0.13 + float(layer) * 0.07));
        float h2 = hash21(cell_id * (0.27 + float(layer) * 0.11));
        float h3 = hash21(cell_id * (0.41 + float(layer) * 0.03));

        if (h1 > 0.85) {
            vec2 star_pos = vec2(h2, h3);
            float dist = length(cell_frac - star_pos);
            float brightness = smoothstep(0.05, 0.0, dist);
            float temp_star = 3000.0 + h2 * 20000.0;
            vec3 star_color = blackbody_rgb(temp_star);
            float magnitude = 0.3 + h3 * 0.7;
            color += star_color * brightness * magnitude;
        }
    }
    return color;
}


// =========================================================================
// SCHWARZSCHILD: RK4 para u'' + u = (3/2)*rs*u^2
// =========================================================================

vec2 schwarzschild_rk4_step(float u, float du, float rs, float dphi) {
    float k1_u  = du;
    float k1_du = -u + 1.5 * rs * u * u;

    float u2  = u  + 0.5 * dphi * k1_u;
    float du2 = du + 0.5 * dphi * k1_du;
    float k2_u  = du2;
    float k2_du = -u2 + 1.5 * rs * u2 * u2;

    float u3  = u  + 0.5 * dphi * k2_u;
    float du3 = du + 0.5 * dphi * k2_du;
    float k3_u  = du3;
    float k3_du = -u3 + 1.5 * rs * u3 * u3;

    float u4  = u  + dphi * k3_u;
    float du4 = du + dphi * k3_du;
    float k4_u  = du4;
    float k4_du = -u4 + 1.5 * rs * u4 * u4;

    float u_new  = u  + (dphi / 6.0) * (k1_u  + 2.0*k2_u  + 2.0*k3_u  + k4_u);
    float du_new = du + (dphi / 6.0) * (k1_du + 2.0*k2_du + 2.0*k3_du + k4_du);

    return vec2(u_new, du_new);
}


// Schwarzschild Doppler
float schwarzschild_doppler(float r_hit, float psi_hit, vec3 e1, vec3 e2, vec3 cam_pos, float m) {
    float cos_psi = cos(psi_hit);
    float sin_psi = sin(psi_hit);
    vec3 hit_pos = r_hit * (cos_psi * e1 + sin_psi * e2);
    vec3 to_cam = cam_pos - hit_pos;
    float dist = length(to_cam);
    if (dist < 1e-12) return 1.0;
    vec3 n_hat = to_cam / dist;

    float r_xy = length(hit_pos.xy);
    if (r_xy < 1e-12) return 1.0;

    float v_mag = sqrt(m / r_hit);
    vec2 phi_hat = vec2(-hit_pos.y / r_xy, hit_pos.x / r_xy);
    float v_dot_n = v_mag * (phi_hat.x * n_hat.x + phi_hat.y * n_hat.y);
    float gamma_lorentz = 1.0 / sqrt(1.0 - v_mag * v_mag);

    return 1.0 / (gamma_lorentz * (1.0 - v_dot_n));
}

float gravitational_redshift(float r, float rs) {
    if (r <= rs) return 0.0;
    return sqrt(1.0 - rs / r);
}


// Schwarzschild trace
vec3 trace_schwarzschild(float b, vec3 e1, vec3 e2, vec3 cam_pos,
    float r_cam, float rs, float m, float phi_max, int n_steps,
    float r_inner, float r_outer, float base_temp, float beaming_power, float time)
{
    float u0 = 1.0 / r_cam;
    float val = 1.0 / (b * b) - u0 * u0 + rs * u0 * u0 * u0;
    if (val < 0.0) val = 0.0;
    float du0 = sqrt(val);

    float dphi = phi_max / float(n_steps);
    float u_capture = 1.0 / rs;
    float u_escape = 0.5 / r_cam;

    float u = u0, du = du0;
    float r_cross[MAX_CROSSINGS];
    float psi_cross[MAX_CROSSINGS];
    float limb_factor[MAX_CROSSINGS];
    int n_cross = 0;
    int ray_fate = 0;
    float phi_final = 0.0;

    float phi_prev = 0.0, r_prev = r_cam;
    float z_prev = r_prev * (cos(phi_prev) * e1.z + sin(phi_prev) * e2.z);

    for (int k = 0; k < n_steps; k++) {
        vec2 result = schwarzschild_rk4_step(u, du, rs, dphi);
        float u_new = result.x, du_new = result.y;
        if (u_new <= 1e-12) u_new = 1e-12;
        float r_new = 1.0 / u_new;
        float phi_new = float(k + 1) * dphi;

        if (u_new >= u_capture) { ray_fate = 2; break; }
        if (u_new < u_escape) { ray_fate = 1; phi_final = phi_new; break; }

        float z_new = r_new * (cos(phi_new) * e1.z + sin(phi_new) * e2.z);
        if (z_prev * z_new < 0.0 && n_cross < MAX_CROSSINGS) {
            float az_prev = abs(z_prev), az_new = abs(z_new);
            float frac = az_prev / (az_prev + az_new);
            float rc = r_prev + frac * (r_new - r_prev);
            float pc = phi_prev + frac * (phi_new - phi_prev);
            if (rc >= r_inner && rc <= r_outer) {
                r_cross[n_cross] = rc;
                psi_cross[n_cross] = pc;
                vec3 p0 = r_prev * (cos(phi_prev) * e1 + sin(phi_prev) * e2);
                vec3 p1 = r_new * (cos(phi_new) * e1 + sin(phi_new) * e2);
                vec3 tangent = p1 - p0;
                float t_len = length(tangent);
                float cte = (t_len > 1e-12) ? abs(tangent.z / t_len) : 1.0;
                limb_factor[n_cross] = limb_darkening(cte);
                n_cross++;
            }
        }
        phi_prev = phi_new; r_prev = r_new; z_prev = z_new;
        u = u_new; du = du_new;
    }

    if (n_cross == 0) {
        if (ray_fate == 1) {
            vec3 d_final = cos(phi_final) * e1 + sin(phi_final) * e2;
            return starfield(d_final);
        }
        return vec3(0.0);
    }

    float r_hit = r_cross[0];
    float psi_hit = psi_cross[0];
    float base_emission = novikov_thorne_emission(r_hit, r_inner);
    float g = gravitational_redshift(r_hit, rs);
    float D = schwarzschild_doppler(r_hit, psi_hit, e1, e2, cam_pos, m);
    float g_d = g * D;

    vec3 hit_pos = r_hit * (cos(psi_hit) * e1 + sin(psi_hit) * e2);
    float azimuth = atan(hit_pos.y, hit_pos.x);
    float turb = disk_turbulence(r_hit, azimuth, m, 0.0, time);
    float intensity = base_emission * limb_factor[0] * turb * pow(g_d, beaming_power);

    for (int c = 1; c < n_cross; c++) {
        float g_e = gravitational_redshift(r_cross[c], rs);
        float D_e = schwarzschild_doppler(r_cross[c], psi_cross[c], e1, e2, cam_pos, m);
        float gd_e = g_e * D_e;
        float em_e = novikov_thorne_emission(r_cross[c], r_inner);
        vec3 hit_e = r_cross[c] * (cos(psi_cross[c]) * e1 + sin(psi_cross[c]) * e2);
        float az_e = atan(hit_e.y, hit_e.x);
        float turb_e = disk_turbulence(r_cross[c], az_e, m, 0.0, time);
        intensity += em_e * limb_factor[c] * turb_e * pow(gd_e, beaming_power) * 0.5;
    }

    intensity = clamp(intensity, 0.0, 1.0);
    if (intensity <= 0.0) return vec3(0.0);

    float temp_obs = base_temp * g_d;
    vec3 color = blackbody_rgb(temp_obs);
    return clamp(color * sqrt(intensity), 0.0, 1.0);
}


// =========================================================================
// KERR: integrador de Carter (6 variables)
// =========================================================================

// Kerr combined redshift + Doppler (g-factor)
// For a photon emitted by matter in Keplerian circular orbit in the equatorial
// plane of a Kerr BH, the frequency ratio g = nu_obs / nu_emit is:
//
//   g = 1 / [ gamma * (1 - omega * xi_local) ]
//
// where:
//   omega = M^{1/2} / (r^{3/2} + a M^{1/2})   Keplerian angular velocity
//   gamma = 1 / sqrt(1 - (r^2 + a^2) omega^2 + 2 a omega / r)   (Kerr Lorentz)
//   xi_local is the photon's conserved angular momentum per unit energy
//
// This properly combines gravitational redshift AND Doppler into one factor.
// Reference: Cunningham (1975), Luminet (1979)
//
float kerr_g_factor(float r_hit, float xi, float a, float m) {
    float sqrt_m = sqrt(m);
    float r32 = pow(r_hit, 1.5);
    float omega = sqrt_m / (r32 + a * sqrt_m);

    // Effective velocity squared in Kerr (from normalization of 4-velocity)
    float r2 = r_hit * r_hit;
    float v2 = (r2 + a * a) * omega * omega - 2.0 * a * omega * m / r_hit;
    v2 = clamp(v2, 0.0, 0.99);

    float gamma = 1.0 / sqrt(1.0 - v2);
    float g = 1.0 / (gamma * (1.0 - omega * xi));
    return clamp(g, 0.05, 5.0);
}


vec3 trace_kerr(float xi, float eta, float beta_B,
    float r_cam, float theta_cam, float phi_cam, float a, float m,
    float lam_max, int n_steps,
    float r_inner, float r_outer, float base_temp, float beaming_power, float time)
{
    // Condiciones iniciales
    float cos_t0 = cos(theta_cam);
    float sin_t0 = sin(theta_cam);
    if (abs(sin_t0) < THETA_GUARD) sin_t0 = THETA_GUARD;

    float delta0 = r_cam * r_cam - 2.0 * m * r_cam + a * a;
    float P0 = r_cam * r_cam + a * a - a * xi;
    float xi_a = xi - a;

    float R0 = P0 * P0 - delta0 * (eta + xi_a * xi_a);
    if (R0 < 0.0) R0 = 0.0;
    float pr_ = -sqrt(R0);  // hacia adentro

    float cot_t0 = cos_t0 / sin_t0;
    float Theta0 = eta + a * a * cos_t0 * cos_t0 - xi * xi * cot_t0 * cot_t0;
    if (Theta0 < 0.0) Theta0 = 1e-4;
    // p_theta = -beta_B para backward ray tracing
    float sqrt_Theta0 = sqrt(Theta0);
    float pt_ = (beta_B > 0.0) ? -sqrt_Theta0 : sqrt_Theta0;

    // Horizonte exterior
    float disc_h = m * m - a * a;
    if (disc_h < 0.0) disc_h = 0.0;
    float r_plus = m + sqrt(disc_h);

    // Paso de integracion base (se escala adaptativamente en el loop)
    float dlam_base = lam_max / float(n_steps);

    // El integrador trabaja con phi relativo al rayo (empieza en 0)
    // phi_cam se suma al final para obtener coordenadas globales
    float r_ = r_cam;
    float th_ = theta_cam;
    float ph_ = 0.0;

    // Cruces ecuatoriales
    float r_cross[MAX_CROSSINGS];
    float phi_cross[MAX_CROSSINGS];
    float limb_fac[MAX_CROSSINGS];
    int n_cross = 0;
    int ray_fate = 0;  // 0=orbiting, 1=escaped, 2=captured

    float th_prev = theta_cam;
    float r_prev = r_cam;
    float ph_prev = 0.0;
    float half_pi = PI / 2.0;
    float eta_xi_a2 = eta + xi_a * xi_a;

    for (int k = 0; k < n_steps; k++) {
        // Adaptive step: smaller near BH, larger far away
        float step_scale = clamp(sqrt(max(r_, 1.0) / r_cam), 0.01, 2.0);
        float dlam = dlam_base * step_scale;

        // === k1 ===
        float cos_t, sin_t, sigma, delta, P, inv_sig;

        cos_t = cos(th_); sin_t = sin(th_);
        if (sin_t < THETA_GUARD && sin_t >= 0.0) sin_t = THETA_GUARD;
        if (sin_t > -THETA_GUARD && sin_t < 0.0) sin_t = -THETA_GUARD;
        sigma = r_ * r_ + a * a * cos_t * cos_t;
        delta = r_ * r_ - 2.0 * m * r_ + a * a;
        if (abs(delta) < DELTA_GUARD) delta = DELTA_GUARD;
        P = r_ * r_ + a * a - a * xi;
        inv_sig = 1.0 / sigma;

        float dr1 = pr_ * inv_sig;
        float dth1 = pt_ * inv_sig;
        float sin2 = sin_t * sin_t;
        float dph1 = (-a + a * P / delta + xi / sin2) * inv_sig;
        float dpr1 = 0.5 * (4.0 * r_ * P - (2.0 * r_ - 2.0 * m) * eta_xi_a2) * inv_sig;
        float dpt1 = 0.5 * (-2.0 * a * a * cos_t * sin_t + 2.0 * xi * xi * cos_t / (sin2 * sin_t)) * inv_sig;

        // === k2 ===
        float r2 = r_ + 0.5 * dlam * dr1;
        float th2 = clamp(th_ + 0.5 * dlam * dth1, THETA_GUARD, PI - THETA_GUARD);
        float pr2 = pr_ + 0.5 * dlam * dpr1;
        float pt2 = pt_ + 0.5 * dlam * dpt1;

        cos_t = cos(th2); sin_t = sin(th2);
        if (sin_t < THETA_GUARD && sin_t >= 0.0) sin_t = THETA_GUARD;
        if (sin_t > -THETA_GUARD && sin_t < 0.0) sin_t = -THETA_GUARD;
        sigma = r2 * r2 + a * a * cos_t * cos_t;
        delta = r2 * r2 - 2.0 * m * r2 + a * a;
        if (abs(delta) < DELTA_GUARD) delta = DELTA_GUARD;
        P = r2 * r2 + a * a - a * xi;
        inv_sig = 1.0 / sigma;
        sin2 = sin_t * sin_t;

        float dr2 = pr2 * inv_sig;
        float dth2 = pt2 * inv_sig;
        float dph2 = (-a + a * P / delta + xi / sin2) * inv_sig;
        float dpr2 = 0.5 * (4.0 * r2 * P - (2.0 * r2 - 2.0 * m) * eta_xi_a2) * inv_sig;
        float dpt2 = 0.5 * (-2.0 * a * a * cos_t * sin_t + 2.0 * xi * xi * cos_t / (sin2 * sin_t)) * inv_sig;

        // === k3 ===
        float r3 = r_ + 0.5 * dlam * dr2;
        float th3 = clamp(th_ + 0.5 * dlam * dth2, THETA_GUARD, PI - THETA_GUARD);
        float pr3 = pr_ + 0.5 * dlam * dpr2;
        float pt3 = pt_ + 0.5 * dlam * dpt2;

        cos_t = cos(th3); sin_t = sin(th3);
        if (sin_t < THETA_GUARD && sin_t >= 0.0) sin_t = THETA_GUARD;
        if (sin_t > -THETA_GUARD && sin_t < 0.0) sin_t = -THETA_GUARD;
        sigma = r3 * r3 + a * a * cos_t * cos_t;
        delta = r3 * r3 - 2.0 * m * r3 + a * a;
        if (abs(delta) < DELTA_GUARD) delta = DELTA_GUARD;
        P = r3 * r3 + a * a - a * xi;
        inv_sig = 1.0 / sigma;
        sin2 = sin_t * sin_t;

        float dr3 = pr3 * inv_sig;
        float dth3 = pt3 * inv_sig;
        float dph3 = (-a + a * P / delta + xi / sin2) * inv_sig;
        float dpr3 = 0.5 * (4.0 * r3 * P - (2.0 * r3 - 2.0 * m) * eta_xi_a2) * inv_sig;
        float dpt3 = 0.5 * (-2.0 * a * a * cos_t * sin_t + 2.0 * xi * xi * cos_t / (sin2 * sin_t)) * inv_sig;

        // === k4 ===
        float r4 = r_ + dlam * dr3;
        float th4 = clamp(th_ + dlam * dth3, THETA_GUARD, PI - THETA_GUARD);
        float pr4 = pr_ + dlam * dpr3;
        float pt4 = pt_ + dlam * dpt3;

        cos_t = cos(th4); sin_t = sin(th4);
        if (sin_t < THETA_GUARD && sin_t >= 0.0) sin_t = THETA_GUARD;
        if (sin_t > -THETA_GUARD && sin_t < 0.0) sin_t = -THETA_GUARD;
        sigma = r4 * r4 + a * a * cos_t * cos_t;
        delta = r4 * r4 - 2.0 * m * r4 + a * a;
        if (abs(delta) < DELTA_GUARD) delta = DELTA_GUARD;
        P = r4 * r4 + a * a - a * xi;
        inv_sig = 1.0 / sigma;
        sin2 = sin_t * sin_t;

        float dr4 = pr4 * inv_sig;
        float dth4 = pt4 * inv_sig;
        float dph4 = (-a + a * P / delta + xi / sin2) * inv_sig;
        float dpr4 = 0.5 * (4.0 * r4 * P - (2.0 * r4 - 2.0 * m) * eta_xi_a2) * inv_sig;
        float dpt4 = 0.5 * (-2.0 * a * a * cos_t * sin_t + 2.0 * xi * xi * cos_t / (sin2 * sin_t)) * inv_sig;

        // === Combinar RK4 ===
        float s6 = dlam / 6.0;
        float r_new = r_ + s6 * (dr1 + 2.0*dr2 + 2.0*dr3 + dr4);
        float th_new = th_ + s6 * (dth1 + 2.0*dth2 + 2.0*dth3 + dth4);
        float pr_new = pr_ + s6 * (dpr1 + 2.0*dpr2 + 2.0*dpr3 + dpr4);
        float pt_new = pt_ + s6 * (dpt1 + 2.0*dpt2 + 2.0*dpt3 + dpt4);
        float ph_new = ph_ + s6 * (dph1 + 2.0*dph2 + 2.0*dph3 + dph4);

        // Clamp theta con reflexion de momento
        if (th_new < THETA_GUARD) { th_new = THETA_GUARD; pt_new = abs(pt_new); }
        if (th_new > PI - THETA_GUARD) { th_new = PI - THETA_GUARD; pt_new = -abs(pt_new); }

        // Captura
        if (r_new <= r_plus * 1.01) { ray_fate = 2; break; }

        // Escape: ray past camera and moving outward
        if (r_new > r_cam && pr_new > 0.0) { ray_fate = 1; break; }

        // Deteccion de cruce ecuatorial (theta cruza pi/2)
        float d_prev = th_prev - half_pi;
        float d_new = th_new - half_pi;
        if (d_prev * d_new < 0.0 && n_cross < MAX_CROSSINGS) {
            float frac = abs(d_prev) / (abs(d_prev) + abs(d_new));
            float rc = r_prev + frac * (r_new - r_prev);
            float pc = ph_prev + frac * (ph_new - ph_prev);

            if (rc >= r_inner && rc <= r_outer) {
                r_cross[n_cross] = rc;
                phi_cross[n_cross] = pc;

                // Limb darkening via tangente 3D
                vec3 p0 = vec3(r_prev * sin(th_prev) * cos(ph_prev),
                               r_prev * sin(th_prev) * sin(ph_prev),
                               r_prev * cos(th_prev));
                vec3 p1 = vec3(r_new * sin(th_new) * cos(ph_new),
                               r_new * sin(th_new) * sin(ph_new),
                               r_new * cos(th_new));
                vec3 tangent = p1 - p0;
                float t_len = length(tangent);
                float cte = (t_len > 1e-8) ? abs(tangent.z / t_len) : 1.0;
                limb_fac[n_cross] = limb_darkening(cte);
                n_cross++;
            }
        }

        th_prev = th_new; r_prev = r_new; ph_prev = ph_new;
        r_ = r_new; th_ = th_new; pr_ = pr_new; pt_ = pt_new; ph_ = ph_new;
    }

    // === Colorear ===
    if (n_cross == 0) {
        if (ray_fate == 1) {
            // Rayo escapado: direccion final en coordenadas globales
            float ph_global = ph_ + phi_cam;
            float sf = max(sin(th_), THETA_GUARD);
            vec3 d_final = vec3(sf * cos(ph_global), sf * sin(ph_global), cos(th_));
            return starfield(normalize(d_final));
        }
        return vec3(0.0);
    }

    // Primer cruce: colorear con Kerr g-factor
    float r_hit = r_cross[0];
    float phi_ray = phi_cross[0];           // phi en coordenadas del rayo
    float phi_global = phi_ray + phi_cam;   // phi en coordenadas globales
    float base_emission = novikov_thorne_emission(r_hit, r_inner);

    // g-factor combina redshift gravitacional + Doppler de Kerr
    float g_d = kerr_g_factor(r_hit, xi, a, m);

    // Azimuth global para turbulencia y rotacion del disco
    float azimuth = phi_global;
    float turb = disk_turbulence(r_hit, azimuth, m, a, time);
    float intensity = base_emission * limb_fac[0] * turb * pow(max(g_d, 0.01), beaming_power);

    for (int c = 1; c < n_cross; c++) {
        float g_e = kerr_g_factor(r_cross[c], xi, a, m);
        float em_e = novikov_thorne_emission(r_cross[c], r_inner);
        float az_e = phi_cross[c] + phi_cam;
        float turb_e = disk_turbulence(r_cross[c], az_e, m, a, time);
        intensity += em_e * limb_fac[c] * turb_e * pow(max(g_e, 0.01), beaming_power) * 0.5;
    }

    intensity = clamp(intensity, 0.0, 1.0);
    if (intensity <= 0.0) return vec3(0.0);

    float temp_obs = base_temp * max(g_d, 0.1);
    vec3 color = blackbody_rgb(temp_obs);
    return clamp(color * sqrt(intensity), 0.0, 1.0);
}


// =========================================================================
// Main
// =========================================================================

void main() {
    float width  = u_resolution.x;
    float height = u_resolution.y;
    float i = v_uv.x * width;
    float j = (1.0 - v_uv.y) * height;

    vec3 e_r     = cam_e_r(u_theta_cam, u_phi_cam);
    vec3 e_theta = cam_e_theta(u_theta_cam, u_phi_cam);
    vec3 e_phi   = cam_e_phi(u_theta_cam, u_phi_cam);
    vec3 cam_pos = u_r_cam * e_r;

    float fov_rad = radians(u_fov);
    float pixel_size = fov_rad / height;

    float alpha_pix = (i - width / 2.0 + 0.5) * pixel_size;
    float beta_pix  = (height / 2.0 - j - 0.5) * pixel_size;

    vec3 color;

    if (u_spin < 1e-6) {
        // === SCHWARZSCHILD ===
        vec3 d = -e_r + tan(alpha_pix) * e_phi + tan(beta_pix) * (-e_theta);
        d = normalize(d);
        vec3 cross_rd = cross(e_r, d);
        float sin_psi = length(cross_rd);
        float b = u_r_cam * sin_psi;

        if (b < 1e-6) {
            frag_color = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }

        vec3 e1 = e_r;
        vec3 e2;
        if (sin_psi < 1e-12) { e2 = e_theta; }
        else {
            vec3 n = cross_rd / sin_psi;
            e2 = normalize(cross(n, e1));
        }

        color = trace_schwarzschild(b, e1, e2, cam_pos,
            u_r_cam, u_rs, u_m, u_phi_max, u_n_steps,
            u_r_inner, u_r_outer, u_base_temp, u_beaming_power, u_time);
    }
    else {
        // === KERR ===
        float alpha_B = u_r_cam * tan(alpha_pix);
        float beta_B  = u_r_cam * tan(beta_pix);

        float sin_t = sin(u_theta_cam);
        float cos_t = cos(u_theta_cam);
        if (abs(sin_t) < 1e-12) sin_t = 1e-12;
        float cot_t = cos_t / sin_t;

        float xi  = -alpha_B * sin_t;
        float eta = beta_B * beta_B - u_spin * u_spin * cos_t * cos_t + xi * xi * cot_t * cot_t;

        color = trace_kerr(xi, eta, beta_B,
            u_r_cam, u_theta_cam, u_phi_cam, u_spin, u_m,
            u_lam_max, u_n_steps,
            u_r_inner, u_r_outer, u_base_temp, u_beaming_power, u_time);
    }

    // Correccion gamma
    color = pow(color, vec3(u_gamma));
    frag_color = vec4(color, 1.0);
}
