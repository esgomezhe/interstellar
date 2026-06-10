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
uniform vec2  u_jitter;         // offset subpixel para supersampling temporal
uniform float u_pt_fmax_inv;    // 1/max(F) del perfil Page-Thorne (normaliza)

// --- Constantes ---
const int MAX_CROSSINGS = 5;
const float PI = 3.14159265359;
const float TURB_AMPLITUDE = 0.35;

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
    // Cada anillo rota a su velocidad Kepleriana (rotacion diferencial);
    // el termino log(r) tuerce el patron en brazos espirales arrastrados
    float phi_s = azimuth - omega * time + 1.5 * log(r);
    // Muestrear el ruido sobre un circulo lo hace periodico en azimut
    // (sin costura en phi = +-pi); log(r) desplaza el circulo radialmente
    vec2 noise_coord = vec2(cos(phi_s), sin(phi_s)) * 2.5
                     + vec2(log(r) * 4.0, 0.0);
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

// Perfil de temperatura normalizado del disco delgado relativista
// (Page & Thorne 1974, ec. 15n): T_norm = (F/F_max)^(1/4).
// Version EXACTA y dependiente del spin del Novikov-Thorne aproximado.
// Con x = sqrt(r/M), x0 = sqrt(r_isco/M) y x1,x2,x3 raices de x^3-3x+2a*=0:
//   F ∝ [x - x0 - (3a*/2)ln(x/x0) - Σ cᵢ ln((x-xᵢ)/(x0-xᵢ))] / [x⁴(x³-3x+2a*)]
//   cᵢ = 3(xᵢ-a*)² / [xᵢ ∏_{j≠i}(xᵢ-xⱼ)]
// La normalizacion u_pt_fmax_inv se calcula en CPU una vez por cambio de spin.
float page_thorne_emission(float r, float r_isco, float a, float m) {
    if (r <= r_isco) return 0.0;
    float a_star = a / m;
    float x  = sqrt(r / m);
    float x0 = sqrt(r_isco / m);

    float acs = acos(clamp(a_star, 0.0, 1.0));
    float x1 = 2.0 * cos((acs - PI) / 3.0);
    float x2 = 2.0 * cos((acs + PI) / 3.0);
    float x3 = -2.0 * cos(acs / 3.0);

    float d1 = x1 * (x1 - x2) * (x1 - x3);
    float d2 = x2 * (x2 - x1) * (x2 - x3);
    float d3 = x3 * (x3 - x1) * (x3 - x2);
    float c1 = 3.0 * (x1 - a_star) * (x1 - a_star) / d1;
    // a* = 0: x2 -> 0 y el limite de c2 es 0 (numerador ~ x2²)
    float c2 = (abs(d2) < 1e-9) ? 0.0 : 3.0 * (x2 - a_star) * (x2 - a_star) / d2;
    float c3 = 3.0 * (x3 - a_star) * (x3 - a_star) / d3;

    float bracket = x - x0
                  - 1.5 * a_star * log(x / x0)
                  - c1 * log((x - x1) / (x0 - x1))
                  - c2 * log(max(x - x2, 1e-9) / max(x0 - x2, 1e-9))
                  - c3 * log((x - x3) / (x0 - x3));
    float flux = bracket / (x*x*x*x * (x*x*x - 3.0*x + 2.0*a_star));
    // f_norm > 1 cerca del pico (nucleo HDR): Reinhard absorbe el exceso
    float f_norm = max(flux * u_pt_fmax_inv, 0.0);
    return pow(f_norm, 0.25);
}

vec3 starfield(vec3 direction) {
    float theta_sky = acos(clamp(direction.z, -1.0, 1.0));
    float phi_sky = atan(direction.y, direction.x);
    vec3 color = vec3(0.0);

    for (int layer = 0; layer < 3; layer++) {
        float scale = 80.0 + float(layer) * 60.0;
        // Numero ENTERO de celdas en phi: el grid es periodico y no hay
        // costura en el corte de rama de atan (phi = +-pi)
        float n_phi = floor(2.0 * PI * scale);
        float u = (phi_sky + PI) / (2.0 * PI) * n_phi;
        float v = theta_sky * scale;
        vec2 cell_id = vec2(mod(floor(u), n_phi), floor(v));
        vec2 cell_frac = vec2(fract(u), fract(v));

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


// Factor g = nu_obs/nu_em exacto (Cunningham 1975) para emisor en orbita
// circular Kepleriana prograda ecuatorial:
//
//   g = sqrt(1 - 3M/r + 2a*sqrt(M)/r^(3/2))
//       / [(1 + a*sqrt(M)/r^(3/2)) * (1 - Omega*lam)]
//
// El numerador con (1 + a*sqrt(M)/r^(3/2)) es 1/u^t del emisor (Bardeen-
// Press-Teukolsky 1972): redshift gravitacional + Doppler transversal.
// (1 - Omega*lam) es el Doppler azimutal, con lam = L_z/E del foton
// (cantidad conservada: xi en Kerr, -b*n_z en Schwarzschild).
// Con a = 0 reduce a g = sqrt(1 - 3M/r) / (1 - Omega*lam).
float disk_g_factor(float r_hit, float lam, float a, float m) {
    float sqrt_m = sqrt(m);
    float r32 = pow(r_hit, 1.5);
    float omega = sqrt_m / (r32 + a * sqrt_m);

    float x = a * sqrt_m / r32;
    float B = max(1.0 - 3.0 * m / r_hit + 2.0 * x, 1e-6);

    float g = sqrt(B) / ((1.0 + x) * (1.0 - omega * lam));
    return clamp(g, 0.05, 5.0);
}


// Schwarzschild trace
// lam = L_z/E del foton (conservada): lam = -b * n_z, con n = e1 x e2.
vec3 trace_schwarzschild(float b, float lam, vec3 e1, vec3 e2,
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
    vec3 esc_dir = vec3(0.0);

    float phi_prev = 0.0, r_prev = r_cam;
    float z_prev = r_prev * (cos(phi_prev) * e1.z + sin(phi_prev) * e2.z);

    for (int k = 0; k < n_steps; k++) {
        vec2 result = schwarzschild_rk4_step(u, du, rs, dphi);
        float u_new = result.x, du_new = result.y;
        if (u_new <= 1e-12) u_new = 1e-12;
        float r_new = 1.0 / u_new;
        float phi_new = float(k + 1) * dphi;

        if (u_new >= u_capture) { ray_fate = 2; break; }
        if (u_new < u_escape) {
            ray_fate = 1;
            // Direccion de propagacion: tangente del ultimo segmento
            vec3 p0 = r_prev * (cos(phi_prev) * e1 + sin(phi_prev) * e2);
            vec3 p1 = r_new * (cos(phi_new) * e1 + sin(phi_new) * e2);
            esc_dir = normalize(p1 - p0);
            break;
        }

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
                // Buffer lleno: el destino del rayo ya no afecta el color
                if (n_cross >= MAX_CROSSINGS) break;
            }
        }
        phi_prev = phi_new; r_prev = r_new; z_prev = z_new;
        u = u_new; du = du_new;
    }

    if (n_cross == 0) {
        if (ray_fate == 1) return starfield(esc_dir);
        return vec3(0.0);
    }

    // Acumular cada cruce con su propio g-factor y color de cuerpo negro
    float total_i = 0.0;
    vec3 acc = vec3(0.0);
    for (int c = 0; c < n_cross; c++) {
        float weight = (c == 0) ? 1.0 : 0.5;  // imagenes secundarias atenuadas
        float g_c = disk_g_factor(r_cross[c], lam, 0.0, m);
        // T_norm para el color; el BRILLO sigue al flujo F = T_norm^4
        float t_norm = page_thorne_emission(r_cross[c], r_inner, 0.0, m);
        float f_norm = t_norm * t_norm * t_norm * t_norm;
        vec3 hit_c = r_cross[c] * (cos(psi_cross[c]) * e1 + sin(psi_cross[c]) * e2);
        float az_c = atan(hit_c.y, hit_c.x);
        float turb_c = disk_turbulence(r_cross[c], az_c, m, 0.0, time);
        float i_c = f_norm * limb_factor[c] * turb_c * pow(g_c, beaming_power) * weight;
        if (i_c <= 0.0) continue;
        acc += i_c * blackbody_rgb(base_temp * t_norm * g_c);
        total_i += i_c;
    }

    if (total_i <= 0.0) return vec3(0.0);

    // Color promedio ponderado + tone mapping Reinhard. La compresion
    // perceptual la hace u_gamma (una sola compresion — dos aplastarian
    // el rango dinamico del flujo fisico de Page-Thorne)
    vec3 color = acc / total_i;
    float mapped = total_i / (1.0 + total_i);
    return clamp(color * mapped, 0.0, 1.0);
}


// =========================================================================
// KERR: integrador de Carter (6 variables)
// =========================================================================

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
    if (Theta0 < 0.0) Theta0 = 0.0;
    // p_theta = -beta_B para backward ray tracing
    float sqrt_Theta0 = sqrt(Theta0);
    float pt_;
    if (beta_B > 0.0) pt_ = -sqrt_Theta0;
    else if (beta_B < 0.0) pt_ = sqrt_Theta0;
    else pt_ = (theta_cam < PI / 2.0) ? sqrt_Theta0 : -sqrt_Theta0;

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
    vec3 esc_dir = vec3(0.0);

    float th_prev = theta_cam;
    float r_prev = r_cam;
    float ph_prev = 0.0;
    float half_pi = PI / 2.0;
    float eta_xi_a2 = eta + xi_a * xi_a;

    for (int k = 0; k < n_steps; k++) {
        // Paso adaptativo: escala sqrt con la distancia, refinado cerca del
        // anillo de fotones donde rayos vecinos divergen exponencialmente
        float step_scale = clamp(sqrt(max(r_, 1.0) / r_cam), 0.01, 2.0);
        step_scale *= clamp((r_ - r_plus) / (3.0 * m), 0.08, 1.0);
        // Refinar cerca de los polos: los rayos casi meridionales tienen un
        // turning point en theta (Theta->0) y el azimut barre pi rapidamente;
        // un paso grueso se pasa del turning y decorrelaciona el cruce
        step_scale *= clamp(sin(th_) / 0.15, 0.1, 1.0);
        float dlam = dlam_base * step_scale;

        // === k1 ===
        float cos_t, sin_t, sigma, delta, P, inv_sig;

        cos_t = cos(th_); sin_t = sin(th_);
        if (sin_t < THETA_GUARD && sin_t >= 0.0) sin_t = THETA_GUARD;
        if (sin_t > -THETA_GUARD && sin_t < 0.0) sin_t = -THETA_GUARD;
        sigma = r_ * r_ + a * a * cos_t * cos_t;
        sigma = max(sigma, 1e-6);  // guard: evita Sigma=0 en singularidad del anillo
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
        sigma = max(sigma, 1e-6);
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
        sigma = max(sigma, 1e-6);
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
        sigma = max(sigma, 1e-6);
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

        // Paso polar: reflexion de momento + salto de PI en phi (el rayo
        // pasa sobre el polo y continua en el meridiano opuesto)
        if (th_new < THETA_GUARD) {
            th_new = THETA_GUARD; pt_new = abs(pt_new); ph_new += PI;
        }
        if (th_new > PI - THETA_GUARD) {
            th_new = PI - THETA_GUARD; pt_new = -abs(pt_new); ph_new += PI;
        }

        // Captura
        if (r_new <= r_plus * 1.01) { ray_fate = 2; break; }

        // Escape: ray past camera and moving outward
        if (r_new > r_cam && pr_new > 0.0) {
            ray_fate = 1;
            // Direccion de propagacion: tangente del ultimo segmento (global)
            float pg0 = ph_ + phi_cam;
            float pg1 = ph_new + phi_cam;
            vec3 p0 = vec3(r_ * sin(th_) * cos(pg0),
                           r_ * sin(th_) * sin(pg0),
                           r_ * cos(th_));
            vec3 p1 = vec3(r_new * sin(th_new) * cos(pg1),
                           r_new * sin(th_new) * sin(pg1),
                           r_new * cos(th_new));
            esc_dir = normalize(p1 - p0);
            break;
        }

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
                // Buffer lleno: el destino del rayo ya no afecta el color
                if (n_cross >= MAX_CROSSINGS) break;
            }
        }

        th_prev = th_new; r_prev = r_new; ph_prev = ph_new;
        r_ = r_new; th_ = th_new; pr_ = pr_new; pt_ = pt_new; ph_ = ph_new;
    }

    // === Colorear ===
    if (n_cross == 0) {
        if (ray_fate == 1) return starfield(esc_dir);
        return vec3(0.0);
    }

    // Acumular cada cruce con su propio g-factor y color de cuerpo negro
    float total_i = 0.0;
    vec3 acc = vec3(0.0);
    for (int c = 0; c < n_cross; c++) {
        float weight = (c == 0) ? 1.0 : 0.5;  // imagenes secundarias atenuadas
        float g_c = disk_g_factor(r_cross[c], xi, a, m);
        // T_norm para el color; el BRILLO sigue al flujo F = T_norm^4
        float t_norm = page_thorne_emission(r_cross[c], r_inner, a, m);
        float f_norm = t_norm * t_norm * t_norm * t_norm;
        float az_c = phi_cross[c] + phi_cam;  // azimuth global para turbulencia
        float turb_c = disk_turbulence(r_cross[c], az_c, m, a, time);
        float i_c = f_norm * limb_fac[c] * turb_c * pow(g_c, beaming_power) * weight;
        if (i_c <= 0.0) continue;
        acc += i_c * blackbody_rgb(base_temp * t_norm * g_c);
        total_i += i_c;
    }

    if (total_i <= 0.0) return vec3(0.0);

    // Color promedio ponderado + tone mapping Reinhard. La compresion
    // perceptual la hace u_gamma (una sola compresion — dos aplastarian
    // el rango dinamico del flujo fisico de Page-Thorne)
    vec3 color = acc / total_i;
    float mapped = total_i / (1.0 + total_i);
    return clamp(color * mapped, 0.0, 1.0);
}


// =========================================================================
// Main
// =========================================================================

void main() {
    float width  = u_resolution.x;
    float height = u_resolution.y;
    // Jitter subpixel: cada frame muestrea una posicion distinta dentro del
    // pixel y la acumulacion temporal promedia (supersampling progresivo)
    float i = v_uv.x * width + u_jitter.x;
    float j = (1.0 - v_uv.y) * height + u_jitter.y;

    vec3 e_r     = cam_e_r(u_theta_cam, u_phi_cam);
    vec3 e_theta = cam_e_theta(u_theta_cam, u_phi_cam);
    vec3 e_phi   = cam_e_phi(u_theta_cam, u_phi_cam);

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
        vec3 n = vec3(0.0, 0.0, 1.0);
        if (sin_psi < 1e-12) { e2 = e_theta; }
        else {
            n = cross_rd / sin_psi;
            e2 = normalize(cross(n, e1));
        }

        // L_z/E del foton (conservada). Coincide con xi = -alpha_B*sin(theta)
        // de Bardeen en el limite a=0.
        float lam = -b * n.z;

        color = trace_schwarzschild(b, lam, e1, e2,
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
