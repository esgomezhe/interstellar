#version 330 core

/*
 * Bloom post-processing: extrae pixeles brillantes y aplica blur gaussiano.
 *
 * Modo 0: extract (umbral de brillo)
 * Modo 1: blur horizontal
 * Modo 2: blur vertical
 * Modo 3: composicion final (original + bloom)
 */

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_texture;
uniform sampler2D u_bloom_texture;
uniform int u_mode;
uniform float u_threshold;     // umbral de brillo para extraccion
uniform float u_intensity;     // intensidad del bloom
uniform vec2 u_texel_size;     // 1.0 / resolution

// Pesos gaussianos (kernel 9-tap)
const float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main() {
    if (u_mode == 0) {
        // Extract: pixeles por encima del umbral
        vec3 color = texture(u_texture, v_uv).rgb;
        float brightness = dot(color, vec3(0.2126, 0.7152, 0.0722));
        if (brightness > u_threshold) {
            frag_color = vec4(color * (brightness - u_threshold), 1.0);
        } else {
            frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        }
    }
    else if (u_mode == 1) {
        // Blur horizontal
        vec3 result = texture(u_texture, v_uv).rgb * weights[0];
        for (int i = 1; i < 5; i++) {
            float offset = float(i) * u_texel_size.x * 2.0;
            result += texture(u_texture, v_uv + vec2(offset, 0.0)).rgb * weights[i];
            result += texture(u_texture, v_uv - vec2(offset, 0.0)).rgb * weights[i];
        }
        frag_color = vec4(result, 1.0);
    }
    else if (u_mode == 2) {
        // Blur vertical
        vec3 result = texture(u_texture, v_uv).rgb * weights[0];
        for (int i = 1; i < 5; i++) {
            float offset = float(i) * u_texel_size.y * 2.0;
            result += texture(u_texture, v_uv + vec2(0.0, offset)).rgb * weights[i];
            result += texture(u_texture, v_uv - vec2(0.0, offset)).rgb * weights[i];
        }
        frag_color = vec4(result, 1.0);
    }
    else {
        // Compose: original + bloom
        vec3 original = texture(u_texture, v_uv).rgb;
        vec3 bloom = texture(u_bloom_texture, v_uv).rgb;
        frag_color = vec4(original + bloom * u_intensity, 1.0);
    }
}
