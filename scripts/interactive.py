"""
Renderer interactivo en tiempo real del agujero negro de Schwarzschild.

Usa ModernGL + GLFW para renderizar el ray tracer como fragment shader GLSL.
Cada pixel ejecuta la integracion RK4 de geodesicas en la GPU.

Controles:
  - Arrastrar raton: orbitar camara (theta, phi)
  - Scroll: zoom (acercar/alejar en r)
  - WASD: orbitar camara (alternativa al raton)
  - R: reset camara a posicion inicial
  - +/-: ajustar pasos RK4 (calidad vs rendimiento)
  - F12: capturar screenshot (PNG)
  - F9: iniciar/detener grabacion de video (MP4 + GIF)
  - ESC: salir
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import glfw
import moderngl
import numpy as np
from PIL import Image

# Directorio raiz del proyecto
ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))


def load_shader(name: str) -> str:
    path = ROOT / "shaders" / name
    return path.read_text(encoding="utf-8")


SCREENSHOTS_DIR = ROOT / "outputs" / "screenshots"
RECORDINGS_DIR = ROOT / "outputs" / "recordings"


def halton(index: int, base: int) -> float:
    """Secuencia de Halton: muestreo subpixel bien distribuido para TAA."""
    result = 0.0
    f = 1.0
    i = index
    while i > 0:
        f /= base
        result += f * (i % base)
        i //= base
    return result


def read_framebuffer(ctx, fb_w: int, fb_h: int) -> Image.Image:
    """Lee los pixeles del framebuffer actual y devuelve una imagen PIL."""
    raw = ctx.screen.read(viewport=(0, 0, fb_w, fb_h), components=3)
    img = Image.frombytes("RGB", (fb_w, fb_h), raw)
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def save_screenshot(ctx, fb_w: int, fb_h: int) -> Path:
    """Captura el frame actual y lo guarda como PNG."""
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SCREENSHOTS_DIR / f"screenshot_{timestamp}.png"
    img = read_framebuffer(ctx, fb_w, fb_h)
    img.save(path, "PNG")
    return path


def save_recording(frames: list, fps: int = 30) -> tuple[Path, Path]:
    """Guarda una lista de frames PIL como MP4 y GIF."""
    import imageio

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mp4_path = RECORDINGS_DIR / f"recording_{timestamp}.mp4"
    gif_path = RECORDINGS_DIR / f"recording_{timestamp}.gif"

    # MP4
    writer = imageio.get_writer(str(mp4_path), fps=fps, codec="libx264",
                                quality=8)
    for frame in frames:
        writer.append_data(np.array(frame))
    writer.close()

    # GIF (reducir fps para tamaño razonable)
    gif_fps = min(fps, 15)
    step = max(1, fps // gif_fps)
    gif_frames = frames[::step]
    imageio.mimsave(str(gif_path), [np.array(f) for f in gif_frames],
                    duration=1000 // gif_fps, loop=0)

    return mp4_path, gif_path


def run_interactive(
    metric: str = "schwarzschild",
    initial_spin: float = 0.0,
    allow_spin_change: bool = False,
) -> None:
    """
    Loop principal del renderer interactivo.

    Args:
        metric: "schwarzschild" o "kerr"
        initial_spin: spin inicial (0.0 para Schwarzschild, 0.998 para Kerr)
        allow_spin_change: si True, teclas 0-9 cambian el spin
    """
    from src.constants import kerr_isco, page_thorne_norm_inv

    # --- Parametros iniciales de la camara ---
    cam_r = 30.0
    cam_theta = np.radians(75.0)
    cam_phi = 0.0
    fov = 30.0

    # --- Parametros fisicos ---
    rs = 2.0
    m = 1.0
    spin = initial_spin
    r_outer = 20.0
    # Temperatura en el radio de referencia del perfil Page-Thorne;
    # T(r) = base_temp * (F/F_ref)^(1/4) — el nucleo supera base_temp (HDR)
    base_temp = 4000.0
    beaming_power = 3.0

    # ISCO segun spin
    if spin < 1e-6:
        r_inner = 6.0
    else:
        r_inner = kerr_isco(spin, m, prograde=True)

    # Normalizacion del perfil Page-Thorne (depende del spin via ISCO)
    pt_fmax_inv = page_thorne_norm_inv(r_inner, r_outer, spin, m)

    # Tabla de spin: teclas 0-9
    SPIN_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.998]

    # --- Parametros de render ---
    # Pasos RK4: con el paso adaptativo (refinado en anillo de fotones y
    # polos), 2000 da calidad indistinguible de 3000 con +42% de FPS.
    # Las teclas +/- permiten subir hasta 3000 si se desea.
    n_steps = 2000
    phi_max = 50.0
    lam_max = 500.0     # parametro afin maximo para Kerr (adaptive stepping budget)
    gamma = 0.45
    time_scale = 6.0    # acelera la rotacion del disco para hacerla visible
    target_fps = 120
    frame_time_min = 1.0 / target_fps

    # --- Limites de la camara ---
    r_min = 4.0         # no entrar al horizonte
    r_max = 200.0
    theta_min = np.radians(5.0)
    theta_max = np.radians(175.0)

    # --- Estado del raton ---
    mouse_pressed = False
    last_mouse_x = 0.0
    last_mouse_y = 0.0

    # --- Estado de grabacion ---
    recording = {"active": False, "frames": [], "fps": 30}

    # --- Inicializar GLFW ---
    if not glfw.init():
        print("Error: no se pudo inicializar GLFW")
        sys.exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    glfw.window_hint(glfw.RESIZABLE, True)

    win_w, win_h = 960, 540
    title_metric = "Schwarzschild" if metric == "schwarzschild" else "Kerr"
    window = glfw.create_window(
        win_w, win_h, f"Agujero Negro — {title_metric}", None, None,
    )
    if not window:
        glfw.terminate()
        print("Error: no se pudo crear la ventana GLFW")
        sys.exit(1)

    glfw.make_context_current(window)
    glfw.swap_interval(0)  # vsync off para medir FPS real

    # --- Contexto ModernGL ---
    ctx = moderngl.create_context()

    # --- Cargar shaders ---
    vert_src = load_shader("blackhole.vert")
    frag_src = load_shader("blackhole.frag")
    bloom_frag_src = load_shader("bloom.frag")
    prog = ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
    bloom_prog = ctx.program(vertex_shader=vert_src, fragment_shader=bloom_frag_src)

    # --- Fullscreen quad ---
    vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    ], dtype="f4")
    vbo = ctx.buffer(vertices)
    vao = ctx.simple_vertex_array(prog, vbo, "in_position")
    bloom_vao = ctx.simple_vertex_array(bloom_prog, vbo, "in_position")

    # --- FBOs para bloom ---
    bloom_threshold = 0.4
    bloom_intensity = 0.6
    fbo_cache = {"w": 0, "h": 0}

    def create_bloom_fbos(w, h):
        """Crea FBOs para el pipeline de bloom y acumulacion TAA."""
        fbo_cache["scene_tex"] = ctx.texture((w, h), 4, dtype="f2")
        fbo_cache["scene_fbo"] = ctx.framebuffer(
            color_attachments=[fbo_cache["scene_tex"]]
        )
        fbo_cache["bright_tex"] = ctx.texture((w, h), 4, dtype="f2")
        fbo_cache["bright_fbo"] = ctx.framebuffer(
            color_attachments=[fbo_cache["bright_tex"]]
        )
        fbo_cache["ping_tex"] = ctx.texture((w, h), 4, dtype="f2")
        fbo_cache["ping_fbo"] = ctx.framebuffer(
            color_attachments=[fbo_cache["ping_tex"]]
        )
        # Ping-pong de acumulacion temporal (TAA)
        for name in ("accum_a", "accum_b"):
            fbo_cache[f"{name}_tex"] = ctx.texture((w, h), 4, dtype="f2")
            fbo_cache[f"{name}_fbo"] = ctx.framebuffer(
                color_attachments=[fbo_cache[f"{name}_tex"]]
            )
        fbo_cache["w"] = w
        fbo_cache["h"] = h

    # --- Callbacks ---
    def on_scroll(_win, _xoff, yoff):
        nonlocal cam_r
        factor = 0.9 if yoff > 0 else 1.1
        cam_r = np.clip(cam_r * factor, r_min, r_max)

    def on_mouse_button(_win, button, action, _mods):
        nonlocal mouse_pressed, last_mouse_x, last_mouse_y
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                mouse_pressed = True
                last_mouse_x, last_mouse_y = glfw.get_cursor_pos(_win)
            else:
                mouse_pressed = False

    def on_cursor_pos(_win, xpos, ypos):
        nonlocal cam_theta, cam_phi, last_mouse_x, last_mouse_y
        if not mouse_pressed:
            return
        dx = xpos - last_mouse_x
        dy = ypos - last_mouse_y
        last_mouse_x = xpos
        last_mouse_y = ypos

        sensitivity = 0.005
        cam_phi -= dx * sensitivity
        cam_theta = np.clip(cam_theta + dy * sensitivity, theta_min, theta_max)

    # Acciones pendientes desde callbacks (se procesan en el loop principal)
    pending_actions = []

    def update_isco():
        """Recalcula ISCO y la normalizacion Page-Thorne segun el spin."""
        nonlocal r_inner, pt_fmax_inv
        if spin < 1e-6:
            r_inner = 6.0  # Schwarzschild ISCO
        else:
            r_inner = kerr_isco(spin, m, prograde=True)
        pt_fmax_inv = page_thorne_norm_inv(r_inner, r_outer, spin, m)

    def on_key(_win, key, _scancode, action, _mods):
        nonlocal cam_theta, cam_phi, cam_r, n_steps, spin
        if action == glfw.RELEASE:
            return
        step_angle = np.radians(2.0)
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(_win, True)
        elif key == glfw.KEY_W:
            cam_theta = np.clip(cam_theta - step_angle, theta_min, theta_max)
        elif key == glfw.KEY_S:
            cam_theta = np.clip(cam_theta + step_angle, theta_min, theta_max)
        elif key == glfw.KEY_A:
            cam_phi += step_angle
        elif key == glfw.KEY_D:
            cam_phi -= step_angle
        elif key == glfw.KEY_R:
            cam_r = 30.0
            cam_theta = np.radians(75.0)
            cam_phi = 0.0
            if allow_spin_change:
                spin = initial_spin
                update_isco()
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            n_steps = min(n_steps + 100, 3000)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            n_steps = max(n_steps - 100, 200)
        elif key == glfw.KEY_F12:
            pending_actions.append("screenshot")
        elif key == glfw.KEY_F9:
            pending_actions.append("toggle_recording")
        # Teclas 0-9: ajustar spin (solo si se permite)
        elif allow_spin_change and glfw.KEY_0 <= key <= glfw.KEY_9:
            idx = key - glfw.KEY_0
            spin = SPIN_VALUES[idx]
            update_isco()
            print(f"  Spin = {spin:.3f} | ISCO = {r_inner:.3f}")

    glfw.set_scroll_callback(window, on_scroll)
    glfw.set_mouse_button_callback(window, on_mouse_button)
    glfw.set_cursor_pos_callback(window, on_cursor_pos)
    glfw.set_key_callback(window, on_key)

    # --- Loop principal ---
    frame_count = 0
    fps_timer = time.perf_counter()
    t_total_start = time.perf_counter()
    fps_display = 0.0

    # Estado de acumulacion temporal (TAA)
    accum_frames = 0
    frame_index = 0
    last_params = None

    spin_controls = ""
    if allow_spin_change:
        spin_controls = "\n  0-9: spin (0=Schwarzschild, 9=Gargantua a=0.998)"

    print("=" * 60)
    print(f"  Agujero Negro — Renderer Interactivo ({title_metric})")
    print("=" * 60)
    print("Controles:")
    print("  Raton: arrastrar para orbitar, scroll para zoom")
    print("  WASD: orbitar camara")
    print("  R: reset camara")
    print("  +/-: ajustar calidad (pasos RK4)", end="")
    print(spin_controls)
    print("  F12: capturar screenshot (PNG)")
    print("  F9:  iniciar/detener grabacion de video")
    print("  ESC: salir")
    print("=" * 60)

    while not glfw.window_should_close(window):
        frame_start = time.perf_counter()
        glfw.poll_events()

        # Obtener tamano actual de la ventana
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        if fb_w == 0 or fb_h == 0:
            continue

        # Recrear FBOs si cambio el tamano
        if fb_w != fbo_cache["w"] or fb_h != fbo_cache["h"]:
            create_bloom_fbos(fb_w, fb_h)
            accum_frames = 0

        # Reset de acumulacion si la camara o la fisica cambiaron
        params_now = (
            round(cam_r, 6), round(cam_theta, 6), round(cam_phi, 6),
            spin, n_steps, fov,
        )
        if params_now != last_params:
            accum_frames = 0
            last_params = params_now
        accum_frames += 1
        frame_index += 1

        # Jitter subpixel (Halton 2,3) — sin jitter en el primer frame tras
        # mover la camara para evitar shimmer durante la interaccion
        if accum_frames == 1:
            jitter = (0.0, 0.0)
        else:
            h_idx = frame_index % 64 + 1
            jitter = (halton(h_idx, 2) - 0.5, halton(h_idx, 3) - 0.5)
        # Peso del frame nuevo: 1/N decreciente con piso (el disco rota)
        taa_blend = max(1.0 / accum_frames, 0.06)

        # --- Pass 1: Render escena a FBO ---
        fbo_cache["scene_fbo"].use()
        ctx.viewport = (0, 0, fb_w, fb_h)

        prog["u_r_cam"].value = float(cam_r)
        prog["u_theta_cam"].value = float(cam_theta)
        prog["u_phi_cam"].value = float(cam_phi)
        prog["u_fov"].value = float(fov)
        prog["u_resolution"].value = (float(fb_w), float(fb_h))

        prog["u_rs"].value = float(rs)
        prog["u_m"].value = float(m)
        prog["u_spin"].value = float(spin)
        prog["u_r_inner"].value = float(r_inner)
        prog["u_r_outer"].value = float(r_outer)
        prog["u_base_temp"].value = float(base_temp)
        prog["u_beaming_power"].value = float(beaming_power)
        prog["u_pt_fmax_inv"].value = float(pt_fmax_inv)

        prog["u_n_steps"].value = int(n_steps)
        prog["u_phi_max"].value = float(phi_max)
        # lam_max: budget for adaptive stepping
        kerr_lam = float(cam_r + lam_max)
        max_dlam = 0.25
        kerr_lam = min(kerr_lam, max_dlam * n_steps)
        prog["u_lam_max"].value = kerr_lam
        prog["u_gamma"].value = float(gamma)
        prog["u_time"].value = float((time.perf_counter() - t_total_start) * time_scale)
        prog["u_jitter"].value = jitter

        ctx.clear(0.0, 0.0, 0.0)
        vao.render(moderngl.TRIANGLE_STRIP)

        texel_size = (1.0 / fb_w, 1.0 / fb_h)

        # --- Pass 2: Acumulacion temporal TAA (scene + accum_a → accum_b) ---
        # Promedia frames jittereados: supersampling progresivo que elimina
        # el speckle del anillo de fotones sin perder nitidez
        fbo_cache["accum_b_fbo"].use()
        fbo_cache["scene_tex"].use(location=0)
        fbo_cache["accum_a_tex"].use(location=1)
        bloom_prog["u_texture"].value = 0
        bloom_prog["u_bloom_texture"].value = 1
        bloom_prog["u_mode"].value = 5
        bloom_prog["u_blend"].value = float(taa_blend)
        bloom_prog["u_texel_size"].value = texel_size
        ctx.clear(0.0, 0.0, 0.0)
        bloom_vao.render(moderngl.TRIANGLE_STRIP)
        # accum_b_tex = imagen acumulada (limpia)

        # --- Pass 3: Extraer pixeles brillantes (desde acumulado) ---
        fbo_cache["bright_fbo"].use()
        fbo_cache["accum_b_tex"].use(location=0)
        bloom_prog["u_texture"].value = 0
        bloom_prog["u_mode"].value = 0
        bloom_prog["u_threshold"].value = float(bloom_threshold)
        ctx.clear(0.0, 0.0, 0.0)
        bloom_vao.render(moderngl.TRIANGLE_STRIP)

        # --- Pass 4: Blur horizontal ---
        fbo_cache["ping_fbo"].use()
        fbo_cache["bright_tex"].use(location=0)
        bloom_prog["u_texture"].value = 0
        bloom_prog["u_mode"].value = 1
        ctx.clear(0.0, 0.0, 0.0)
        bloom_vao.render(moderngl.TRIANGLE_STRIP)

        # --- Pass 5: Blur vertical ---
        fbo_cache["bright_fbo"].use()
        fbo_cache["ping_tex"].use(location=0)
        bloom_prog["u_texture"].value = 0
        bloom_prog["u_mode"].value = 2
        ctx.clear(0.0, 0.0, 0.0)
        bloom_vao.render(moderngl.TRIANGLE_STRIP)

        # --- Pass 6: Composicion final a pantalla ---
        # accum_b_tex = escena acumulada, bright_tex = bloom difuminado
        ctx.screen.use()
        ctx.viewport = (0, 0, fb_w, fb_h)
        fbo_cache["accum_b_tex"].use(location=0)
        fbo_cache["bright_tex"].use(location=1)
        bloom_prog["u_texture"].value = 0
        bloom_prog["u_bloom_texture"].value = 1
        bloom_prog["u_mode"].value = 3
        bloom_prog["u_intensity"].value = float(bloom_intensity)
        ctx.clear(0.0, 0.0, 0.0)
        bloom_vao.render(moderngl.TRIANGLE_STRIP)

        glfw.swap_buffers(window)

        # Swap ping-pong de acumulacion para el proximo frame
        fbo_cache["accum_a_tex"], fbo_cache["accum_b_tex"] = (
            fbo_cache["accum_b_tex"], fbo_cache["accum_a_tex"],
        )
        fbo_cache["accum_a_fbo"], fbo_cache["accum_b_fbo"] = (
            fbo_cache["accum_b_fbo"], fbo_cache["accum_a_fbo"],
        )

        # --- Procesar acciones de captura ---
        for act in pending_actions:
            if act == "screenshot":
                path = save_screenshot(ctx, fb_w, fb_h)
                print(f"  [F12] Screenshot guardado: {path}")
            elif act == "toggle_recording":
                if not recording["active"]:
                    recording["active"] = True
                    recording["frames"] = []
                    print("  [F9] Grabacion iniciada...")
                else:
                    recording["active"] = False
                    n_frames = len(recording["frames"])
                    print(f"  [F9] Grabacion detenida ({n_frames} frames). Codificando...")
                    if n_frames > 0:
                        mp4, gif = save_recording(recording["frames"],
                                                  recording["fps"])
                        print(f"        MP4: {mp4}")
                        print(f"        GIF: {gif}")
                    recording["frames"] = []
        pending_actions.clear()

        # Capturar frame si estamos grabando
        if recording["active"]:
            recording["frames"].append(read_framebuffer(ctx, fb_w, fb_h))

        # Frame limiter
        elapsed = time.perf_counter() - frame_start
        sleep_time = frame_time_min - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

        # FPS
        frame_count += 1
        now = time.perf_counter()
        elapsed = now - fps_timer
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            fps_timer = now
            theta_deg = np.degrees(cam_theta)
            phi_deg = np.degrees(cam_phi) % 360
            rec_tag = " | [REC]" if recording["active"] else ""
            if metric == "kerr":
                spin_tag = f" a={spin:.3f}"
            else:
                spin_tag = ""
            glfw.set_window_title(
                window,
                f"{title_metric}{spin_tag} — {fps_display:.1f} FPS | "
                f"r={cam_r:.1f} θ={theta_deg:.1f}° φ={phi_deg:.1f}° | "
                f"ISCO={r_inner:.2f} steps={n_steps}{rec_tag}"
            )

    # Guardar grabacion pendiente si se cerro la ventana mientras grababa
    if recording["active"] and len(recording["frames"]) > 0:
        print(f"\n  Grabacion activa al cerrar ({len(recording['frames'])} frames). Codificando...")
        mp4, gif = save_recording(recording["frames"], recording["fps"])
        print(f"  MP4: {mp4}")
        print(f"  GIF: {gif}")

    glfw.terminate()
    print("\nSesion finalizada.")


def main() -> None:
    run_interactive(metric="schwarzschild", initial_spin=0.0, allow_spin_change=False)


if __name__ == "__main__":
    main()
