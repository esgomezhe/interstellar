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
  - ESC: salir
"""

import sys
import time
from pathlib import Path

import glfw
import moderngl
import numpy as np

# Directorio raiz del proyecto
ROOT = Path(__file__).resolve().parent.parent


def load_shader(name: str) -> str:
    path = ROOT / "shaders" / name
    return path.read_text(encoding="utf-8")


def main() -> None:
    # --- Parametros iniciales de la camara ---
    cam_r = 30.0
    cam_theta = np.radians(75.0)
    cam_phi = 0.0
    fov = 30.0

    # --- Parametros fisicos ---
    rs = 2.0
    m = 1.0
    r_inner = 6.0       # ISCO
    r_outer = 20.0
    base_temp = 2200.0
    beaming_power = 3.0

    # --- Parametros de render ---
    n_steps = 3000      # pasos RK4 (GPU permite alta calidad)
    phi_max = 50.0
    gamma = 0.45
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
    window = glfw.create_window(win_w, win_h, "Agujero Negro de Schwarzschild — Interactivo", None, None)
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
    prog = ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)

    # --- Fullscreen quad ---
    vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    ], dtype="f4")
    vbo = ctx.buffer(vertices)
    vao = ctx.simple_vertex_array(prog, vbo, "in_position")

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

    def on_key(_win, key, _scancode, action, _mods):
        nonlocal cam_theta, cam_phi, cam_r, n_steps
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
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            n_steps = min(n_steps + 100, 3000)
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            n_steps = max(n_steps - 100, 200)

    glfw.set_scroll_callback(window, on_scroll)
    glfw.set_mouse_button_callback(window, on_mouse_button)
    glfw.set_cursor_pos_callback(window, on_cursor_pos)
    glfw.set_key_callback(window, on_key)

    # --- Loop principal ---
    frame_count = 0
    fps_timer = time.perf_counter()
    fps_display = 0.0

    print("=" * 60)
    print("  Agujero Negro de Schwarzschild — Renderer Interactivo")
    print("=" * 60)
    print("Controles:")
    print("  Raton: arrastrar para orbitar, scroll para zoom")
    print("  WASD: orbitar camara")
    print("  R: reset camara")
    print("  +/-: ajustar calidad (pasos RK4)")
    print("  ESC: salir")
    print("=" * 60)

    while not glfw.window_should_close(window):
        frame_start = time.perf_counter()
        glfw.poll_events()

        # Obtener tamano actual de la ventana
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        if fb_w == 0 or fb_h == 0:
            continue

        ctx.viewport = (0, 0, fb_w, fb_h)

        # Pasar uniforms al shader
        prog["u_r_cam"].value = float(cam_r)
        prog["u_theta_cam"].value = float(cam_theta)
        prog["u_phi_cam"].value = float(cam_phi)
        prog["u_fov"].value = float(fov)
        prog["u_resolution"].value = (float(fb_w), float(fb_h))

        prog["u_rs"].value = float(rs)
        prog["u_m"].value = float(m)
        prog["u_r_inner"].value = float(r_inner)
        prog["u_r_outer"].value = float(r_outer)
        prog["u_base_temp"].value = float(base_temp)
        prog["u_beaming_power"].value = float(beaming_power)

        prog["u_n_steps"].value = int(n_steps)
        prog["u_phi_max"].value = float(phi_max)
        prog["u_gamma"].value = float(gamma)

        # Renderizar
        ctx.clear(0.0, 0.0, 0.0)
        vao.render(moderngl.TRIANGLE_STRIP)

        glfw.swap_buffers(window)

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
            glfw.set_window_title(
                window,
                f"Agujero Negro — {fps_display:.1f} FPS | "
                f"r={cam_r:.1f} theta={theta_deg:.1f} phi={phi_deg:.1f} | "
                f"RK4 steps={n_steps}"
            )

    glfw.terminate()
    print("\nSesion finalizada.")


if __name__ == "__main__":
    main()
