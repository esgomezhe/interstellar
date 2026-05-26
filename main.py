"""
Punto de entrada principal para la simulacion del agujero negro de Schwarzschild.

Uso:
  python main.py                   # interfaz grafica
  python main.py geodesics         # graficar geodesicas
  python main.py frame             # renderizar un frame (Numba CPU)
  python main.py animation         # animacion cinematografica 10s
  python main.py interactive       # renderer interactivo GPU (GLSL)
"""

import sys
import subprocess
import threading
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
OUTPUTS_DIR = ROOT / "outputs"

SCRIPT_MAP = {
    "geodesics":   "plot_geodesics",
    "frame":       "render_frame",
    "animation":   "render_animation",
    "interactive": "interactive",
}

OUTPUT_FILES = {
    "geodesics": "outputs/geodesic_plots/phase1_geodesics.png",
    "frame":     "outputs/frames/frame_numba.png",
    "animation": "outputs/animation.gif",
}


def run_mode(mode: str) -> None:
    script_name = SCRIPT_MAP.get(mode)
    if not script_name:
        print(f"Modo desconocido: '{mode}'")
        sys.exit(1)

    sys.path.insert(0, str(SCRIPTS_DIR))
    module = importlib.import_module(script_name)
    sys.path.pop(0)

    if hasattr(module, "main"):
        module.main()
    else:
        print(f"El script '{script_name}.py' no tiene funcion main()")
        sys.exit(1)


def launch_gui() -> None:
    import tkinter as tk
    from PIL import Image, ImageTk

    # --- Colores ---
    BG = "#0d0d0d"
    BG_CARD = "#1a1a1a"
    BG_HOVER = "#252525"
    FG = "#e0e0e0"
    FG_DIM = "#808080"
    ACCENT = "#d4a017"
    RUNNING_COLOR = "#3a7d3a"

    # --- Ventana principal ---
    root = tk.Tk()
    root.title("Schwarzschild Black Hole")
    root.configure(bg=BG)
    root.resizable(False, False)

    win_w, win_h = 520, 580
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    x = (screen_w - win_w) // 2
    y = (screen_h - win_h) // 2
    root.geometry(f"{win_w}x{win_h}+{x}+{y}")

    # ------------------------------------------------------------------
    # Viewer: ventana para mostrar imagenes y animaciones
    # ------------------------------------------------------------------
    def open_viewer(file_path: str, title: str) -> None:
        path = ROOT / file_path
        if not path.exists():
            return

        img = Image.open(path)
        is_gif = hasattr(img, "n_frames") and img.n_frames > 1

        viewer = tk.Toplevel(root)
        viewer.title(title)
        viewer.configure(bg="#000000")

        # Escalar para que quepa en pantalla con margen
        max_w = int(screen_w * 0.85)
        max_h = int(screen_h * 0.85)

        if is_gif:
            # Extraer frames y duracion
            frames_pil = []
            try:
                while True:
                    frame = img.copy().convert("RGBA")
                    frames_pil.append(frame)
                    img.seek(img.tell() + 1)
            except EOFError:
                pass

            duration = img.info.get("duration", 42)  # ms por frame

            # Escalar todos los frames
            orig_w, orig_h = frames_pil[0].size
            scale = min(max_w / orig_w, max_h / orig_h, 1.0)
            disp_w = int(orig_w * scale)
            disp_h = int(orig_h * scale)

            frames_tk = []
            for f in frames_pil:
                if scale < 1.0:
                    f = f.resize((disp_w, disp_h), Image.LANCZOS)
                frames_tk.append(ImageTk.PhotoImage(f))

            viewer.geometry(f"{disp_w}x{disp_h + 36}")
            viewer.resizable(False, False)

            canvas = tk.Label(viewer, bg="#000000")
            canvas.pack(expand=True)

            # Barra de control
            ctrl = tk.Frame(viewer, bg="#111111", height=36)
            ctrl.pack(fill="x", side="bottom")
            ctrl.pack_propagate(False)

            playing = {"on": True}
            frame_idx = {"i": 0}

            info_label = tk.Label(
                ctrl,
                text=f"Frame 1/{len(frames_tk)}",
                font=("Consolas", 9), fg=FG_DIM, bg="#111111",
            )
            info_label.pack(side="left", padx=10)

            def toggle_play():
                playing["on"] = not playing["on"]
                play_btn.configure(text="||" if playing["on"] else ">>")

            play_btn = tk.Label(
                ctrl, text="||",
                font=("Consolas", 10, "bold"), fg=FG, bg=ACCENT,
                padx=10, pady=2, cursor="hand2",
            )
            play_btn.pack(side="right", padx=10, pady=4)
            play_btn.bind("<Button-1>", lambda e: toggle_play())

            def animate():
                if not viewer.winfo_exists():
                    return
                idx = frame_idx["i"]
                canvas.configure(image=frames_tk[idx])
                info_label.configure(text=f"Frame {idx + 1}/{len(frames_tk)}")
                if playing["on"]:
                    frame_idx["i"] = (idx + 1) % len(frames_tk)
                viewer.after(duration, animate)

            animate()

        else:
            # Imagen estatica
            img = img.convert("RGBA")
            orig_w, orig_h = img.size
            scale = min(max_w / orig_w, max_h / orig_h, 1.0)
            disp_w = int(orig_w * scale)
            disp_h = int(orig_h * scale)

            if scale < 1.0:
                img = img.resize((disp_w, disp_h), Image.LANCZOS)

            viewer.geometry(f"{disp_w}x{disp_h}")
            viewer.resizable(False, False)

            photo = ImageTk.PhotoImage(img)
            label = tk.Label(viewer, image=photo, bg="#000000")
            label.image = photo  # keep reference
            label.pack(expand=True)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    header = tk.Frame(root, bg=BG)
    header.pack(fill="x", padx=30, pady=(28, 0))

    tk.Label(
        header, text="SCHWARZSCHILD",
        font=("Segoe UI", 22, "bold"), fg=ACCENT, bg=BG,
        anchor="w",
    ).pack(fill="x")

    tk.Label(
        header, text="BLACK HOLE SIMULATION",
        font=("Segoe UI", 10), fg=FG_DIM, bg=BG,
        anchor="w",
    ).pack(fill="x", pady=(0, 4))

    tk.Frame(root, bg=ACCENT, height=1).pack(fill="x", padx=30, pady=(8, 20))

    # --- Estado global ---
    status_var = tk.StringVar(value="")
    running_task = {"active": False}

    # --- Status bar ---
    status_frame = tk.Frame(root, bg=BG)
    status_frame.pack(fill="x", padx=30, side="bottom", pady=(0, 18))

    status_label = tk.Label(
        status_frame, textvariable=status_var,
        font=("Segoe UI", 9), fg=FG_DIM, bg=BG, anchor="w",
    )
    status_label.pack(fill="x")

    # --- Cards container ---
    container = tk.Frame(root, bg=BG)
    container.pack(fill="both", expand=True, padx=30, pady=(0, 10))

    modes = [
        {
            "key": "geodesics",
            "title": "Geodesics",
            "desc": "Plot photon trajectories around the black hole.\n20 rays with different impact parameters.",
            "icon": "~",
            "tag": "CPU",
        },
        {
            "key": "frame",
            "title": "Single Frame",
            "desc": "Render one static frame with full relativistic\neffects using Numba parallel kernels.",
            "icon": "#",
            "tag": "CPU  NUMBA",
        },
        {
            "key": "animation",
            "title": "Cinematic Animation",
            "desc": "10-second animation (240 frames, 24 fps).\nSmooth camera orbit from pole to equator.",
            "icon": ">",
            "tag": "CPU  NUMBA",
        },
        {
            "key": "interactive",
            "title": "Interactive Viewer",
            "desc": "Real-time GPU ray tracer with GLSL shaders.\nOrbit, zoom, F12 screenshot, F9 record video.",
            "icon": "*",
            "tag": "GPU  GLSL",
        },
    ]

    cards = []

    PROTECTED_BG = {ACCENT, RUNNING_COLOR, "#2a6496"}

    def set_card_bg(card: tk.Frame, color: str) -> None:
        card.configure(bg=color)
        for child in card.winfo_children():
            if isinstance(child, tk.Label) and child.cget("bg") not in PROTECTED_BG:
                child.configure(bg=color)
            if isinstance(child, tk.Frame):
                child.configure(bg=color)
                for sub in child.winfo_children():
                    if isinstance(sub, (tk.Label, tk.Frame)):
                        if sub.cget("bg") not in PROTECTED_BG:
                            sub.configure(bg=color)
                            for s2 in sub.winfo_children():
                                if isinstance(s2, tk.Label) and s2.cget("bg") not in PROTECTED_BG:
                                    s2.configure(bg=color)

    def has_output(mode_key: str) -> bool:
        output = OUTPUT_FILES.get(mode_key)
        return output is not None and (ROOT / output).exists()

    def view_output(mode_key: str) -> None:
        output = OUTPUT_FILES.get(mode_key)
        if output and (ROOT / output).exists():
            open_viewer(output, mode_key.title())

    def update_card_buttons():
        """Actualiza los botones de cada card segun si el output existe."""
        for _card, key, btn, view_btn in cards:
            if key == "interactive":
                continue
            if has_output(key):
                btn.configure(text="NEW")
                view_btn.pack(side="right", padx=(0, 6))
            else:
                btn.configure(text="RUN")
                view_btn.pack_forget()

    def run_task(mode_key: str, card_frame: tk.Frame, btn: tk.Label) -> None:
        if running_task["active"]:
            return

        running_task["active"] = True
        btn.configure(text="RUNNING...", bg=RUNNING_COLOR)
        status_var.set(f"Running: {mode_key}...")

        for _, _, b, _ in cards:
            b.configure(fg="#555555")

        def task():
            try:
                script = SCRIPT_MAP[mode_key]
                proc = subprocess.run(
                    [sys.executable, str(SCRIPTS_DIR / f"{script}.py")],
                    cwd=str(ROOT),
                )
                if proc.returncode == 0:
                    root.after(0, lambda: status_var.set(f"Completed: {mode_key}"))
                    output = OUTPUT_FILES.get(mode_key)
                    if output:
                        root.after(100, lambda: open_viewer(output, mode_key.title()))
                else:
                    root.after(0, lambda: status_var.set(
                        f"Error running {mode_key} (code {proc.returncode})"
                    ))
            except Exception as e:
                root.after(0, lambda: status_var.set(f"Error: {e}"))
            finally:
                running_task["active"] = False
                root.after(0, update_card_buttons)
                root.after(0, lambda: [b.configure(fg=FG) for _, _, b, _ in cards])

        threading.Thread(target=task, daemon=True).start()

    VIEW_BG = "#2a6496"

    for mode in modes:
        card = tk.Frame(container, bg=BG_CARD, cursor="hand2")
        card.pack(fill="x", pady=5)
        card.pack_propagate(False)
        card.configure(height=90)

        icon_label = tk.Label(
            card, text=mode["icon"],
            font=("Consolas", 20, "bold"), fg=ACCENT, bg=BG_CARD,
            width=3,
        )
        icon_label.pack(side="left", padx=(12, 4))

        btn = tk.Label(
            card, text="RUN",
            font=("Segoe UI", 9, "bold"), fg=FG, bg=ACCENT,
            padx=14, pady=4, cursor="hand2",
        )
        btn.pack(side="right", padx=14)

        # Boton VIEW (solo para modos con output)
        view_btn = tk.Label(
            card, text="VIEW",
            font=("Segoe UI", 9, "bold"), fg=FG, bg=VIEW_BG,
            padx=14, pady=4, cursor="hand2",
        )

        text_frame = tk.Frame(card, bg=BG_CARD)
        text_frame.pack(side="left", fill="both", expand=True, pady=10)

        title_row = tk.Frame(text_frame, bg=BG_CARD)
        title_row.pack(fill="x", anchor="w")

        tk.Label(
            title_row, text=mode["title"],
            font=("Segoe UI", 12, "bold"), fg=FG, bg=BG_CARD,
            anchor="w",
        ).pack(side="left")

        tk.Label(
            title_row, text=mode["tag"],
            font=("Consolas", 8), fg=FG_DIM, bg=BG_CARD,
            anchor="w", padx=8,
        ).pack(side="left")

        tk.Label(
            text_frame, text=mode["desc"],
            font=("Segoe UI", 9), fg=FG_DIM, bg=BG_CARD,
            anchor="w", justify="left",
        ).pack(fill="x", anchor="w")

        card.bind("<Enter>", lambda e, c=card: set_card_bg(c, BG_HOVER))
        card.bind("<Leave>", lambda e, c=card: set_card_bg(c, BG_CARD))

        key = mode["key"]
        btn.bind("<Button-1>", lambda e, k=key, c=card, b=btn: run_task(k, c, b))
        view_btn.bind("<Button-1>", lambda e, k=key: view_output(k))

        # Click en la card: si hay output, view; si no, run
        card.bind("<Button-1>", lambda e, k=key, c=card, b=btn: (
            view_output(k) if has_output(k) else run_task(k, c, b)
        ))

        cards.append((card, key, btn, view_btn))

    # Estado inicial de los botones
    update_card_buttons()

    root.mainloop()


def main() -> None:
    if len(sys.argv) > 1:
        run_mode(sys.argv[1].lower())
        return
    launch_gui()


if __name__ == "__main__":
    main()
