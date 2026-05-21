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
    from tkinter import ttk

    # --- Colores ---
    BG = "#0d0d0d"
    BG_CARD = "#1a1a1a"
    BG_HOVER = "#252525"
    FG = "#e0e0e0"
    FG_DIM = "#808080"
    ACCENT = "#d4a017"
    ACCENT_HOVER = "#e8b830"
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

    # --- Header ---
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
            "desc": "Real-time GPU ray tracer with GLSL shaders.\nOrbit, zoom, explore the black hole freely.",
            "icon": "*",
            "tag": "GPU  GLSL",
        },
    ]

    cards = []

    def run_task(mode_key: str, card_frame: tk.Frame, btn: tk.Label) -> None:
        if running_task["active"]:
            return

        running_task["active"] = True
        btn.configure(text="RUNNING...", bg=RUNNING_COLOR)
        status_var.set(f"Running: {mode_key}...")

        # Disable all buttons visually
        for _, _, b in cards:
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
                else:
                    root.after(0, lambda: status_var.set(f"Error running {mode_key} (code {proc.returncode})"))
            except Exception as e:
                root.after(0, lambda: status_var.set(f"Error: {e}"))
            finally:
                running_task["active"] = False
                root.after(0, lambda: btn.configure(text="RUN", bg=ACCENT))
                root.after(0, lambda: [b.configure(fg=FG) for _, _, b in cards])

        threading.Thread(target=task, daemon=True).start()

    for mode in modes:
        card = tk.Frame(container, bg=BG_CARD, cursor="hand2")
        card.pack(fill="x", pady=5)
        card.pack_propagate(False)
        card.configure(height=90)

        # Left: icon
        icon_label = tk.Label(
            card, text=mode["icon"],
            font=("Consolas", 20, "bold"), fg=ACCENT, bg=BG_CARD,
            width=3,
        )
        icon_label.pack(side="left", padx=(12, 4))

        # Right: run button
        btn = tk.Label(
            card, text="RUN",
            font=("Segoe UI", 9, "bold"), fg=FG, bg=ACCENT,
            padx=14, pady=4, cursor="hand2",
        )
        btn.pack(side="right", padx=14)

        # Center: text
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

        # Hover effects
        def on_enter(e, c=card):
            c.configure(bg=BG_HOVER)
            for child in c.winfo_children():
                if isinstance(child, tk.Label) and child.cget("bg") not in (ACCENT, RUNNING_COLOR):
                    child.configure(bg=BG_HOVER)
                if isinstance(child, tk.Frame):
                    child.configure(bg=BG_HOVER)
                    for sub in child.winfo_children():
                        if isinstance(sub, (tk.Label, tk.Frame)):
                            if sub.cget("bg") not in (ACCENT, RUNNING_COLOR):
                                sub.configure(bg=BG_HOVER)
                                for s2 in sub.winfo_children():
                                    if isinstance(s2, tk.Label) and s2.cget("bg") not in (ACCENT, RUNNING_COLOR):
                                        s2.configure(bg=BG_HOVER)

        def on_leave(e, c=card):
            c.configure(bg=BG_CARD)
            for child in c.winfo_children():
                if isinstance(child, tk.Label) and child.cget("bg") not in (ACCENT, RUNNING_COLOR):
                    child.configure(bg=BG_CARD)
                if isinstance(child, tk.Frame):
                    child.configure(bg=BG_CARD)
                    for sub in child.winfo_children():
                        if isinstance(sub, (tk.Label, tk.Frame)):
                            if sub.cget("bg") not in (ACCENT, RUNNING_COLOR):
                                sub.configure(bg=BG_CARD)
                                for s2 in sub.winfo_children():
                                    if isinstance(s2, tk.Label) and s2.cget("bg") not in (ACCENT, RUNNING_COLOR):
                                        s2.configure(bg=BG_CARD)

        card.bind("<Enter>", on_enter)
        card.bind("<Leave>", on_leave)

        # Click handlers
        key = mode["key"]
        btn.bind("<Button-1>", lambda e, k=key, c=card, b=btn: run_task(k, c, b))
        card.bind("<Button-1>", lambda e, k=key, c=card, b=btn: run_task(k, c, b))

        cards.append((card, key, btn))

    root.mainloop()


def main() -> None:
    if len(sys.argv) > 1:
        run_mode(sys.argv[1].lower())
        return
    launch_gui()


if __name__ == "__main__":
    main()
