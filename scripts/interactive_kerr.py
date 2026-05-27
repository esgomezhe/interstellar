"""
Renderer interactivo en tiempo real del agujero negro de Kerr.

Usa ModernGL + GLFW para renderizar el ray tracer como fragment shader GLSL.
Cada pixel ejecuta la integracion RK4 de geodesicas en la GPU.

Controles:
  - Arrastrar raton: orbitar camara (theta, phi)
  - Scroll: zoom (acercar/alejar en r)
  - WASD: orbitar camara (alternativa al raton)
  - R: reset camara y spin
  - +/-: ajustar pasos RK4 (calidad vs rendimiento)
  - 0-9: ajustar spin del agujero negro (0=sin spin, 9=Gargantua a=0.998)
  - F12: capturar screenshot (PNG)
  - F9: iniciar/detener grabacion de video (MP4 + GIF)
  - ESC: salir
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from interactive import run_interactive


def main() -> None:
    run_interactive(metric="kerr", initial_spin=0.998, allow_spin_change=True)


if __name__ == "__main__":
    main()
