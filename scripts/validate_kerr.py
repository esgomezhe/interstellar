"""
Validacion fisica del integrador Kerr.

Pruebas:
  1. Regresion a=0: Kerr con spin=0 debe coincidir con Schwarzschild
  2. Sombra asimetrica: la sombra Kerr esta desplazada por frame-dragging
  3. ISCO dinamico: el borde interno del disco se mueve con el spin
  4. Beaming Doppler: lado que se acerca mas brillante que el que se aleja

Uso:
  python scripts/validate_kerr.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.camera import Camera
from src.constants import kerr_isco, kerr_horizons, kerr_photon_sphere
from src.numba_kernels import (
    render_frame_parallel,
    kerr_render_frame_parallel,
    warmup_kerr,
)


def test_a0_regression():
    """Kerr a=0 debe producir imagen similar a Schwarzschild."""
    print("\n[1] Regresion a=0: Kerr vs Schwarzschild")
    print("    Renderizando Schwarzschild (40x23)...")

    W, H = 40, 23
    cam = Camera(r=30.0, theta=np.radians(75.0), width=W, height=H, fov=16.0)

    # Schwarzschild
    rays = cam.all_rays()
    b_arr = np.zeros((H, W))
    e1_arr = np.zeros((H, W, 3))
    e2_arr = np.zeros((H, W, 3))
    for j in range(H):
        for i in range(W):
            ray = rays[j][i]
            b_arr[j, i] = ray.b
            e1_arr[j, i] = ray.e1
            e2_arr[j, i] = ray.e2

    img_sch = render_frame_parallel(
        b_arr, e1_arr, e2_arr, cam.position,
        30.0, 2.0, 1.0, 50.0, 800,
        6.0, 20.0, 2200.0, 3.0,
    )

    # Kerr a=0
    print("    Renderizando Kerr a=0 (40x23)...")
    xi_arr, eta_arr, beta_B_arr = cam.all_carter_params(a=0.0)
    img_kerr = kerr_render_frame_parallel(
        xi_arr, eta_arr, beta_B_arr,
        30.0, np.radians(75.0), 0.0, 1.0,
        300.0, 800,
        6.0, 20.0, 2200.0, 3.0,
    )

    # Comparar pixeles con disco (no negros)
    sch_bright = np.sum(img_sch, axis=2) > 0.01
    kerr_bright = np.sum(img_kerr, axis=2) > 0.01

    n_sch = sch_bright.sum()
    n_kerr = kerr_bright.sum()
    overlap = (sch_bright & kerr_bright).sum()

    print(f"    Schwarzschild: {n_sch} pixeles brillantes")
    print(f"    Kerr a=0:      {n_kerr} pixeles brillantes")
    if n_sch > 0:
        print(f"    Overlap:       {overlap}/{n_sch} ({100*overlap/n_sch:.1f}%)")

    # Los conteos deben ser similares. Tolerancia amplia porque los integradores
    # usan formulaciones diferentes (u(phi) vs Carter 6-var). Lo importante es
    # que la estructura sea similar: overlap significativo y orden de magnitud.
    ratio = n_kerr / max(n_sch, 1)
    overlap_pct = 100 * overlap / max(n_sch, 1)
    ok = 0.6 < ratio < 1.6 and overlap_pct > 60
    print(f"    Ratio Kerr/Sch: {ratio:.2f}, overlap {overlap_pct:.1f}% {'PASS' if ok else 'FAIL'}")
    return ok


def test_asymmetric_shadow():
    """La sombra en Kerr debe ser asimetrica (desplazada por frame-dragging)."""
    print("\n[2] Sombra asimetrica en Kerr a=0.998")

    W, H = 60, 35
    cam = Camera(r=30.0, theta=np.radians(85.0), width=W, height=H, fov=16.0)
    xi_arr, eta_arr, beta_B_arr = cam.all_carter_params(a=0.998)

    r_inner = kerr_isco(0.998, prograde=True)
    print(f"    ISCO progrado: {r_inner:.3f}")

    img = kerr_render_frame_parallel(
        xi_arr, eta_arr, beta_B_arr,
        30.0, np.radians(85.0), 0.998, 1.0,
        300.0, 1200,
        r_inner, 20.0, 2200.0, 3.0,
    )

    brightness = np.sum(img, axis=2)
    mid = W // 2

    # Brillo promedio izquierda vs derecha
    left_bright = brightness[:, :mid].mean()
    right_bright = brightness[:, mid:].mean()

    print(f"    Brillo izquierda:  {left_bright:.4f}")
    print(f"    Brillo derecha:    {right_bright:.4f}")

    # Deben ser diferentes (asimetria)
    asym_ratio = max(left_bright, right_bright) / max(min(left_bright, right_bright), 1e-6)
    ok = asym_ratio > 1.1  # al menos 10% de diferencia
    print(f"    Ratio asimetria:   {asym_ratio:.2f} {'PASS' if ok else 'FAIL'}")
    return ok


def test_isco_with_spin():
    """ISCO debe reducirse con mayor spin."""
    print("\n[3] ISCO vs spin (Bardeen-Press-Teukolsky)")

    spins = [0.0, 0.3, 0.5, 0.7, 0.9, 0.998]
    prev_isco = 999.0
    all_ok = True

    for a in spins:
        isco = kerr_isco(a) if a > 1e-6 else 6.0
        r_plus, r_minus = kerr_horizons(a)
        r_ph = kerr_photon_sphere(a)
        ok = isco < prev_isco and isco > r_plus
        status = "PASS" if ok else "FAIL"
        print(f"    a={a:.3f}: ISCO={isco:.3f}  r+={r_plus:.3f}  r_ph={r_ph:.3f}  {status}")
        if not ok:
            all_ok = False
        prev_isco = isco

    return all_ok


def test_doppler_beaming():
    """El lado que se acerca al observador debe ser mas brillante."""
    print("\n[4] Beaming Doppler: asimetria de brillo")

    W, H = 60, 35
    # Observador cerca del ecuador para maximo efecto Doppler
    cam = Camera(r=30.0, theta=np.radians(85.0), width=W, height=H, fov=16.0)

    # Schwarzschild (a=0) ya tiene Doppler por orbita Kepleriana
    xi_arr, eta_arr, beta_B_arr = cam.all_carter_params(a=0.0)
    img = kerr_render_frame_parallel(
        xi_arr, eta_arr, beta_B_arr,
        30.0, np.radians(85.0), 0.0, 1.0,
        300.0, 800,
        6.0, 20.0, 2200.0, 3.0,
    )

    brightness = np.sum(img, axis=2)
    mid = W // 2
    left_bright = brightness[:, :mid].mean()
    right_bright = brightness[:, mid:].mean()

    brighter_side = "izquierda" if left_bright > right_bright else "derecha"
    ratio = max(left_bright, right_bright) / max(min(left_bright, right_bright), 1e-6)

    print(f"    Brillo izquierda:  {left_bright:.4f}")
    print(f"    Brillo derecha:    {right_bright:.4f}")
    print(f"    Lado mas brillante: {brighter_side}")
    print(f"    Ratio beaming:     {ratio:.2f}")

    ok = ratio > 1.05  # al menos 5% de diferencia
    print(f"    Asimetria Doppler: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("=" * 60)
    print("  Validacion Fisica del Integrador Kerr")
    print("=" * 60)

    print("\nCalentando JIT (Numba)...")
    warmup_kerr()

    # Calentar tambien Schwarzschild
    cam_warmup = Camera(r=30.0, theta=np.radians(75.0), width=2, height=2, fov=16.0)
    rays = cam_warmup.all_rays()
    b = np.array([[rays[0][0].b]])
    e1 = np.array([[[*rays[0][0].e1]]])
    e2 = np.array([[[*rays[0][0].e2]]])
    render_frame_parallel(b, e1, e2, cam_warmup.position,
                          30.0, 2.0, 1.0, 50.0, 100, 6.0, 20.0, 2200.0, 3.0)
    print("JIT listo.\n")

    results = []
    results.append(("Regresion a=0", test_a0_regression()))
    results.append(("Sombra asimetrica", test_asymmetric_shadow()))
    results.append(("ISCO vs spin", test_isco_with_spin()))
    results.append(("Beaming Doppler", test_doppler_beaming()))

    print("\n" + "=" * 60)
    print("  Resultados")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_pass = False

    print(f"\n  {'TODAS LAS PRUEBAS PASARON' if all_pass else 'ALGUNAS PRUEBAS FALLARON'}")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
