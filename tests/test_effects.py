"""Tests de validacion fisica para el g-factor relativista del disco."""

import numpy as np
import pytest

from src.effects import cunningham_g_factor, photon_angular_momentum
from src.constants import kerr_isco


class TestGFactorSchwarzschild:
    """g = sqrt(1 - 3M/r) / (1 - Omega*lam) con a=0."""

    def test_isco_cara_a_cara(self) -> None:
        # Foton sin momento angular desde el ISCO: g = sqrt(1 - 3/6) = sqrt(1/2)
        g = cunningham_g_factor(6.0, 0.0)
        assert abs(g - np.sqrt(0.5)) < 1e-12

    def test_radio_grande_tiende_a_uno(self) -> None:
        # Lejos del agujero negro el corrimiento desaparece
        g = cunningham_g_factor(1e6, 0.0)
        assert abs(g - 1.0) < 1e-5

    def test_asimetria_doppler(self) -> None:
        # lam > 0 (lado que se acerca): blueshift relativo
        # lam < 0 (lado que se aleja): redshift relativo
        g_acerca = cunningham_g_factor(8.0, 3.0)
        g_aleja = cunningham_g_factor(8.0, -3.0)
        assert g_acerca > g_aleja
        # El producto refleja la simetria: ambos comparten sqrt(1-3M/r)
        omega = np.sqrt(1.0) / 8.0 ** 1.5
        esperado = (1.0 - 3.0 / 8.0) / (1.0 - (omega * 3.0) ** 2)
        assert abs(g_acerca * g_aleja - esperado) < 1e-10

    def test_formula_explicita(self) -> None:
        r, lam = 10.0, 2.0
        omega = 1.0 / 10.0 ** 1.5
        esperado = np.sqrt(1.0 - 0.3) / (1.0 - omega * lam)
        assert abs(cunningham_g_factor(r, lam) - esperado) < 1e-12


class TestGFactorKerr:
    """Forma exacta de Bardeen-Press-Teukolsky con spin."""

    def test_reduccion_a_schwarzschild(self) -> None:
        # a=0 debe coincidir exactamente con la formula de Schwarzschild
        for r in [6.0, 10.0, 20.0]:
            for lam in [-3.0, 0.0, 3.0]:
                assert cunningham_g_factor(r, lam, a=0.0) == pytest.approx(
                    cunningham_g_factor(r, lam), abs=1e-14,
                )

    def test_normalizacion_cuatro_velocidad(self) -> None:
        # 1/u^t segun la metrica de Kerr (ecuatorial, Boyer-Lindquist):
        # 1/(u^t)^2 = 1 - 2M/r + 4MaOmega/r - (r^2 + a^2 + 2Ma^2/r)*Omega^2
        a, m, r = 0.9, 1.0, 4.0
        omega = np.sqrt(m) / (r ** 1.5 + a * np.sqrt(m))
        inv_ut2 = (
            1.0 - 2.0 * m / r + 4.0 * m * a * omega / r
            - (r * r + a * a + 2.0 * m * a * a / r) * omega * omega
        )
        g = cunningham_g_factor(r, 0.0, a=a, m=m)
        assert abs(g - np.sqrt(inv_ut2)) < 1e-12

    def test_isco_extremo_muy_corrido_al_rojo(self) -> None:
        # Gargantua (a=0.998): el ISCO esta muy cerca del horizonte y la
        # emision cara-a-cara debe estar fuertemente corrida al rojo
        isco = kerr_isco(0.998)
        g = cunningham_g_factor(isco, 0.0, a=0.998)
        assert g < 0.3


class TestMomentoAngularFoton:
    """lam = -b*n_z con n = e1 x e2 (normal al plano orbital del rayo)."""

    def test_rayo_ecuatorial(self) -> None:
        # Plano orbital = plano ecuatorial: n = +z, lam = -b
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        assert abs(photon_angular_momentum(5.0, e1, e2) - (-5.0)) < 1e-12

    def test_rayo_polar_sin_lz(self) -> None:
        # Plano orbital que contiene el eje z: n_z = 0, lam = 0
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 0.0, 1.0])
        assert abs(photon_angular_momentum(5.0, e1, e2)) < 1e-12


class TestConsistenciaKernels:
    """El kernel Numba y la version Python deben dar lo mismo."""

    def test_disk_g_factor_coincide(self) -> None:
        from src.numba_kernels import disk_g_factor

        for r in [6.0, 8.0, 15.0]:
            for lam in [-4.0, 0.0, 4.0]:
                for a in [0.0, 0.5, 0.998]:
                    assert disk_g_factor(r, lam, a, 1.0) == pytest.approx(
                        cunningham_g_factor(r, lam, a=a), rel=1e-12,
                    )


class TestPageThorne:
    """Perfil de flujo exacto del disco delgado (Page & Thorne 1974)."""

    def test_cero_en_isco_y_positivo_dentro(self) -> None:
        from src.constants import page_thorne_flux

        assert page_thorne_flux(6.0, 6.0) == 0.0
        assert page_thorne_flux(5.0, 6.0) == 0.0
        for r in [6.5, 8.0, 12.0, 20.0]:
            assert page_thorne_flux(r, 6.0) > 0.0

    def test_asintotico_shakura_sunyaev(self) -> None:
        # Lejos del ISCO el flujo debe caer como r^-3 (T ~ r^-3/4)
        from src.constants import page_thorne_flux

        f1 = page_thorne_flux(1000.0, 6.0)
        f2 = page_thorne_flux(2000.0, 6.0)
        assert f1 / f2 == pytest.approx(8.0, rel=0.05)

    def test_pico_en_zona_interna(self) -> None:
        from src.constants import page_thorne_flux

        radios = np.linspace(6.01, 20.0, 2000)
        flujos = [page_thorne_flux(r, 6.0) for r in radios]
        r_pico = radios[int(np.argmax(flujos))]
        # El pico del flujo esta entre el ISCO y ~2x ISCO
        assert 6.0 < r_pico < 12.0

    def test_consistencia_numba_python(self) -> None:
        from src.constants import page_thorne_flux
        from src.numba_kernels import page_thorne_flux as pt_numba
        from src.constants import kerr_isco

        for a in [0.0, 0.5, 0.9, 0.998]:
            r_isco = kerr_isco(a) if a > 1e-6 else 6.0
            for r in [r_isco * 1.05, r_isco * 1.5, 10.0, 19.0]:
                assert pt_numba(r, r_isco, a, 1.0) == pytest.approx(
                    page_thorne_flux(r, r_isco, a), rel=1e-10,
                )

    def test_normalizacion(self) -> None:
        from src.constants import page_thorne_flux, page_thorne_norm_inv

        inv = page_thorne_norm_inv(6.0, 20.0, 0.0)
        # La referencia es el radio medio geometrico: F(r_ref) * inv == 1
        r_ref = np.sqrt(6.0 * 20.0)
        assert page_thorne_flux(r_ref, 6.0) * inv == pytest.approx(1.0, rel=1e-12)
        # El pico interior queda por encima de 1 (nucleo HDR)
        radios = np.linspace(6.01, 20.0, 2000)
        f_norm_max = max(page_thorne_flux(r, 6.0) * inv for r in radios)
        assert f_norm_max > 1.0

    def test_spin_aumenta_eficiencia_interna(self) -> None:
        # Con mas spin el ISCO baja y el disco emite desde radios menores:
        # a igual radio r=4M, el disco de a=0.998 (ISCO~1.24) emite,
        # el de a=0 (ISCO=6) no.
        from src.constants import page_thorne_flux, kerr_isco

        r_isco_998 = kerr_isco(0.998)
        assert page_thorne_flux(4.0, r_isco_998, 0.998) > 0.0
        assert page_thorne_flux(4.0, 6.0, 0.0) == 0.0
