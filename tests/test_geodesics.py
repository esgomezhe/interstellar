"""Tests de validacion fisica para la integracion de geodesicas."""

import numpy as np
import pytest

from src.constants import RS, B_CRIT, R_CAMERA, R_PHOTON
from src.geodesics import trace_geodesic, classical_deflection, RayFate


class TestDeflexionCampoDebil:
    """Rayos con b >> b_crit deben desviarse ~4M/b (formula de Einstein)."""

    @pytest.mark.parametrize("b", [30.0, 50.0, 80.0, 100.0])
    def test_rayos_lejanos_escapan(self, b: float) -> None:
        result = trace_geodesic(b, R_CAMERA)
        assert result.fate == RayFate.ESCAPED

    @pytest.mark.parametrize("b", [50.0, 80.0, 100.0])
    def test_deflexion_disminuye_con_b(self, b: float) -> None:
        r1 = trace_geodesic(b, R_CAMERA)
        r2 = trace_geodesic(b * 2, R_CAMERA)
        # Mayor b debe tener menos recorrido total en phi
        assert r1.phi[-1] >= r2.phi[-1]


class TestRayosCriticos:
    """Rayos cerca de b_crit deben orbitar varias veces cerca de la esfera de fotones."""

    def test_ligeramente_arriba_bcrit_escapa(self) -> None:
        result = trace_geodesic(B_CRIT * 1.001, R_CAMERA, phi_max=100.0)
        assert result.fate == RayFate.ESCAPED

    def test_ligeramente_abajo_bcrit_capturado(self) -> None:
        result = trace_geodesic(B_CRIT * 0.999, R_CAMERA, phi_max=100.0)
        assert result.fate == RayFate.CAPTURED

    def test_rayo_critico_orbita_cerca_esfera_fotones(self) -> None:
        result = trace_geodesic(B_CRIT * 1.001, R_CAMERA, phi_max=100.0)
        r_min = result.r.min()
        # Debe acercarse a la esfera de fotones (r = 3M)
        assert r_min < R_PHOTON * 1.5
        assert r_min > RS  # pero sin caer

    def test_rayo_critico_multiples_orbitas(self) -> None:
        result = trace_geodesic(B_CRIT * 1.0001, R_CAMERA, phi_max=100.0)
        assert result.n_orbits > 1.0


class TestRayosCapturados:
    """Rayos con b < b_crit deben caer al agujero negro."""

    @pytest.mark.parametrize("b", [1.0, 2.0, 3.0, 4.0, B_CRIT * 0.9])
    def test_capturado(self, b: float) -> None:
        result = trace_geodesic(b, R_CAMERA, phi_max=50.0)
        assert result.fate == RayFate.CAPTURED

    @pytest.mark.parametrize("b", [1.0, 2.0, 3.0, 4.0])
    def test_llega_al_horizonte(self, b: float) -> None:
        result = trace_geodesic(b, R_CAMERA, phi_max=50.0)
        # El rayo debe llegar cerca del horizonte de eventos
        assert result.r.min() <= RS * 1.1


class TestTamanoSombra:
    """La frontera de la sombra debe estar en b_crit = 3*sqrt(3)*M."""

    def test_valor_bcrit(self) -> None:
        expected = 3.0 * np.sqrt(3.0)  # M=1
        assert abs(B_CRIT - expected) < 1e-10

    def test_frontera_sombra_separa_destinos(self) -> None:
        b_arriba = B_CRIT * 1.01
        b_abajo = B_CRIT * 0.99

        r_arriba = trace_geodesic(b_arriba, R_CAMERA, phi_max=100.0)
        r_abajo = trace_geodesic(b_abajo, R_CAMERA, phi_max=100.0)

        assert r_arriba.fate == RayFate.ESCAPED
        assert r_abajo.fate == RayFate.CAPTURED


class TestDeflexionClasica:
    """Test de la formula analitica de campo debil."""

    def test_formula(self) -> None:
        # dphi = 2*rs/b = 4M/b (ya que rs=2M)
        assert abs(classical_deflection(100.0) - 0.04) < 1e-10
        assert abs(classical_deflection(50.0) - 0.08) < 1e-10
