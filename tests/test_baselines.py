import numpy as np
import pytest

from infrastructure.exact import ExactRiemannSolver
from infrastructure.solver_config import EulerConfig
from euler1d.solver import EulerSolver
from euler1d.schemes.reconstruction.constant import Const
from euler1d.schemes.reconstruction.weno5Z import WENO5Z

G = 1.4
SOD_L = (1.0, 0.0, 1e5)
SOD_R = (0.125, 0.0, 1e4)


def cons(rho, u, p):
    return np.array([rho, rho * u, p / (G - 1) + 0.5 * rho * u**2])


def flux(rho, u, p):
    return np.array([rho * u, rho * u**2 + p, u * (p / (G - 1) + 0.5 * rho * u**2 + p)])


def run(flux_name, rec, integ, N=400, dt=1.5e-6, t_max=6e-4):
    Q = np.zeros((3, N))
    Q[:, :N // 2] = cons(*SOD_L)[:, None]
    Q[:, N // 2:] = cons(*SOD_R)[:, None]
    cfg = EulerConfig(domain_size=1.0, N_cells=N, IC=Q, t_max=t_max, dt=dt)
    res = EulerSolver(cfg, flux=flux_name, reconstruction=rec, integrator=integ, verbose=False)
    ex = ExactRiemannSolver(SOD_L, SOD_R, 0.5, G)
    return res, ex.cell_avg(np.linspace(0, 1, N + 1), t_max)


def l1(res, Qe):
    return np.mean(np.abs(res.Q_hist[-1][0] - Qe[0]))


def test_sod_star_state():
    ex = ExactRiemannSolver(SOD_L, SOD_R, 0.5, G)
    assert ex.P_star == pytest.approx(0.30313 * 1e5, rel=1e-3)
    assert ex.u_star == pytest.approx(0.92745 * np.sqrt(1e5), rel=1e-3)
    assert ex.rho_star_L == pytest.approx(0.42632, rel=1e-3)
    assert ex.rho_star_R == pytest.approx(0.26557, rel=1e-3)


def test_exact_conservation_random():
    rng = np.random.default_rng(0)
    t, N = 2e-4, 8000
    e = np.linspace(-1.0, 1.0, N + 1)
    for _ in range(30):
        sL = (rng.uniform(0.1, 2.5), rng.normal(0, 5), rng.uniform(1e4, 1.25e5))
        sR = (rng.uniform(0.1, 2.5), rng.normal(0, 5), rng.uniform(1e4, 1.25e5))
        Q = ExactRiemannSolver(sL, sR, 0.0, G).cell_avg(e, t)
        tot = Q.sum(axis=1) * (2.0 / N)
        fL, fR = flux(*sL), flux(*sR)
        want = cons(*sL) + cons(*sR) + t * (fL - fR)
        scale = np.abs(cons(*sL)) + np.abs(cons(*sR)) + t * (np.abs(fL) + np.abs(fR))
        assert np.all(np.abs(tot - want) < 5e-4 * scale)


def test_exact_wave_structure():
    ex = ExactRiemannSolver(SOD_L, SOD_R, 0.5, G)
    x = np.linspace(0, 1, 2001)
    rho, u, p = ex.sample(x, 6e-4)
    assert np.all(np.isfinite(rho)) and rho.min() > 0 and p.min() > 0
    assert rho[0] == 1.0 and rho[-1] == 0.125
    assert np.abs(np.diff(rho)).max() > 0.1 and np.abs(np.diff(p)).max() > 0.5e4
    assert np.all(np.diff(rho[x < 0.25]) == 0)


def test_time_axis_matches_states():
    res, _ = run('HLLC', 'Const', 'euler', N=100, dt=4e-6, t_max=1e-4)
    t = np.array(res.t_hist)
    assert len(t) == len(res.Q_hist) == len(res.F_hist)
    assert np.allclose(t, np.minimum(np.arange(len(t)) * 4e-6, 1e-4))
    assert t[-1] == pytest.approx(1e-4)


def test_config_floor_and_copy():
    Q = np.zeros((3, 8))
    cfg = EulerConfig(domain_size=1.0, N_cells=8, IC=Q)
    assert cfg.IC[0].min() == 1e-9 and cfg.IC[2].min() == 1e-9
    assert Q.max() == 0.0


def test_const_shapes():
    Q = np.random.rand(3, 106)
    QL, QR = Const(Q)
    assert QL.shape == QR.shape == (3, 101)
    assert np.array_equal(QL[:, 0], Q[:, 2]) and np.array_equal(QR[:, 0], Q[:, 3])


def test_weno5z_keeps_negative_momentum():
    Q = np.tile(cons(1.0, -50.0, 1e5)[:, None], (1, 20))
    QL, QR = WENO5Z(Q, dx=0.01)
    assert np.allclose(QL[1], Q[1, 0]) and np.allclose(QR[1], Q[1, 0])


@pytest.mark.parametrize('name', ['HLLC', 'Rusanov', 'AUSM+'])
def test_first_order_baselines(name):
    res, Qe = run(name, 'Const', 'euler')
    Qn = res.Q_hist[-1]
    p = (G - 1) * (Qn[2] - 0.5 * Qn[1]**2 / Qn[0])
    assert np.all(np.isfinite(Qn)) and Qn[0].min() > 0 and p.min() > 0
    dx = 1 / 400
    assert Qn[0].sum() * dx == pytest.approx(0.5 * 1.0 + 0.5 * 0.125, rel=1e-12)
    assert Qn[2].sum() * dx == pytest.approx(0.5 * (cons(*SOD_L)[2] + cons(*SOD_R)[2]), rel=1e-12)
    err = l1(res, Qe)
    print(f"\n{name}+Const+euler  L1(rho) = {err:.4e}")
    assert err < 0.03


def test_hllc_sharper_than_rusanov():
    assert l1(*run('HLLC', 'Const', 'euler')) < l1(*run('Rusanov', 'Const', 'euler'))


@pytest.mark.parametrize('flux_name,rec', [('HLLC', 'WENO5Z'), ('AUSM+', 'WENO5')])
def test_weno_beats_first_order(flux_name, rec):
    e1 = l1(*run(flux_name, 'Const', 'euler'))
    e5 = l1(*run(flux_name, rec, 'rk4'))
    print(f"\n{flux_name}+{rec}+rk4  L1(rho) = {e5:.4e}  (first order {e1:.4e})")
    assert e5 < e1