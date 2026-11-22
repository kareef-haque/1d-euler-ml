import math
import numpy as np
import pytest
import torch

import dg_check as dc
from euler1d.dg import solver as dg
from euler1d.neural import weighted as ws
from infrastructure.exact import ExactRiemannSolver

D = torch.float64
SOD = ((1.0, 0.0, 1e5), (0.125, 0.0, 1e4))


def si_states(B=100, seed=0):
    r = np.random.default_rng(seed)
    return torch.tensor(dg.cons(r.uniform(0.1, 2.5, B), r.normal(0, 200, B), r.uniform(1e4, 1.25e5, B)), dtype=D)[None]


@pytest.mark.parametrize('P', [1, 2, 3, 4])
def test_basis_orthonormal_and_derivative(P):
    x, w = dg.gauss(8)
    F = dg.phi(P, x)
    assert torch.allclose((F[:, None] * F[None] * w).sum(-1), torch.eye(P, dtype=D), atol=1e-12)
    fd = (dg.phi(P, x + 1e-6) - dg.phi(P, x - 1e-6)) / 2e-6
    assert torch.allclose(fd, dg.dphi(P, x), atol=1e-6)


@pytest.mark.parametrize('name', ['hllc', 'rusanov'])
def test_fluxes_consistent_and_match_fv(name):
    QL, QR = si_states(), si_states(seed=1)
    F = dg.FLUX[name]
    assert torch.allclose(F(QL, QL), dg.euler_flux(QL), rtol=1e-12)
    assert torch.allclose(F(QL, QR), getattr(ws, name)(QL, QR), rtol=1e-10)


def test_projection():
    e = np.linspace(0, 1, 11)
    U = dg.project(lambda x: dg.cons(1 + x + x**2, np.zeros_like(x), np.ones_like(x)), e, 3)
    xq, _ = dg.gauss(5)
    xc = (e[:-1] + 0.5 / 10)[:, None] + xq.numpy() / 10
    assert np.allclose(dg.at(U[None], xq)[0, 0].numpy(), 1 + xc + xc**2, atol=1e-12)
    ex = ExactRiemannSolver(*SOD, 0.2)
    e = np.linspace(0, 0.4, 41)
    t = 1e-4
    U = dg.project_exact(ex, e, t, 1)
    fine = np.linspace(0, 0.4, 40 * 4000 + 1)
    xm = 0.5 * (fine[1:] + fine[:-1])
    rho = ex.sample(xm, t)[0].reshape(40, 4000).mean(1)
    assert np.abs(U[0, :, 0].numpy() - rho).max() < 1e-3


@pytest.mark.parametrize('P', [1, 2, 3])
@pytest.mark.parametrize('flux', ['hllc', 'rusanov'])
def test_uniform_state_preserved(P, flux):
    U = torch.zeros(1, 3, 12, P, dtype=D)
    U[0, :, :, 0] = torch.tensor(dg.cons(1.3, 120.0, 7e4), dtype=D)[:, None]
    assert dg.rhs(dg.transmissive(U), 1e-3, flux).abs().max() < 1e-9 * dg.euler_flux(U[..., 0]).abs().max()
    V, _ = dg.solve(U, 1e-3, 2e-5, flux=flux)
    assert torch.allclose(V, U, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize('P', [2, 3])
def test_conservative_periodic(P):
    r = np.random.default_rng(2)
    e = np.linspace(0, 1, 33)
    U = dg.project(lambda x: dg.cons(1 + 0.3 * np.sin(2 * np.pi * x) + 0.1 * (x > 0.5), np.ones_like(x),
                                     np.ones_like(x) + 0.2 * (x < 0.3)), e, P)[None]
    V, _ = dg.solve(U, 1 / 32, 0.1, bc=dg.periodic)
    assert torch.allclose(V[..., 0].sum(-1), U[..., 0].sum(-1), rtol=1e-12)


@pytest.mark.parametrize('p,want', [(1, 2.0), (2, 3.0)])
def test_convergence_order(p, want):
    errs, rates = dc.convergence(p, Ns=(10, 20, 40), T=0.25)
    assert abs(rates[-1] - want) < 0.15
    errs_l, rates_l = dc.convergence(p, M=100.0, Ns=(10, 20, 40), T=0.25)
    assert np.allclose(errs_l, errs)


def test_tvb_limits_a_jump_and_keeps_smooth_data():
    e = np.linspace(0, 1, 21)
    U = dg.project(lambda x: dg.cons(1 + (x > 0.5), np.zeros_like(x), np.ones_like(x)), e, 3)[None]
    L = dg.tvb(dg.transmissive(U), 1 / 20, 0.0)
    assert L[0, 0, 9:11, 1:].abs().max() < U[0, 0, 9:11, 1:].abs().max()
    assert torch.allclose(L[..., 0], U[..., 0])
    S = dg.project(dc.density_wave, e, 3)[None]
    assert torch.allclose(dg.tvb(dg.periodic(S), 1 / 20, 100.0), S)


def test_positivity_limiter_fixes_point_values_and_keeps_means():
    U = torch.zeros(1, 3, 1, 2, dtype=D)
    U[0, :, 0, 0] = torch.tensor(dg.cons(0.1, 0.0, 0.01), dtype=D)
    U[0, 0, 0, 1] = 0.2
    V = dg.positivity(U)
    pts = dg.at(V, torch.tensor([-0.5, 0.5], dtype=D))
    assert (pts[:, 0] > 0).all() and (dg.pressure(pts) > 0).all() and torch.equal(V[..., 0], U[..., 0])


@pytest.mark.parametrize('P', [2, 3])
def test_123_stays_positive(P):
    (pl, pr), Lx, T = dc.PROBLEMS['123']
    ex = ExactRiemannSolver(pl, pr, 0.5, 1.4)
    e = np.linspace(0, 1, 61)
    U, _ = dg.solve(dg.project_exact(ex, e, 0.0, P)[None], 1 / 60, 0.1)
    xq, _ = dg.gauss(P + 1)
    V = dg.at(U, torch.cat([torch.tensor([-0.5, 0.5], dtype=D), xq]))
    assert (V[:, 0] > 0).all() and (dg.pressure(V) > 0).all()


def test_sod_more_accurate_than_first_order_fv():
    err, T, n = dc.riemann('sod', 100, 1)
    assert err < 0.5 * dc.fv('sod', 100)['HLLC-1']
