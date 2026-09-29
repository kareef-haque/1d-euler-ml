#!/usr/bin/env python
"""
Animate neural flux vs WENO5+HLLC on Sod shock tube with real-time metrics:
  - RMSE (vs exact solution) for each variable
  - Shock position tracking (actual vs exact)
  - Contact position tracking (actual vs exact)

Usage:
    python animate_with_metrics.py \\
        --model /home/claude/repo/runs/demo/model_best.pt \\
        --output sod_with_metrics.mp4 \\
        [--fps 15] [--include-exact]
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import sys

sys.path.insert(0, '/home/claude/repo')

from euler1d.solver import EulerSolver
from euler1d.neural.adapter import NeuralFluxScheme
from infrastructure.solver_config import EulerConfig
from infrastructure.results import ExactRiemannSolver


def sod_config(N=200, dt=5e-6, t_max=5e-4):
    """Standard Sod shock tube."""
    rho_L, u_L, p_L = 1.0, 0.0, 1.0
    rho_R, u_R, p_R = 0.125, 0.0, 0.1
    
    IC = np.zeros((3, N))
    IC[0, :N//2] = rho_L
    IC[0, N//2:] = rho_R
    IC[1, :N//2] = rho_L * u_L
    IC[1, N//2:] = rho_R * u_R
    IC[2, :N//2] = p_L / 0.4 + 0.5 * rho_L * u_L**2
    IC[2, N//2:] = p_R / 0.4 + 0.5 * rho_R * u_R**2
    
    return EulerConfig(
        domain_size=1.0,
        N_cells=N,
        IC=IC,
        t_max=t_max,
        dt=dt,
        BC='Zero-Gradient',
        N_ghost=3,
        gamma=1.4
    )


def run_solver(config, flux, name, **kwargs):
    """Run solver, return (rho, u, p), t_hist, x_grid."""
    print(f"  {name:20s}...", end=' ', flush=True)
    res = EulerSolver(config, flux=flux, time_integrator='SSPRK3', verbose=False, **kwargs)
    x_grid = np.linspace(0, config.domain_size, config.N_cells)
    
    Q_hist = np.array(res.Q_hist)
    t_hist = np.array(res.t_hist)
    
    rho = Q_hist[:, 0, :]
    u = Q_hist[:, 1, :] / rho
    p = 0.4 * (Q_hist[:, 2, :] - 0.5 * rho * u**2)
    
    print(f"✓ {len(t_hist)} steps")
    return (rho, u, p), t_hist, x_grid


def get_exact_solution(x_grid, t):
    """Sample exact Riemann solution at time t."""
    solver = ExactRiemannSolver(
        state_L=(1.0, 0.0, 1.0),
        state_R=(0.125, 0.0, 0.1),
        x_interface=0.5,
        gamma=1.4
    )
    rho, u, p = solver.sample(x_grid, t)
    return rho, u, p


def find_shock_position(rho, x_grid, threshold_frac=0.5):
    """
    Find shock position by finding the steepest gradient in density.
    Returns x position of shock, or None if not found.
    """
    drho = np.abs(np.gradient(rho))
    if drho.max() < 1e-6:
        return None
    # Find where gradient is significant (above threshold)
    steep = drho > threshold_frac * drho.max()
    if np.any(steep):
        # Return position of steepest gradient
        idx = np.argmax(drho)
        return x_grid[idx]
    return None


def find_contact_position(rho, x_grid, threshold_frac=0.3):
    """
    Find contact discontinuity by detecting large but not-steepest density jump.
    Returns x position of contact, or None if not found.
    """
    drho = np.abs(np.gradient(rho))
    if drho.max() < 1e-6:
        return None
    # Look for jumps that are significant but not the steepest
    steep_threshold = 0.3 * drho.max()
    candidates = np.where(drho > steep_threshold)[0]
    if len(candidates) > 0:
        # Pick the one closest to x=0.5 (interface) but not the steepest
        dists = np.abs(x_grid[candidates] - 0.5)
        idx = candidates[np.argmin(dists)]
        return x_grid[idx]
    return None


def compute_exact_features(x_grid, t):
    """
    Compute shock and contact positions from exact solution.
    Returns (shock_x, contact_x) or (None, None) if not found.
    """
    rho_ex, u_ex, p_ex = get_exact_solution(x_grid, t)
    shock_x = find_shock_position(rho_ex, x_grid, threshold_frac=0.5)
    contact_x = find_contact_position(rho_ex, x_grid, threshold_frac=0.3)
    return shock_x, contact_x


def compute_rmse(rho_num, u_num, p_num, rho_ex, u_ex, p_ex):
    """Compute RMSE for each variable vs exact solution."""
    rmse_rho = np.sqrt(np.mean((rho_num - rho_ex)**2))
    rmse_u = np.sqrt(np.mean((u_num - u_ex)**2))
    rmse_p = np.sqrt(np.mean((p_num - p_ex)**2))
    return rmse_rho, rmse_u, rmse_p


def create_animation_with_metrics(model_path, output_path, fps=15, include_exact=False):
    """Create animation with real-time metrics."""
    
    cfg = sod_config(N=200, dt=5e-6, t_max=5e-4)
    x_grid = np.linspace(0, cfg.domain_size, cfg.N_cells)
    
    # Run solvers
    print("\nRunning solvers:")
    baseline, t_hist, _ = run_solver(cfg, 'HLLC', 'WENO5+HLLC', reconstruction='WENO5')
    neural_scheme = NeuralFluxScheme(model_path)
    neural, _, _ = run_solver(cfg, 'Neural', 'Neural', neural_scheme=neural_scheme)
    
    # Trim to same length
    T = min(len(t_hist), len(baseline[0]), len(neural[0]))
    baseline = tuple(d[:T] for d in baseline)
    neural = tuple(d[:T] for d in neural)
    t_hist = t_hist[:T]
    
    print(f"\nTimesteps: {T}, fps: {fps}")
    
    # Precompute metrics for all timesteps
    print("Computing metrics...")
    metrics_base = {'rmse_rho': [], 'rmse_u': [], 'rmse_p': [],
                    'shock_x': [], 'contact_x': []}
    metrics_neur = {'rmse_rho': [], 'rmse_u': [], 'rmse_p': [],
                    'shock_x': [], 'contact_x': []}
    exact_features = {'shock_x': [], 'contact_x': []}
    
    for frame in range(T):
        t = t_hist[frame]
        
        # Exact solution
        rho_ex, u_ex, p_ex = get_exact_solution(x_grid, t)
        shock_x_ex, contact_x_ex = compute_exact_features(x_grid, t)
        exact_features['shock_x'].append(shock_x_ex)
        exact_features['contact_x'].append(contact_x_ex)
        
        # Baseline metrics
        rho_base, u_base, p_base = baseline[0][frame], baseline[1][frame], baseline[2][frame]
        rmse_rho, rmse_u, rmse_p = compute_rmse(rho_base, u_base, p_base, rho_ex, u_ex, p_ex)
        metrics_base['rmse_rho'].append(rmse_rho)
        metrics_base['rmse_u'].append(rmse_u)
        metrics_base['rmse_p'].append(rmse_p)
        metrics_base['shock_x'].append(find_shock_position(rho_base, x_grid))
        metrics_base['contact_x'].append(find_contact_position(rho_base, x_grid))
        
        # Neural metrics
        rho_neur, u_neur, p_neur = neural[0][frame], neural[1][frame], neural[2][frame]
        rmse_rho, rmse_u, rmse_p = compute_rmse(rho_neur, u_neur, p_neur, rho_ex, u_ex, p_ex)
        metrics_neur['rmse_rho'].append(rmse_rho)
        metrics_neur['rmse_u'].append(rmse_u)
        metrics_neur['rmse_p'].append(rmse_p)
        metrics_neur['shock_x'].append(find_shock_position(rho_neur, x_grid))
        metrics_neur['contact_x'].append(find_contact_position(rho_neur, x_grid))
    
    # Convert to arrays
    for key in metrics_base:
        metrics_base[key] = np.array(metrics_base[key])
        metrics_neur[key] = np.array(metrics_neur[key])
    for key in exact_features:
        exact_features[key] = np.array(exact_features[key])
    
    # Figure layout: 3 rows (ρ,u,p) + 2 metric rows (RMSE, positions)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(5, 3, hspace=0.35, wspace=0.3, top=0.95, bottom=0.05)
    
    
    # State variable plots (top 3 rows)
    axes_state = []
    labels = ['Density ρ', 'Velocity u', 'Pressure p']
    for row in range(3):
        ax_base = fig.add_subplot(gs[row, 0])
        ax_neur = fig.add_subplot(gs[row, 1])
        ax_exact = fig.add_subplot(gs[row, 2]) if include_exact else None
        
        ax_base.set_ylabel(labels[row], fontsize=11, weight='bold')
        ax_base.set_xlim(0, 1.0)
        ax_neur.set_xlim(0, 1.0)
        if ax_exact:
            ax_exact.set_xlim(0, 1.0)
        
        ax_base.grid(True, alpha=0.3)
        ax_neur.grid(True, alpha=0.3)
        if ax_exact:
            ax_exact.grid(True, alpha=0.3)
        
        ax_base.set_title(f'{labels[row]} (WENO5+HLLC)', fontsize=10)
        ax_neur.set_title(f'{labels[row]} (Neural)', fontsize=10)
        if ax_exact:
            ax_exact.set_title(f'{labels[row]} (Exact)', fontsize=10)
        
        # Set y-limits
        var_idx = row
        base_var, neur_var = baseline[var_idx], neural[var_idx]
        rho_ex_all, u_ex_all, p_ex_all = [], [], []
        for t in t_hist:
            r, u, p = get_exact_solution(x_grid, t)
            rho_ex_all.append(r)
            u_ex_all.append(u)
            p_ex_all.append(p)
        
        rho_ex_all = np.array(rho_ex_all)
        u_ex_all = np.array(u_ex_all)
        p_ex_all = np.array(p_ex_all)
        
        all_vars = [base_var, neur_var] + ([rho_ex_all, u_ex_all, p_ex_all][var_idx:var_idx+1] if ax_exact else [])
        vmin = min(v.min() for v in all_vars)
        vmax = max(v.max() for v in all_vars)
        margin = 0.05 * (vmax - vmin) if vmax > vmin else 1.0
        
        ax_base.set_ylim(vmin - margin, vmax + margin)
        ax_neur.set_ylim(vmin - margin, vmax + margin)
        if ax_exact:
            ax_exact.set_ylim(vmin - margin, vmax + margin)
        
        ax_base.set_xlabel('x' if row == 2 else '')
        ax_neur.set_xlabel('x' if row == 2 else '')
        if ax_exact:
            ax_exact.set_xlabel('x' if row == 2 else '')
        
        axes_state.append({'base': ax_base, 'neur': ax_neur, 'exact': ax_exact, 'var': var_idx})
    
    # RMSE comparison (row 3)
    ax_rmse = fig.add_subplot(gs[3, :])
    ax_rmse.set_ylabel('RMSE', fontsize=11, weight='bold')
    ax_rmse.set_xlabel('Time', fontsize=10)
    ax_rmse.grid(True, alpha=0.3)
    ax_rmse.set_title('RMSE vs Exact Solution', fontsize=10)
    
    line_rmse_base_rho, = ax_rmse.plot([], [], color='#1f77b4', lw=2, label='WENO5+HLLC (ρ)', linestyle='-')
    line_rmse_base_u, = ax_rmse.plot([], [], color='#1f77b4', lw=2, label='WENO5+HLLC (u)', linestyle='--')
    line_rmse_base_p, = ax_rmse.plot([], [], color='#1f77b4', lw=2, label='WENO5+HLLC (p)', linestyle=':')
    line_rmse_neur_rho, = ax_rmse.plot([], [], color='#ff7f0e', lw=2, label='Neural (ρ)', linestyle='-')
    line_rmse_neur_u, = ax_rmse.plot([], [], color='#ff7f0e', lw=2, label='Neural (u)', linestyle='--')
    line_rmse_neur_p, = ax_rmse.plot([], [], color='#ff7f0e', lw=2, label='Neural (p)', linestyle=':')
    
    ax_rmse.legend(loc='upper left', fontsize=8, ncol=3)
    ax_rmse.set_xlim(0, t_hist[-1])
    
    # Feature tracking (row 4)
    ax_feat = fig.add_subplot(gs[4, :])
    ax_feat.set_ylabel('Position', fontsize=11, weight='bold')
    ax_feat.set_xlabel('Time', fontsize=10)
    ax_feat.grid(True, alpha=0.3)
    ax_feat.set_title('Shock & Contact Positions', fontsize=10)
    
    line_shock_exact, = ax_feat.plot([], [], color='black', lw=2, linestyle='--', label='Exact shock')
    line_shock_base, = ax_feat.plot([], [], color='#1f77b4', lw=1.5, marker='o', markersize=3, label='WENO5+HLLC shock')
    line_shock_neur, = ax_feat.plot([], [], color='#ff7f0e', lw=1.5, marker='s', markersize=3, label='Neural shock')
    
    line_cont_exact, = ax_feat.plot([], [], color='gray', lw=2, linestyle='--', label='Exact contact')
    line_cont_base, = ax_feat.plot([], [], color='#1f77b4', lw=1.5, marker='^', markersize=3, label='WENO5+HLLC contact')
    line_cont_neur, = ax_feat.plot([], [], color='#ff7f0e', lw=1.5, marker='v', markersize=3, label='Neural contact')
    
    ax_feat.legend(loc='upper left', fontsize=8, ncol=3)
    ax_feat.set_xlim(0, t_hist[-1])
    ax_feat.set_ylim(-0.05, 1.05)
    
    # Time text
    time_text = fig.text(0.5, 0.01, '', ha='center', fontsize=11, weight='bold')
    
    # Animation update
    def update(frame):
        t = t_hist[frame]
        
        # Update state variables
        for ax_info in axes_state:
            var_idx = ax_info['var']
            base_var = baseline[var_idx][frame]
            neur_var = neural[var_idx][frame]
            
            # Plot lines (will be created on first call)
            if not hasattr(ax_info['base'], '_line'):
                ax_info['base']._line, = ax_info['base'].plot([], [], color='#1f77b4', lw=2)
                ax_info['neur']._line, = ax_info['neur'].plot([], [], color='#ff7f0e', lw=2)
                if ax_info['exact']:
                    ax_info['exact']._line, = ax_info['exact'].plot([], [], color='black', lw=1.5, linestyle='--')
            
            ax_info['base']._line.set_data(x_grid, base_var)
            ax_info['neur']._line.set_data(x_grid, neur_var)
            
            if ax_info['exact']:
                if var_idx == 0:
                    rho_ex, _, _ = get_exact_solution(x_grid, t)
                    ax_info['exact']._line.set_data(x_grid, rho_ex)
                elif var_idx == 1:
                    _, u_ex, _ = get_exact_solution(x_grid, t)
                    ax_info['exact']._line.set_data(x_grid, u_ex)
                else:
                    _, _, p_ex = get_exact_solution(x_grid, t)
                    ax_info['exact']._line.set_data(x_grid, p_ex)
        
        # Update RMSE plot (cumulative)
        times_so_far = t_hist[:frame+1]
        line_rmse_base_rho.set_data(times_so_far, metrics_base['rmse_rho'][:frame+1])
        line_rmse_base_u.set_data(times_so_far, metrics_base['rmse_u'][:frame+1])
        line_rmse_base_p.set_data(times_so_far, metrics_base['rmse_p'][:frame+1])
        line_rmse_neur_rho.set_data(times_so_far, metrics_neur['rmse_rho'][:frame+1])
        line_rmse_neur_u.set_data(times_so_far, metrics_neur['rmse_u'][:frame+1])
        line_rmse_neur_p.set_data(times_so_far, metrics_neur['rmse_p'][:frame+1])
        
        # Auto-scale RMSE y-axis
        all_rmse = np.concatenate([
            metrics_base['rmse_rho'][:frame+1],
            metrics_base['rmse_u'][:frame+1],
            metrics_base['rmse_p'][:frame+1],
            metrics_neur['rmse_rho'][:frame+1],
            metrics_neur['rmse_u'][:frame+1],
            metrics_neur['rmse_p'][:frame+1]
        ])
        if len(all_rmse) > 0 and all_rmse.max() > 0:
            ax_rmse.set_ylim(0, all_rmse.max() * 1.1)
        
        # Update feature tracking
        times_so_far = t_hist[:frame+1]
        
        # Shock positions
        shock_ex_so_far = []
        shock_base_so_far = []
        shock_neur_so_far = []
        for i in range(frame+1):
            shock_ex_so_far.append(exact_features['shock_x'][i])
            shock_base_so_far.append(metrics_base['shock_x'][i])
            shock_neur_so_far.append(metrics_neur['shock_x'][i])
        
        # Filter out None values for plotting
        shock_ex_times = [t for t, x in zip(times_so_far, shock_ex_so_far) if x is not None]
        shock_ex_pos = [x for x in shock_ex_so_far if x is not None]
        shock_base_times = [t for t, x in zip(times_so_far, shock_base_so_far) if x is not None]
        shock_base_pos = [x for x in shock_base_so_far if x is not None]
        shock_neur_times = [t for t, x in zip(times_so_far, shock_neur_so_far) if x is not None]
        shock_neur_pos = [x for x in shock_neur_so_far if x is not None]
        
        line_shock_exact.set_data(shock_ex_times, shock_ex_pos)
        line_shock_base.set_data(shock_base_times, shock_base_pos)
        line_shock_neur.set_data(shock_neur_times, shock_neur_pos)
        
        # Contact positions
        cont_ex_so_far = []
        cont_base_so_far = []
        cont_neur_so_far = []
        for i in range(frame+1):
            cont_ex_so_far.append(exact_features['contact_x'][i])
            cont_base_so_far.append(metrics_base['contact_x'][i])
            cont_neur_so_far.append(metrics_neur['contact_x'][i])
        
        cont_ex_times = [t for t, x in zip(times_so_far, cont_ex_so_far) if x is not None]
        cont_ex_pos = [x for x in cont_ex_so_far if x is not None]
        cont_base_times = [t for t, x in zip(times_so_far, cont_base_so_far) if x is not None]
        cont_base_pos = [x for x in cont_base_so_far if x is not None]
        cont_neur_times = [t for t, x in zip(times_so_far, cont_neur_so_far) if x is not None]
        cont_neur_pos = [x for x in cont_neur_so_far if x is not None]
        
        line_cont_exact.set_data(cont_ex_times, cont_ex_pos)
        line_cont_base.set_data(cont_base_times, cont_base_pos)
        line_cont_neur.set_data(cont_neur_times, cont_neur_pos)
        
        # Time text with summary statistics
        rmse_base_curr = np.mean([metrics_base['rmse_rho'][frame], 
                                   metrics_base['rmse_u'][frame],
                                   metrics_base['rmse_p'][frame]])
        rmse_neur_curr = np.mean([metrics_neur['rmse_rho'][frame],
                                  metrics_neur['rmse_u'][frame],
                                  metrics_neur['rmse_p'][frame]])
        
        improvement = (rmse_base_curr - rmse_neur_curr) / max(rmse_base_curr, 1e-12) * 100
        
        time_text.set_text(
            f't = {t:.6f} s  |  Step {frame+1}/{T}  |  ' +
            f'RMSE: WENO5 {rmse_base_curr:.2e}  Neural {rmse_neur_curr:.2e}  ' +
            f'({improvement:+.1f}%)'
        )
        
        all_artists = []
        for ax_info in axes_state:
            if hasattr(ax_info['base'], '_line'):
                all_artists.append(ax_info['base']._line)
                all_artists.append(ax_info['neur']._line)
                if ax_info['exact']:
                    all_artists.append(ax_info['exact']._line)
        
        all_artists.extend([
            line_rmse_base_rho, line_rmse_base_u, line_rmse_base_p,
            line_rmse_neur_rho, line_rmse_neur_u, line_rmse_neur_p,
            line_shock_exact, line_shock_base, line_shock_neur,
            line_cont_exact, line_cont_base, line_cont_neur,
            time_text
        ])
        
        return all_artists
    
    print(f"Creating animation... ({T} frames)")
    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=True, repeat=True)
    
    writer = FFMpegWriter(fps=fps, bitrate=2400, codec='libx264')
    anim.save(output_path, writer=writer, dpi=100)
    
    print(f"\n✓ Saved to {output_path}")
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Animate neural vs WENO5+HLLC with metrics'
    )
    parser.add_argument('--model', type=str,
                        default='/home/claude/repo/runs/demo/model_best.pt',
                        help='Path to neural model checkpoint')
    parser.add_argument('--output', type=str, default='sod_with_metrics.mp4',
                        help='Output video path')
    parser.add_argument('--fps', type=int, default=15, help='Frames per second')
    parser.add_argument('--include-exact', action='store_true',
                        help='Include exact solution in third column')
    
    args = parser.parse_args()
    
    create_animation_with_metrics(args.model, args.output, args.fps, args.include_exact)