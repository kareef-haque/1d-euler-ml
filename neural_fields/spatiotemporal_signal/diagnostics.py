import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import yaml

from neural_fields.spatiotemporal_signal.models import INR, SIREN, WIRE
from infrastructure.results import ExactRiemannSolver


def visualize_model(model, 
                    phys_config = r'.\neural_fields\spatiotemporal_signal\configs\physics.yaml', 
                    device = 'cpu'):
    
    N_samp = 500
    x_samp = np.linspace(-1, 1, N_samp)
    t_samp = np.linspace(0, 1, 1000)


    model = model.to(device)
    model.eval()

    rho_hist = []
    u_hist = []
    P_hist = []

    with open(phys_config, 'r') as file:
        config = yaml.safe_load(file)

    mean = config['normalize']['mean']
    std = config['normalize']['std']

    
    for t in t_samp:
        t_match = t * torch.from_numpy(np.ones_like(x_samp))
        coord = torch.stack([torch.from_numpy(x_samp), t_match], axis=1).to(device, non_blocking=True) # (N_samp, 2)

        R_pred, _ = model(coord) # (N_samp, 3)
        R = R_pred.detach().cpu().T

        rho = R[0, :]*std[0] + mean[0]
        u = R[1, :]*std[1] + mean[1]
        P = R[2, :]*std[2] + mean[2]

        rho_hist.append(rho)
        u_hist.append(u)
        P_hist.append(P)

    

    # animate_pred_results(rho_hist, u_hist, P_hist, x_samp, t_samp, N_samp)

        # Standard Sod Shock Tube problem (or custom initial states)
    rho_L, u_L, P_L = 1.0, 0, 100000.0
    rho_R, u_R, P_R = 0.125, 0, 10000.0

    # Primitive states for Exact Solver
    state_L = (rho_L, u_L, P_L)
    state_R = (rho_R, u_R, P_R)

    exact_solver = ExactRiemannSolver(state_L, state_R, x_interface=0.5, gamma=1.4)


    animate_comparison(rho_hist, u_hist, P_hist, x_samp, t_samp, N_samp, exact_solver)

def animate_pred_results(rho_hist, u_hist, p_hist,
                         x, t_hist, N_samp, 
                         interval: int = 30, save_path: str = None):
    '''
    Animates Density, Velocity, and Pressure profiles over time.

    :param EulerResults results: Results dataclass from EulerSolver
    :param int interval: Delay between frames in milliseconds (default: 30ms)
    :param str save_path: Optional path to save as .mp4 or .gif (e.g. 'sod_shock.gif')
    '''



    # Extract mesh and time history
    # N_cells = results.config.N_cells
    domain_size = 1

    # Cell center coordinates
    x = np.linspace(0, 1, N_samp)
    t_hist = t_hist



    # Set up figure and 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    fig.suptitle('1D Euler Equation Dynamics', fontsize=14, fontweight='bold')

    # Line initializations
    line_rho, = ax1.plot([], [], 'b-', lw=2, label=r'Density ($\rho$)')
    line_u,   = ax2.plot([], [], 'r-', lw=2, label=r'Velocity ($u$)')
    line_p,   = ax3.plot([], [], 'g-', lw=2, label=r'Pressure ($P$)')

    # Configure axes limits
    for ax, label in zip([ax1, ax2, ax3], ['Density', 'Velocity', 'Pressure']):
        ax.set_xlim(x[0], x[-1])
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right')

    # Dynamic dynamic limits with padding
    rho_flat = np.concatenate(rho_hist)
    u_flat   = np.concatenate(u_hist)
    p_flat   = np.concatenate(p_hist)

    ax1.set_ylim(np.min(rho_flat) - 0.1, np.max(rho_flat) + 0.1)
    ax2.set_ylim(np.min(u_flat) - 0.5,   np.max(u_flat) + 0.5)
    ax3.set_ylim(np.min(p_flat) - 0.1,   np.max(p_flat) + 0.1)

    ax3.set_xlabel('x')
    time_text = ax1.text(0.02, 0.85, '', transform=ax1.transAxes, 
                         fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def init():
        line_rho.set_data([], [])
        line_u.set_data([], [])
        line_p.set_data([], [])
        time_text.set_text('')
        return line_rho, line_u, line_p, time_text

    def update(frame):
        t = t_hist[frame]
        
        line_rho.set_data(x, rho_hist[frame])
        line_u.set_data(x, u_hist[frame])
        line_p.set_data(x, p_hist[frame])
        
        time_text.set_text(f'Time: {t:.4f} s')
        return line_rho, line_u, line_p, time_text

    anim = FuncAnimation(
        fig, 
        update, 
        frames=len(t_hist),
        init_func=init, 
        blit=True, 
        interval=interval
    )

    if save_path:
        anim.save(save_path, writer='ffmpeg' if save_path.endswith('.mp4') else 'pillow')

    plt.tight_layout()
    plt.show()

    return anim


def animate_comparison(rho_hist, u_hist, p_hist,
                         x, t_hist, N_samp, 
                         exact_solver, interval=30):

    gamma = 1.4

        # Extract mesh and time history
    # N_cells = results.config.N_cells
    domain_size = 1


    # Cell center coordinates
    x_num = np.linspace(0, 1, N_samp)
    x_exact = np.linspace(0, domain_size, 1000)
    t_hist = np.linspace(0, 1000, 1000)*1e-6



    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    fig.suptitle('1D Euler Solver: Surrogate vs Exact Analytical', fontsize=13, fontweight='bold')

    # Numerical lines (scatter/lines)
    line_rho_num, = ax1.plot([], [], 'b-o', ms=3, lw=1.5, label='Neural Field')
    line_u_num,   = ax2.plot([], [], 'r-o', ms=3, lw=1.5, label='Neural Field')
    line_p_num,   = ax3.plot([], [], 'g-o', ms=3, lw=1.5, label='Neural Field')

    # Exact solution lines (dashed)
    line_rho_exact, = ax1.plot([], [], 'k--', lw=2, label='Exact Analytical')
    line_u_exact,   = ax2.plot([], [], 'k--', lw=2, label='Exact Analytical')
    line_p_exact,   = ax3.plot([], [], 'k--', lw=2, label='Exact Analytical')

    for ax, title in zip([ax1, ax2, ax3], ['Density (kg/m³)', 'Velocity (m/s)', 'Pressure (Pa)']):
        ax.set_ylabel(title)
        ax.set_xlim(0, domain_size)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')

    time_text = ax1.text(0.02, 0.82, '', transform=ax1.transAxes, 
                         fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Auto scale y limits based on max values
    rho_all = np.concatenate(rho_hist)
    u_all   = np.concatenate(u_hist)
    p_all   = np.concatenate(p_hist)


    ax1.set_ylim(min(0, np.min(rho_all)), np.max(rho_all) * 1.15)
    ax2.set_ylim(min(-10, np.min(u_all) * 1.1), max(10, np.max(u_all) * 1.1))
    ax3.set_ylim(min(0, np.min(p_all)), np.max(p_all) * 1.15)
    ax3.set_xlabel('x (m)')

    def init():
        for l in [line_rho_num, line_u_num, line_p_num, line_rho_exact, line_u_exact, line_p_exact]:
            l.set_data([], [])
        time_text.set_text('')
        return line_rho_num, line_u_num, line_p_num, line_rho_exact, line_u_exact, line_p_exact, time_text

    def update(frame):
        t = t_hist[frame]

        # Numerical update
        line_rho_num.set_data(x_num, rho_hist[frame])
        line_u_num.set_data(x_num, u_hist[frame])
        line_p_num.set_data(x_num, p_hist[frame])

        # Exact update
        rho_ex, u_ex, p_ex = exact_solver.sample(x_exact, t)
        line_rho_exact.set_data(x_exact, rho_ex)
        line_u_exact.set_data(x_exact, u_ex)
        line_p_exact.set_data(x_exact, p_ex)
        

        time_text.set_text(f't = {t:.5f} s')
        return line_rho_num, line_u_num, line_p_num, line_rho_exact, line_u_exact, line_p_exact, time_text

    anim = FuncAnimation(
        fig, update, frames=len(t_hist),
        init_func=init, blit=True, interval=interval
    )

    plt.tight_layout()
    plt.show()
    return anim



if __name__ == '__main__':
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    NF = INR(N_in=2, 
             N_hidden_features=64, 
             N_hidden_layers=4, 
             N_out = 3)
    NF.load_state_dict(torch.load(r'./models/neural_field_weights.pth', weights_only=True))

    visualize_model(NF, device = device)

    NF = SIREN(N_in=2, 
             N_hidden_features=64, 
             N_hidden_layers=4, 
             N_out = 3,
             omega_0 = 10,
             first_omega_0 = 10,
             final_linear=True)
    NF.load_state_dict(torch.load(r'./models/SIREN_weights.pth', weights_only=True))

    visualize_model(NF, device = device)

    NF = WIRE(N_in=2, 
             N_hidden_features=64, 
             N_hidden_layers=4, 
             N_out = 3,
             omega_0 = 10,
             first_omega_0 = 10,
             s_0=30,
             first_s_0=30,
             final_linear=True)
    NF.load_state_dict(torch.load(r'./models/WIRE_weights.pth', weights_only=True))

    visualize_model(NF, device = device)