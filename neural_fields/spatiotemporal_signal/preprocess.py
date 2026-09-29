import yaml
import numpy as np
rng = np.random.default_rng(seed=6767)



def normalize(R_flat, config = r'.\neural_fields\spatiotemporal_signal\configs\physics.yaml'):
    '''
    param nd.array R_flat: (N_time * N_cells * N_sims, 3) Flattened array of all primative R states
    param nd.array R_norm: (N_time * N_cells * N_sims, 3) Flattened array of all normalized primative R states

    will modify provided config with normalization parameters
    '''

    # z-standardize the data
    mean_train = np.mean(R_flat, axis=0, keepdims=True)
    std_train = np.std(R_flat, axis=0, keepdims=True)
    R_norm = (R_flat - mean_train) / std_train

    with open(config, 'r') as f:
        param = yaml.safe_load(f)



    param['normalize'] = {'mean': mean_train[0].tolist(), 
                          'std': std_train[0].tolist()}

    with open(config, 'w') as f:
        yaml.safe_dump(param, f, sort_keys=False)

    return R_norm



def SodPreprocessor(config = r'.\neural_fields\spatiotemporal_signal\configs\physics.yaml', 
                   Sod_Path = './generated_data/train/train_-1.npy'):

    Q_hist = np.load(file=Sod_Path,
                     mmap_mode = 'r') # (N_time, 3, N_cell)

    with open(config, 'r') as f:
        param = yaml.safe_load(f)
        gamma = param['flow_parameters']['gamma'] 

    N_time, __, N_cell = Q_hist.shape

    # convert to primative testues
    rho = Q_hist[:, 0, :]
    mom = Q_hist[:, 1, :]
    E = Q_hist[:, 2, :]

    u = mom/rho
    P = (gamma - 1.0) * (E - 0.5 * rho * u**2)

    R_hist = np.stack([rho, u, P], axis =1)

    # flatten the data
    R_flat = R_hist.transpose(0, 2, 1).reshape(-1, 3)  # (N_time*N_cell, 3)

    #split into sets
    mask = np.full(R_flat.shape[0], False)
    mask[:int(R_flat.shape[0]*0.8)] = True
    np.random.shuffle(mask)
    

    # z-standardize the data
    R_norm = normalize(R_flat)


    # split into train and test sets
    train_flow = R_norm[mask]
    test_flow = R_norm[~mask]


    x = np.linspace(-1, 1, N_cell)
    t = np.linspace(0, 1, N_time) 

    # Create a mesh grid and flatten
    x_mesh, t_mesh = np.meshgrid(x, t, indexing='ij')  # both shape: (N_cell, N_time)
    x_flat = x_mesh.T.flatten()  # shape: (N_time * N_cell,)
    t_flat = t_mesh.T.flatten()  # shape: (N_time * N_cell,)
    coords = np.stack([x_flat, t_flat], axis=1)  # (N_time*N_cell, 2)

    train_coords = coords[mask]
    test_coords = coords[~mask]

    return (train_coords, train_flow), (test_coords, test_flow)


