import yaml
import os
import glob
import numpy as np
rng = np.random.default_rng(seed=6767)



def z_normalize(R_flat, config = r'.\neural_fields\capability_test\configs\physics.yaml'):
    '''
    Z-Score normalization
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

def phys_normalize(R_flat, 
                   config = r'.\neural_fields\hypernet_studies\configs\physics.yaml'):
    '''
    Physical Scale Normalization
        - Normalization Parameters are
            rho = 1.0
            u = sqrt(gamma * P/rho)
            P = 100,000
    
    param nd.array R_flat: (N_time * N_cells * N_sims, 3) Flattened array of all primative R states
    
    return nd.array R_norm: (N_time * N_cells * N_sims, 3) Flattened array of all normalized primative R states

    will modify provided config with normalization parameters
    '''

    with open(config, 'r') as f:
        param = yaml.safe_load(f)
        gamma = param['flow_parameters']['gamma'] 

    # physially normalize the data
    rho = 1.0
    P = 100000
    u = np.sqrt(gamma * P/rho)

    R_norm = R_flat / np.array([rho, u, P])

    param['normalize'] = {'rho': float(rho), 'u': float(u), 'P': float(P)}

    with open(config, 'w') as f:
        yaml.safe_dump(param, f, sort_keys=False)

    return R_norm


def SodPreprocessor(config = r'.\neural_fields\capability_test\configs\physics.yaml', 
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

    # z-standardize the data
    R_norm = phys_normalize(R_flat)


    x = np.linspace(-1, 1, N_cell)
    t = np.linspace(0, 1, N_time) 

    # Create a mesh grid and flatten
    x_mesh, t_mesh = np.meshgrid(x, t, indexing='ij')  # both shape: (N_cell, N_time)
    x_flat = x_mesh.T.flatten()  # shape: (N_time * N_cell,)
    t_flat = t_mesh.T.flatten()  # shape: (N_time * N_cell,)
    coords = np.stack([x_flat, t_flat], axis=1)  # (N_time*N_cell, 2)

# Time-based block splits: 4 interior 8% blocks centered at tau = 0.2, 0.4, 0.6, 0.8
    val_block_1 = (t_flat >= 0.16) & (t_flat < 0.24)  # Val centered at 0.20
    test_block_1 = (t_flat >= 0.36) & (t_flat < 0.44) # Test centered at 0.40
    val_block_2 = (t_flat >= 0.56) & (t_flat < 0.64)  # Val centered at 0.60
    test_block_2 = (t_flat >= 0.76) & (t_flat < 0.84) # Test centered at 0.80

    val_mask = val_block_1 | val_block_2
    test_mask = test_block_1 | test_block_2
    train_mask = ~(val_mask | test_mask)

    # split into train, val, and test sets
    train_flow = R_norm[train_mask]
    val_flow = R_norm[val_mask]
    test_flow = R_norm[test_mask]

    train_coords = coords[train_mask]
    val_coords = coords[val_mask]
    test_coords = coords[test_mask]

    return (train_coords, train_flow), (val_coords, val_flow), (test_coords, test_flow)


def EulerPreprocessor(dataset_type = 'train',
                      config = r'.\neural_fields\hypernet_studies\configs\physics.yaml', 
                      data_path = './generated_data/',
                      time_stride = 10,
                      baseline = False):


    with open(config, 'r') as f:
        param = yaml.safe_load(f)
        gamma = param['flow_parameters']['gamma'] 

    #collects all data from each simulation first, then will concatenate after
        # normalzation occurs for all the data (so after all concatenated)
    R_list = []
    coords_list = []
    IC_list = []


    folder_path = os.path.join(data_path, dataset_type)

    # 1. Match ONLY the primary files (e.g., train_0.npy, train_-1.npy)
    # using a regex/filter to avoid matching train_init_*.npy
    primary_files = glob.glob(os.path.join(folder_path, f"{dataset_type}_[!-]*"))
    primary_files += glob.glob(
        os.path.join(folder_path, f"{dataset_type}_-*.npy")
    )  # Include negative indices

    for file_path in primary_files:
        # Skip init files if matched
        if "_init_" in file_path:
            continue

        # Extract index string (e.g., '-1', '0', '15')
        filename = os.path.basename(file_path)
        idx_str = filename.replace(f"{dataset_type}_", "").replace(".npy", "")

        # Define paths for both files
        main_file = file_path
        init_file = os.path.join(folder_path, f"{dataset_type}_init_{idx_str}.npy")

        # Load both files
        Q_hist = np.load(file=main_file, mmap_mode="r")
        IC_assoc = np.load(file=init_file, mmap_mode="r")

        # Temporal Subsampling: slice every `time_stride` snapshots directly on disk/memory
        Q_hist_sub = Q_hist[::time_stride, :, :]
        N_time_sub, __, N_cell = Q_hist_sub.shape

        # N_time, __, N_cell = Q_hist.shape

        # create the the data array
        # convert to primative testues
        rho = Q_hist_sub[:, 0, :]
        mom = Q_hist_sub[:, 1, :]
        E = Q_hist_sub[:, 2, :]

        u = mom/rho
        P = (gamma - 1.0) * (E - 0.5 * rho * u**2)

        R_hist = np.stack([rho, u, P], axis =1)

        # flatten the data
        R_flat = R_hist.transpose(0, 2, 1).reshape(-1, 3)  # (N_time*N_cell, 3)

        # create the coords array
        x = np.linspace(-1, 1, N_cell)
        t = np.linspace(0, 1, Q_hist.shape[0])[::time_stride] # ensure time values match original grid points

        # Create a mesh grid and flatten
        x_mesh, t_mesh = np.meshgrid(x, t, indexing='ij')  # both shape: (N_cell, N_time)
        x_flat = x_mesh.T.flatten()  # shape: (N_time * N_cell,)
        t_flat = t_mesh.T.flatten()  # shape: (N_time * N_cell,)
        coords = np.stack([x_flat, t_flat], axis=1)  # (N_time*N_cell, 2)

        # Create the IC Array
        QL = IC_assoc[0] #(rho, mom, E)
        QR = IC_assoc[1] #(rho, mom, E)
        x_split_percent = IC_assoc[2][0] #(x_split%, x_split%, x_split%)

        IC = np.array([QR[0]/QL[0], 
                       ((gamma - 1.0) * (QR[2] - 0.5 * QR[0] * (QR[1]/QR[0])**2))/((gamma - 1.0) * (QL[2] - 0.5 * QL[0] * (QL[1]/QL[0])**2)),
                       x_split_percent])
        # IC_block = np.copy(IC)
        IC_flat = np.tile(IC, (N_time_sub*N_cell, 1))
        

        R_list.append(R_flat)
        coords_list.append(coords)
        IC_list.append(IC_flat)
        

    R_dataset = np.vstack(R_list) # (N_time*N_cell*N_dataset, 3)
    coords_dataset = np.vstack(coords_list) # (N_time*N_cell*N_dataset, 2)
    IC_dataset = np.vstack(IC_list) # (N_time*N_cell*N_dataset, 3)


    # physically standardize the data
    R_norm = phys_normalize(R_dataset)


    if baseline == True:
        return (np.hstack((coords_dataset, IC_dataset)), R_norm, IC_dataset)
    else:
        return (coords_dataset, R_norm, IC_dataset)



