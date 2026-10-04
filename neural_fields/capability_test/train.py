'''
Training Module for the Spatiotemporal INR Models
    - For use with the base INR, SIREN, & WIRE
    - Implements the training loops and dataset loading
    
'''
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
rng = np.random.default_rng(seed=6767)
import os
import yaml
import warnings

from neural_fields.capability_test.models import INR, SIREN, WIRE
from neural_fields.preprocess import SodPreprocessor



def train_model(NF_def, model_type = 'neural_field', config  =r'.\neural_fields\capability_test\configs'):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    warnings.filterwarnings("ignore")

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)


        # str(dict).replace(' ', '_').replace(':', '-').replace('"', '').replace

    # model time babyyyyy
    model_time = 1000


    if model_type.lower() == 'siren':
        EulerNF = NF_def(
            N_in=config["model"]["architecture"]["input_features"],
            N_hidden_features=config["model"]["architecture"]["hidden_features"],
            N_hidden_layers=config["model"]["architecture"]["hidden_layers"],
            N_out=config["model"]["architecture"]["output_features"],
            omega_0=config["model"]["siren"]["w0"],
            first_omega_0=config["model"]["siren"]["first_w0"],
            final_linear=True
        )
        model_name = model_type + f'wn{config["model"]["siren"]["w0"]}-w1{config["model"]["siren"]["first_w0"]}'

    elif model_type.lower() == 'wire':
        EulerNF = NF_def(
            N_in=config["model"]["architecture"]["input_features"],
            N_hidden_features=config["model"]["architecture"]["hidden_features"],
            N_hidden_layers=config["model"]["architecture"]["hidden_layers"],
            N_out=config["model"]["architecture"]["output_features"],
            omega_0=config["model"]["wire"]["w0"],
            first_omega_0=config["model"]["wire"]["first_w0"],
            s_0=config["model"]["wire"]["s0"],
            first_s_0=config["model"]["wire"]["first_s0"],
            final_linear=True
        )
        model_name = model_type + f'wn{config["model"]["wire"]["w0"]}-sn{config["model"]["wire"]["s0"]}__w1{config["model"]["siren"]["first_w0"]}-s0{config["model"]["siren"]["first_s0"]}'

    else:
        EulerNF = NF_def(
            N_in=config["model"]["architecture"]["input_features"],
            N_hidden_features=config["model"]["architecture"]["hidden_features"],
            N_hidden_layers=config["model"]["architecture"]["hidden_layers"],
            N_out=config["model"]["architecture"]["output_features"]
        )


    train, test = SodPreprocessor()

    train_ds = SodDataset(train[0], train[1])
    test_ds = SodDataset(test[0], test[1])

    # Create DataLoaders with multiple workers for speed
    train_load = DataLoader(train_ds, batch_size=config['training']['batch_size'], pin_memory=True)
    test_load = DataLoader(test_ds, batch_size=config['training']['batch_size'], pin_memory=True)



    loss_fn = nn.functional.mse_loss
    opt_adam  = torch.optim.Adam(EulerNF.parameters(), lr= config['training']['lr'])

    writer = SummaryWriter(log_dir = f'./logs/exp_1/{model_name}')

    NField_Train(NF = EulerNF,
                 train_loader = train_load,
                 test_loader = test_load,
                 loss_fn = loss_fn,
                 optimizer = opt_adam,
                 training_config = config,
                 writer = writer, 
                 model_name = model_name, 
                 device = device)


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


class SodDataset(Dataset):
    '''
    Pytorch Training Dataset
    - exclusively the cannocial sod shock tube problem
    - data passed to this class MUST already be pre-processed
        - array flattening, test/testidation split, etc.
    '''
    def __init__(self, coords, flow):
        self.coords = coords # (N_time*N_cell, 2) | (x, t)
        self.flow = flow # (N_time*N_cell, 3) | (dens, vel, pres)

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        return self.coords[idx], self.flow[idx]

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def NField_Train(NF, train_loader, test_loader,
                 loss_fn, optimizer, 
                 hyperparameters : dict, 
                 writer, model_name = 'neural_field', 
                 device = 'cpu'):

    epochs = hyperparameters['training']['epochs']
    batch_size = hyperparameters['training']['batch_size']
    lr = hyperparameters['training']['lr']

    os.makedirs('models', exist_ok=True)
    save_path = './models'
    




    init_epoch = 0

    for epoch in range(init_epoch, epochs):
        print(f"-------------------------------\nEpoch {epoch+1}\n-------------------------------")

        train_epoch(dataloader = train_loader,
                    model = NF, 
                    loss_fn = loss_fn, 
                    optimizer = optimizer,
                    writer = writer, ep_curr = epoch, 
                    device = device)

        test_epoch(dataloader = test_loader,
                    model = NF, 
                    loss_fn = loss_fn, 
                    writer = writer, ep_curr = epoch, 
                    device = device)

        writer.flush()

    torch.save(NF.state_dict(), os.path.join(save_path, f'{model_name}_weights.pth'))
    print("Training Complete!")
    writer.close()


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def train_epoch(dataloader, model, 
                loss_fn, optimizer,
                writer, ep_curr,
                device = 'cpu'):

    model = model.to(device)
    model.train()

    loss = 0.0

    for batch, (X, y) in enumerate(dataloader):
        # X = (x, t)
        # y = (rho, u, P)

        coord = X.to(device, non_blocking=True)
        R = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # forward pass
        R_pred, __ = model(coord)
        loss = loss_fn(R_pred, R)

        # propogate gradients
        loss.backward()
        optimizer.step()

    print(f"loss: {loss:>7f}")
    writer.add_scalar("Loss/train", loss, ep_curr)

def test_epoch(dataloader, model, 
                loss_fn,
                writer, ep_curr,
                device = 'cpu'):
    
    model = model.to(device)
    model.eval()

    loss = 0.0
    for X, y in dataloader:
        coord = X.to(device, non_blocking=True)
        R = y.to(device, non_blocking=True)

        # forward pass
        R_pred, __ = model(coord)
        loss = loss_fn(R_pred, R)


    print('VALIDATION:')
    print(f"loss: {loss:>7f}")
    writer.add_scalar("Loss/testidate", loss, ep_curr)


