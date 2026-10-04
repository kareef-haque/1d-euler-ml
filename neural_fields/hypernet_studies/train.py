'''
Training Module for the Hypernet-Modulating INR Models
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

from neural_fields.hypernet_studies.models import INR, ModFiLM, ModLoRA
from neural_fields.preprocess import SodPreprocessor, EulerPreprocessor



def train_base(lora_rank = 4,
               lora_alpha = 4.0, 
               config  =r'.\neural_fields\hypernet_studies\configs'):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    warnings.filterwarnings("ignore")
    '''
    EXP2
    '''

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    with open(os.path.join(config, 'model.yaml'), 'r') as file:
        config = yaml.safe_load(file)


        # str(dict).replace(' ', '_').replace(':', '-').replace('"', '').replace

    # model time babyyyyy
    model_time = 1000


    EulerNF = INR(
        N_in=config["model"]["architecture"]["input_features"],
        N_hidden=config["model"]["architecture"]["hidden_features"],
        N_layers=config["model"]["architecture"]["hidden_layers"],
        N_out=config["model"]["architecture"]["output_features"]
    )


    train, test = SodPreprocessor(config = r'.\neural_fields\hypernet_studies\configs\physics.yaml', 
                                  train_test_split = 0.85)

    train_ds = SodDataset(train[0], train[1])
    test_ds = SodDataset(test[0], test[1])

    # Create DataLoaders with multiple workers for speed
    train_load = DataLoader(train_ds, batch_size=config['model']['training']['batch_size'], 
                            pin_memory=True, shuffle=True)
    test_load = DataLoader(test_ds, batch_size=config['model']['training']['batch_size'], 
                           pin_memory=True, shuffle=True)


    loss_fn = nn.functional.mse_loss
    opt_adam  = torch.optim.Adam(EulerNF.parameters(), lr= config['model']['training']['lr'],
                                 fused=True)

    writer = SummaryWriter(log_dir = f'./logs/exp_2/base/')

    model = NField_Train(NF = EulerNF,
                         train_loader = train_load,
                         test_loader = test_load,
                         loss_fn = loss_fn,
                         optimizer = opt_adam,
                         hyperparameters = config,
                         writer = writer, 
                         device = device)
    return model


def train_hyper(neural_field, input_features,
                hyper_type = 'LoRA', 
                config  =r'.\neural_fields\hypernet_studies\configs'):

    '''
    EXP2
    '''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    warnings.filterwarnings("ignore")

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    with open(os.path.join(config, 'model.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    
    neural_field.freeze()

    # model time babyyyyy
    model_time = 1000


    if hyper_type.lower() == 'lora':
        hyper = ModLoRA(
            cond_dim=input_features,
            trunk_hidden=config["hypernetwork"]["trunk_architecture"]["hidden_features"],
            trunk_layers=config['hypernetwork']["trunk_architecture"]["hidden_layers"],
            trunk_out=config["hypernetwork"]["trunk_architecture"]["output_features"],
            inr = neural_field
        )
    else:
        hyper = ModFiLM(
            cond_dim=input_features,
            trunk_hidden=config["hypernetwork"]["trunk_architecture"]["hidden_features"],
            trunk_layers=config['hypernetwork']["trunk_architecture"]["hidden_layers"],
            trunk_out=config["hypernetwork"]["trunk_architecture"]["output_features"],
            inr = neural_field
        )


    # train, test = SodPreprocessor()
    x, y, z = EulerPreprocessor(dataset_type = 'train')
    train_ds = EulerDataset(x, y, z)

    x, y, z = EulerPreprocessor(dataset_type = 'validate')
    test_ds = EulerDataset(x, y, z)

    # Create DataLoaders with multiple workers for speed
    train_load = DataLoader(train_ds, batch_size=config['hypernetwork']['training']['batch_size'], 
                            pin_memory=True, shuffle=True)
    test_load = DataLoader(test_ds, batch_size=config['hypernetwork']['training']['batch_size'], 
                           pin_memory=True, shuffle = True)


    loss_fn = nn.functional.mse_loss
    opt_adam  = torch.optim.Adam(hyper.parameters(), lr= config['hypernetwork']['training']['lr'],
                                 fused = True)

    writer = SummaryWriter(log_dir = f'./logs/exp_2/hyper/')

    hyper = HN_Train(NF = neural_field, hyper=hyper,
                     HN_type = hyper_type,
                     train_loader = train_load,
                     test_loader = test_load,
                     loss_fn = loss_fn,
                     optimizer = opt_adam,
                     hyperparameters = config,
                     writer = writer, 
                     device = device)

    return hyper


def train_EndtoEnd(input_features,
                   hyper_type = 'LoRA', 
                   lora_rank = 4,
                   lora_alpha = 4.0,
                   config  =r'.\neural_fields\hypernet_studies\configs'):

    '''
    EXP3
    '''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    warnings.filterwarnings("ignore")

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    with open(os.path.join(config, 'model.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    neural_field = INR(
        N_in=config["model"]["architecture"]["input_features"]-1,
        N_hidden=config["model"]["architecture"]["hidden_features"],
        N_layers=config["model"]["architecture"]["hidden_layers"],
        N_out=config["model"]["architecture"]["output_features"],
        lora_rank=lora_rank,
        lora_alpha=lora_alpha
    )


    # model time babyyyyy
    model_time = 1000


    if hyper_type.lower() == 'lora':
        hyper = ModLoRA(
            cond_dim=input_features,
            trunk_hidden=config["hypernetwork"]["trunk_architecture"]["hidden_features"],
            trunk_layers=config['hypernetwork']["trunk_architecture"]["hidden_layers"],
            trunk_out=config["hypernetwork"]["trunk_architecture"]["output_features"],
            inr = neural_field
        )
    else:
        hyper = ModFiLM(
            cond_dim=input_features,
            trunk_hidden=config["hypernetwork"]["trunk_architecture"]["hidden_features"],
            trunk_layers=config['hypernetwork']["trunk_architecture"]["hidden_layers"],
            trunk_out=config["hypernetwork"]["trunk_architecture"]["output_features"],
            inr = neural_field
        )


    train, test = SodPreprocessor(config = r'.\neural_fields\hypernet_studies\configs\physics.yaml', 
                                  train_test_split = 0.85)

    train_ds = SodDataset(train[0], train[1])
    test_ds = SodDataset(test[0], test[1])

    # Create DataLoaders with multiple workers for speed
    train_load = DataLoader(train_ds, batch_size=config['model']['training']['batch_size'], 
                            pin_memory=True, shuffle=True)
    test_load = DataLoader(test_ds, batch_size=config['model']['training']['batch_size'], 
                           pin_memory=True, shuffle=True)


    loss_fn = nn.functional.mse_loss
    opt_adam  = torch.optim.Adam([{'params':neural_field.parameters(), 'lr':config['model']['training']['lr']},
                                  {'params':hyper.parameters(), 'lr':config['hypernetwork']['training']['lr']}],
                                  fused=True)

    writer = SummaryWriter(log_dir = f'./logs/exp_3/')

    base, hyper = HN_NField_Train(NF = neural_field, hyper = hyper, 
                                  HN_type = hyper_type,
                                  train_loader = train_load,
                                  test_loader = test_load,
                                  loss_fn = loss_fn,
                                  optimizer = opt_adam,
                                  hyperparameters = config,
                                  writer = writer, 
                                  device = device)
  
    return base, hyper

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
        return self.coords[idx], self.flow[idx], self.coords[:, 1][idx]

class EulerDataset(Dataset):
    '''
    Pytorch Training Dataset
    - Sod Shock Problem with Varying initial condtions (Pres ratio, Dens Ratio, Diaphram Location)
    - data passed to this class MUST already be pre-processed
        - array flattening, test/testidation split, etc.
    '''
    def __init__(self, coords, flow, IC):
        self.coords = coords # (N_time*N_cell, 2) | (x, t)
        self.flow = flow # (N_time*N_cell, 3) | (dens, vel, pres)
        self.IC = IC # (N_time*N_cell, 3) | (P2_P1, rho2_rho1, x0)

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        return self.coords[idx], self.flow[idx], self.IC[idx]

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def NField_Train(NF, train_loader, test_loader,
                 loss_fn, optimizer, 
                 hyperparameters : dict, 
                 writer, 
                 device = 'cpu'):


    epochs = hyperparameters['model']['training']['epochs']
    batch_size = hyperparameters['model']['training']['batch_size']
    lr = hyperparameters['model']['training']['lr']

    # save_path = r'.neural_fields/hypernet_studies/models'
    # os.makedirs(save_path, exist_ok=True)
    

    init_epoch = 0

    for epoch in range(init_epoch, epochs):
        print(f"-------------------------------\nEpoch {epoch+1}\n-------------------------------")

        NFtrain_epoch(dataloader = train_loader,
                    model = NF, 
                    loss_fn = loss_fn, 
                    optimizer = optimizer,
                    writer = writer, ep_curr = epoch, 
                    device = device)

        NFtest_epoch(dataloader = test_loader,
                    model = NF, 
                    loss_fn = loss_fn, 
                    writer = writer, ep_curr = epoch, 
                    device = device)

        writer.flush()

    # torch.save(NF.state_dict(), os.path.join(save_path, f'{model_name}_weights.pth'))
    print("Base Net Training Complete!")
    writer.close()

    return NF

# ----------------------------------------------------------------------------------------------------

def HN_Train(NF, hyper, HN_type : str,
             train_loader, test_loader,
             loss_fn, optimizer, 
             hyperparameters : dict, 
             writer, 
             device = 'cpu'):


    epochs = hyperparameters['hypernetwork']['training']['epochs']
    batch_size = hyperparameters['hypernetwork']['training']['batch_size']
    lr = hyperparameters['hypernetwork']['training']['lr']

    # save_path = r'.neural_fields/hypernet_studies/models'
    # os.makedirs(save_path, exist_ok=True)
    

    init_epoch = 0

    for epoch in range(init_epoch, epochs):
        print(f"-------------------------------\nEpoch {epoch+1}\n-------------------------------")

        HNtrain_epoch(dataloader = train_loader,
                      model = NF, hyper = hyper, hyper_type = HN_type, 
                      loss_fn = loss_fn, 
                      optimizer = optimizer,
                      writer = writer, ep_curr = epoch, 
                      device = device)

        HNtest_epoch(dataloader = test_loader,
                     model = NF, hyper = hyper, hyper_type = HN_type, 
                     loss_fn = loss_fn, 
                     writer = writer, ep_curr = epoch, 
                     device = device)

        writer.flush()

    # torch.save(hyper.state_dict(), os.path.join(save_path, f'{model_name}_weights.pth'))
    print(f"HyperNet [{HN_type}] Training Complete!")
    writer.close()
    return hyper

# ----------------------------------------------------------------------------------------------------

def HN_NField_Train(NF, hyper, HN_type : str,
                    train_loader, test_loader,
                    loss_fn, optimizer, 
                    hyperparameters : dict, 
                    writer, 
                    device = 'cpu'):

    epochs = hyperparameters['model']['training']['epochs']
    batch_size = hyperparameters['model']['training']['batch_size']
    lr = hyperparameters['model']['training']['lr']

    # save_path = r'.neural_fields/hypernet_studies/models'
    # os.makedirs(save_path, exist_ok=True)
    

    init_epoch = 0

    for epoch in range(init_epoch, epochs):
        print(f"-------------------------------\nEpoch {epoch+1}\n-------------------------------")

        HN_NFtrain_epoch(dataloader = train_loader,
                         model = NF, hyper = hyper, hyper_type = HN_type, 
                         loss_fn = loss_fn, 
                         optimizer = optimizer,
                         writer = writer, ep_curr = epoch, 
                         device = device)

        HN_NFtest_epoch(dataloader = test_loader,
                        model = NF, hyper = hyper, hyper_type = HN_type, 
                        loss_fn = loss_fn, 
                        writer = writer, ep_curr = epoch, 
                        device = device)

        writer.flush()

    # torch.save(NF.state_dict(), os.path.join(save_path, f'{model_name}_weights.pth'))
    # torch.save(hyper.state_dict(), os.path.join(save_path, f'{model_name}_weights.pth'))

    print("Training Complete!")
    writer.close()
    return NF, hyper


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def NFtrain_epoch(dataloader, model, 
                loss_fn, optimizer,
                writer, ep_curr,
                device = 'cpu'):

    model = model.to(device)
    model.train()

    loss = 0.0
    total_loss = torch.zeros((), device=device)
    n_samples = 0
    
    for batch, (X, y, z) in enumerate(dataloader):
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


        total_loss += loss.detach() * R.shape[0]
        n_samples += R.shape[0]

    epoch_loss = (total_loss / n_samples).item()
    print(f"loss: {epoch_loss:>7f}")
    writer.add_scalar("Loss/train", epoch_loss, ep_curr)

def NFtest_epoch(dataloader, model, 
                loss_fn,
                writer, ep_curr,
                device = 'cpu'):
    
    model = model.to(device)
    model.eval()

    loss = 0.0
    total_loss = torch.zeros((), device=device)
    n_samples = 0

    with torch.no_grad():
        for X, y, z in dataloader:
            coord = X.to(device, non_blocking=True)
            R = y.to(device, non_blocking=True)

            # forward pass
            R_pred, __ = model(coord)
            loss = loss_fn(R_pred, R)
            total_loss += loss * R.shape[0]
            n_samples += R.shape[0]

    epoch_loss = (total_loss / n_samples).item()
    print('VALIDATION:')
    print(f"loss: {epoch_loss:>7f}")
    writer.add_scalar("Loss/validate", epoch_loss, ep_curr)


# ----------------------------------------------------------------------------------------------------


def HNtrain_epoch(dataloader, 
                  model, hyper, hyper_type,
                  loss_fn, optimizer,
                  writer, ep_curr,
                  device = 'cpu'):

    model = model.to(device)
    hyper = hyper.to(device)
    model.eval()
    hyper.train()

    loss = 0.0
    total_loss = torch.zeros((), device=device)
    n_samples = 0

    for batch, (X, y, z) in enumerate(dataloader):
        # X = (x, t)
        # y = (rho, u, P)

        coord = X.to(device, non_blocking=True)
        R = y.to(device, non_blocking=True)
        cond = z.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        #apply hypernetwork if needed
        if hyper_type.lower() == 'film':
            # apply hypernetwork
            film = hyper(cond)
            R_pred, __ = model(coord, film=film)
        elif hyper_type.lower() == 'lora':
            # apply hypernetwork
            lora = hyper(cond)
            R_pred, __ = model(coord, lora=lora)
        else:
            raise ValueError(f"Hypernetwork type {hyper_type} not supported")

        loss = loss_fn(R_pred, R)

        # propogate gradients
        loss.backward()
        optimizer.step()

        total_loss += loss.detach() * R.shape[0]
        n_samples += R.shape[0]

    epoch_loss = (total_loss / n_samples).item()
    print(f"loss: {epoch_loss:>7f}")
    writer.add_scalar("Loss/train", epoch_loss, ep_curr)

def HNtest_epoch(dataloader, 
                 model, hyper, hyper_type,
                 loss_fn,
                 writer, ep_curr,
                 device = 'cpu'):
    
    model = model.to(device)
    hyper = hyper.to(device)
    model.eval()
    hyper.eval()

    loss = 0.0
    total_loss = torch.zeros((), device=device)
    n_samples = 0

    with torch.no_grad():
        for X, y, z in dataloader:
            coord = X.to(device, non_blocking=True)
            R = y.to(device, non_blocking=True)
            cond = z.to(device, non_blocking=True)

                    #apply hypernetwork if needed
            if hyper_type.lower() == 'film':
                # apply hypernetwork
                film = hyper(cond)
                R_pred, __ = model(coord, film=film)
            elif hyper_type.lower() == 'lora':
                # apply hypernetwork
                lora = hyper(cond)
                R_pred, __ = model(coord, lora=lora)
            else:
                raise ValueError(f"Hypernetwork type {hyper_type} not supported")

            loss = loss_fn(R_pred, R)
            total_loss += loss * R.shape[0]
            n_samples += R.shape[0]

    epoch_loss = (total_loss / n_samples).item()
    print('VALIDATION:')
    print(f"loss: {epoch_loss:>7f}")
    writer.add_scalar("Loss/validate", epoch_loss, ep_curr)


# ----------------------------------------------------------------------------------------------------


def HN_NFtrain_epoch(dataloader, 
                  model, hyper, hyper_type,
                  loss_fn, optimizer,
                  writer, ep_curr,
                  device = 'cpu'):

    model = model.to(device)
    hyper = hyper.to(device)
    model.train()
    hyper.train()

    loss = 0.0
    total_loss = torch.zeros((), device=device)
    n_samples = 0

    for batch, (X, y, z) in enumerate(dataloader):
        # X = (x, t)
        # y = (rho, u, P)
        # z = (t)

        coord = X[:, :1].to(device, non_blocking=True)
        R = y.to(device, non_blocking=True)
        cond = z.unsqueeze(-1).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        #apply hypernetwork if needed
        if hyper_type.lower() == 'film':
            # apply hypernetwork
            film = hyper(cond)
            R_pred, __ = model(coord, film=film)
        elif hyper_type.lower() == 'lora':
            # apply hypernetwork
            lora = hyper(cond)
            R_pred, __ = model(coord, lora=lora)
        else:
            raise ValueError(f"Hypernetwork type {hyper_type} not supported")

        loss = loss_fn(R_pred, R)

        # propogate gradients
        loss.backward()
        optimizer.step()

        total_loss += loss.detach() * R.shape[0]
        n_samples += R.shape[0]

    epoch_loss = (total_loss / n_samples).item()
    print(f"loss: {epoch_loss:>7f}")
    writer.add_scalar("Loss/train", epoch_loss, ep_curr)

def HN_NFtest_epoch(dataloader, 
                 model, hyper, hyper_type,
                 loss_fn,
                 writer, ep_curr,
                 device = 'cpu'):
    
    model = model.to(device)
    hyper = hyper.to(device)
    model.eval()
    hyper.eval()

    loss = 0.0
    total_loss = torch.zeros((), device=device)
    n_samples = 0

    with torch.no_grad():
        for X, y, z in dataloader:
            coord = X[:, :1].to(device, non_blocking=True)
            R = y.to(device, non_blocking=True)
            cond = z.unsqueeze(-1).to(device, non_blocking=True)

                    #apply hypernetwork if needed
            if hyper_type.lower() == 'film':
                # apply hypernetwork
                film = hyper(cond)
                R_pred, __ = model(coord, film=film)
            elif hyper_type.lower() == 'lora':
                # apply hypernetwork
                lora = hyper(cond)
                R_pred, __ = model(coord, lora=lora)
            else:
                raise ValueError(f"Hypernetwork type {hyper_type} not supported")


            loss = loss_fn(R_pred, R)
            total_loss += loss.detach() * R.shape[0]
            n_samples += R.shape[0]

    epoch_loss = (total_loss / n_samples).item()
    print('VALIDATION:')
    print(f"loss: {epoch_loss:>7f}")
    writer.add_scalar("Loss/validate", epoch_loss, ep_curr)


