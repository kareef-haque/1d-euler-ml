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
import time

from neural_fields.hypernet_studies.models import INR, ModFiLM, ModLoRA
from neural_fields.preprocess import SodPreprocessor, EulerPreprocessor

grad_clip = 10

def train_base(lora_rank = 4,
               lora_alpha = 4.0, 
               run_name = f"run_{int(time.time())}",
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


    train, validate, test = SodPreprocessor(config = r'.\neural_fields\hypernet_studies\configs\physics.yaml')

    train_ds = SodDataset(train[0], train[1])
    validate_ds = SodDataset(validate[0], validate[1])

    # Create DataLoaders with multiple workers for speed
    train_load = DataLoader(train_ds, batch_size=config['model']['training']['batch_size'], 
                            pin_memory=True, shuffle=True)
    validate_load = DataLoader(validate_ds, batch_size=config['model']['training']['batch_size'], 
                           pin_memory=True, shuffle=False)





    #training specific
    loss_fn = nn.functional.mse_loss

    # appliying weight decays and LR scheduler
    optim_groups = apply_decay_mask(EulerNF, 
                                    lr = config['model']['training']['lr'],
                                    weight_decay = config['optimization']['weight_decay'],
                                    )

    opt_adam  = torch.optim.AdamW(optim_groups,
                                 fused=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_adam,
        T_max = len(train_load)*config['model']['training']['epochs'], #batchs per epoch * epochs
        eta_min = config['optimization']['lr_min']
    )



    writer = SummaryWriter(log_dir = f'./logs/exp_2/base/{run_name}')

    model = NField_Train(NF = EulerNF,
                         train_loader = train_load,
                         validate_loader = validate_load,
                         loss_fn = loss_fn,
                         optimizer = opt_adam,
                         scheduler = scheduler, 
                         hyperparameters = config,
                         writer = writer, 
                         device = device)
    return model


def train_hyper(neural_field, input_features,
                hyper_type = 'LoRA', 
                run_name = f"run_{int(time.time())}",
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
    train_load = DataLoader(train_ds, batch_size=config['hypernetwork']['training']['batch_size'], 
                        pin_memory=True, shuffle=True)

    x, y, z = EulerPreprocessor(dataset_type = 'validate')
    validate_ds = EulerDataset(x, y, z)

    # Create DataLoaders with multiple workers for speed
    train_load = DataLoader(train_ds, batch_size=config['hypernetwork']['training']['batch_size'], 
                            pin_memory=True, shuffle=True)
    validate_load = DataLoader(validate_ds, batch_size=config['hypernetwork']['training']['batch_size'], 
                           pin_memory=True, shuffle = True)


    loss_fn = nn.functional.mse_loss

    # appliying weight decays and LR scheduler
    optim_groups = apply_decay_mask(hyper, 
                                    lr = config['hypernetwork']['training']['lr'],
                                    weight_decay = config['optimization']['weight_decay'],
                                    )

    opt_adam  = torch.optim.AdamW(optim_groups,
                                 fused=True)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_adam,
        T_max = len(train_load)*config['hypernetwork']['training']['epochs'], #batchs per epoch * epochs
        eta_min = config['optimization']['lr_min']
    )

    writer = SummaryWriter(log_dir = f'./logs/exp_2/hyper/{run_name}')

    hyper = HN_Train(NF = neural_field, hyper=hyper,
                     HN_type = hyper_type,
                     train_loader = train_load,
                     validate_loader = validate_load,
                     loss_fn = loss_fn,
                     optimizer = opt_adam,
                     scheduler = scheduler,
                     hyperparameters = config,
                     writer = writer, 
                     device = device)

    return hyper


def train_EndtoEnd(input_features,
                   hyper_type = 'LoRA', 
                   lora_rank = 4,
                   lora_alpha = 4.0,
                   run_name = f"run_{int(time.time())}",
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


    train, validate, test = SodPreprocessor(config = r'.\neural_fields\hypernet_studies\configs\physics.yaml')

    train_ds = SodDataset(train[0], train[1])
    validate_ds = SodDataset(validate[0], validate[1])

    # Create DataLoaders with multiple workers for speed
    train_load = DataLoader(train_ds, batch_size=config['model']['training']['batch_size'], 
                            pin_memory=True, shuffle=True)
    validate_load = DataLoader(validate_ds, batch_size=config['model']['training']['batch_size'], 
                           pin_memory=True, shuffle=True)


    loss_fn = nn.functional.mse_loss

    # appliying weight decays and LR scheduler

    nf_optim_groups = apply_decay_mask(neural_field, 
                                lr = config['model']['training']['lr'],
                                weight_decay = config['optimization']['weight_decay'],
                                )   

    hy_optim_groups = apply_decay_mask(hyper, 
                                       lr = config['hypernetwork']['training']['lr'],
                                       weight_decay = config['optimization']['weight_decay'],
                                      )

    opt_adam  = torch.optim.AdamW(nf_optim_groups + hy_optim_groups,
                                 fused=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_adam,
        T_max = len(train_load)*config['model']['training']['epochs'], #batchs per epoch * epochs
        eta_min = config['optimization']['lr_min']
    )


    writer = SummaryWriter(log_dir = f'./logs/exp_3/{run_name}')

    base, hyper = HN_NField_Train(NF = neural_field, hyper = hyper, 
                                  HN_type = hyper_type,
                                  train_loader = train_load,
                                  validate_loader = validate_load,
                                  loss_fn = loss_fn,
                                  optimizer = opt_adam,
                                  scheduler = scheduler,
                                  hyperparameters = config,
                                  writer = writer, 
                                  device = device)
  
    return base, hyper


def train_EXP2_baseline(lora_rank = 4,
                        lora_alpha = 4.0, 
                        run_name = f"run_{int(time.time())}",
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
        N_in=config["model"]["architecture"]["input_features"] + 3,
        N_hidden=config["model"]["architecture"]["hidden_features"],
        N_layers=config["model"]["architecture"]["hidden_layers"],
        N_out=config["model"]["architecture"]["output_features"]
    )




    # train, test = SodPreprocessor()
    x, y, z = EulerPreprocessor(dataset_type = 'train', baseline = True)
    train_ds = EulerDataset(x, y, z)
    train_load = DataLoader(train_ds, batch_size=config['hypernetwork']['training']['batch_size'], 
                        pin_memory=True, shuffle=True)

    x, y, z = EulerPreprocessor(dataset_type = 'validate', baseline = True)
    validate_ds = EulerDataset(x, y, z)

    # Create DataLoaders with multiple workers for speed
    train_load = DataLoader(train_ds, batch_size=config['hypernetwork']['training']['batch_size'], 
                            pin_memory=True, shuffle=True)
    validate_load = DataLoader(validate_ds, batch_size=config['hypernetwork']['training']['batch_size'], 
                           pin_memory=True, shuffle = True)

    

    loss_fn = nn.functional.mse_loss

    # appliying weight decays and LR scheduler
    optim_groups = apply_decay_mask(EulerNF, 
                                    lr = config['model']['training']['lr'],
                                    weight_decay = config['optimization']['weight_decay'],
                                    )

    opt_adam  = torch.optim.AdamW(optim_groups,
                                 fused=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_adam,
        T_max = len(train_load)*config['model']['training']['epochs'], #batchs per epoch * epochs
        eta_min = config['optimization']['lr_min']
    )


    writer = SummaryWriter(log_dir = f'./logs/baseline/{run_name}')

    model = NField_Train(NF = EulerNF,
                         train_loader = train_load,
                         validate_loader = validate_load,
                         loss_fn = loss_fn,
                         optimizer = opt_adam,
                         scheduler = scheduler,
                         hyperparameters = config,
                         writer = writer, 
                         device = device)
    return model


def train_EXP3_baseline(lora_rank = 4,
                        lora_alpha = 4.0, 
                        run_name = f"run_{int(time.time())}",
                        config  =r'.\neural_fields\hypernet_studies\configs'):
            
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    warnings.filterwarnings("ignore")
    '''
    EXP3
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




    train, validate, test = SodPreprocessor(config = r'.\neural_fields\hypernet_studies\configs\physics.yaml')

    train_ds = SodDataset(train[0], train[1])
    validate_ds = SodDataset(validate[0], validate[1])

    # Create DataLoaders with multiple workers for speed
    train_load = DataLoader(train_ds, batch_size=config['model']['training']['batch_size'], 
                            pin_memory=True, shuffle=True)
    validate_load = DataLoader(validate_ds, batch_size=config['model']['training']['batch_size'], 
                           pin_memory=True, shuffle=True)


    # Create DataLoaders with multiple workers for speed
    train_load = DataLoader(train_ds, batch_size=config['hypernetwork']['training']['batch_size'], 
                            pin_memory=True, shuffle=True)
    validate_load = DataLoader(validate_ds, batch_size=config['hypernetwork']['training']['batch_size'], 
                           pin_memory=True, shuffle = True)

    

    loss_fn = nn.functional.mse_loss

    # appliying weight decays and LR scheduler
    optim_groups = apply_decay_mask(EulerNF, 
                                    lr = config['model']['training']['lr'],
                                    weight_decay = config['optimization']['weight_decay'],
                                    )

    opt_adam  = torch.optim.AdamW(optim_groups,
                                 fused=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_adam,
        T_max = len(train_load)*config['model']['training']['epochs'], #batchs per epoch * epochs
        eta_min = config['optimization']['lr_min']
    )

    writer = SummaryWriter(log_dir = f'./logs/baseline/{run_name}')

    model = NField_Train(NF = EulerNF,
                         train_loader = train_load,
                         validate_loader = validate_load,
                         loss_fn = loss_fn,
                         optimizer = opt_adam,
                         scheduler = scheduler,
                         hyperparameters = config,
                         writer = writer, 
                         device = device)
    return model




# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def NField_Train(NF, train_loader, validate_loader,
                 loss_fn, optimizer, 
                 scheduler,
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
                    scheduler = scheduler,
                    writer = writer, ep_curr = epoch, 
                    device = device)

        NFvalidate_epoch(dataloader = validate_loader,
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
             train_loader, validate_loader,
             loss_fn, optimizer, 
             scheduler,
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
                      scheduler = scheduler,
                      writer = writer, ep_curr = epoch, 
                      device = device)

        HNvalidate_epoch(dataloader = validate_loader,
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
                    train_loader, validate_loader,
                    loss_fn, optimizer, 
                    scheduler,
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
                         scheduler = scheduler,
                         writer = writer, ep_curr = epoch, 
                         device = device)

        HN_NFvalidate_epoch(dataloader = validate_loader,
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

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------



def NFtrain_epoch(dataloader, model, 
                loss_fn, optimizer,
                scheduler,
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

        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()



        total_loss += loss.detach() * R.shape[0]
        n_samples += R.shape[0]

    epoch_loss = (total_loss / n_samples).item()
    print(f"loss: {epoch_loss:>7f}")
    writer.add_scalar("Loss/train", epoch_loss, ep_curr)

def NFvalidate_epoch(dataloader, model, 
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
                  scheduler,
                  writer, ep_curr,
                  device = 'cpu'):

    model = model.to(device)
    hyper = hyper.to(device)
    model.eval()
    hyper.train()

    loss = 0.0
    total_loss = torch.zeros((), device=device)
    n_samples = 0

    field_se = torch.zeros(3, device=device)          # NEW: per-field squared error
    grad_norms = []                                   # NEW: pre-clip gradient norms
    pred_absmax = torch.zeros((), device=device)      # NEW: blow-up detector
    feat_norms, mod_log = [], {}                      # NEW: trunk-feature norms, modulation stats
    hook = hyper.trunk.register_forward_hook(         # NEW: captures ||f|| with no extra compute
        lambda m, i, o: feat_norms.append(o.detach().norm(dim=-1)))


    for batch, (X, y, z) in enumerate(dataloader):
        # X = (x, t)
        # y = (rho, u, P)

        coord = X.to(device, non_blocking=True).unsqueeze(1)
        R = y.to(device, non_blocking=True)
        cond = z.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        #apply hypernetwork if needed
        if hyper_type.lower() == 'film':
            # apply hypernetwork
            film = hyper(cond)
            R_pred, __ = model(coord, film=film)
            mod = film
        elif hyper_type.lower() == 'lora':
            # apply hypernetwork
            lora = hyper(cond)
            R_pred, __ = model(coord, lora=lora)
            mod = lora
        else:
            raise ValueError(f"Hypernetwork type {hyper_type} not supported")

        R_pred = R_pred.squeeze(1) 

        assert R_pred.shape == R.shape, f"pred {tuple(R_pred.shape)} vs target {tuple(R.shape)}"

        loss = loss_fn(R_pred, R)

        # propogate gradients
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(hyper.parameters(), grad_clip)
        grad_norms.append(grad_norm.detach())    
        optimizer.step()
        scheduler.step()

        total_loss += loss.detach() * R.shape[0]
        n_samples += R.shape[0]

        field_se += ((R_pred.detach() - R) ** 2).sum(0)                                   # NEW
        pred_absmax = torch.maximum(pred_absmax, R_pred.detach().abs().max())             # NEW
        if not mod_log:                                                                   # NEW
            mod_log = mod_stats(model, mod, hyper_type, "Mod/train")                      # NEW
 
    hook.remove()                                                                         # NEW


    epoch_loss = (total_loss / n_samples).item()
    print(f"loss: {epoch_loss:>7f}")
    writer.add_scalar("Loss/train", epoch_loss, ep_curr)

    # NEW: health logs
    gn = torch.stack(grad_norms)
    fn = torch.cat(feat_norms)
    for name, v in zip(("rho", "u", "P"), (field_se / n_samples).tolist()):
        writer.add_scalar(f"Loss/train_{name}", v, ep_curr)
    writer.add_scalar("Grad/total_mean", gn.mean().item(), ep_curr)
    writer.add_scalar("Grad/total_max", gn.max().item(), ep_curr)
    writer.add_scalar("Grad/clip_frac", (gn > grad_clip).float().mean().item(), ep_curr)   # 1.0 = your clip value
    writer.add_scalar("Feat/norm_mean", fn.mean().item(), ep_curr)
    writer.add_scalar("Feat/norm_max", fn.max().item(), ep_curr)
    writer.add_scalar("Out/pred_absmax", pred_absmax.item(), ep_curr)
    writer.add_scalar("LR/group0", optimizer.param_groups[0]["lr"], ep_curr)
    for k, v in mod_log.items():
        writer.add_scalar(k, v, ep_curr)


def HNvalidate_epoch(dataloader, 
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

    field_se = torch.zeros(3, device=device)          # NEW
    pred_absmax = torch.zeros((), device=device)      # NEW
    feat_norms, mod_log = [], {}                      # NEW
    hook = hyper.trunk.register_forward_hook(         # NEW
        lambda m, i, o: feat_norms.append(o.detach().norm(dim=-1)))


    with torch.no_grad():
        for X, y, z in dataloader:
            coord = X.to(device, non_blocking=True).unsqueeze(1)
            R = y.to(device, non_blocking=True)
            cond = z.to(device, non_blocking=True)

                    #apply hypernetwork if needed
            if hyper_type.lower() == 'film':
                # apply hypernetwork
                film = hyper(cond)
                R_pred, __ = model(coord, film=film)
                mod = film
            elif hyper_type.lower() == 'lora':
                # apply hypernetwork
                lora = hyper(cond)
                R_pred, __ = model(coord, lora=lora)
                mod = lora
            else:
                raise ValueError(f"Hypernetwork type {hyper_type} not supported")

            R_pred = R_pred.squeeze(1)
            assert R_pred.shape == R.shape, f"pred {tuple(R_pred.shape)} vs target {tuple(R.shape)}"   # NEW

            loss = loss_fn(R_pred, R)
            total_loss += loss * R.shape[0]
            n_samples += R.shape[0]

            field_se += ((R_pred - R) ** 2).sum(0)                                            # NEW
            pred_absmax = torch.maximum(pred_absmax, R_pred.abs().max())                      # NEW
            if not mod_log:                                                                   # NEW
                mod_log = mod_stats(model, mod, hyper_type, "Mod/validate")                   # NEW
 
    hook.remove()               

    epoch_loss = (total_loss / n_samples).item()
    print('VALIDATION:')
    print(f"loss: {epoch_loss:>7f}")
    writer.add_scalar("Loss/validate", epoch_loss, ep_curr)

    # NEW: health logs
    fn = torch.cat(feat_norms)
    for name, v in zip(("rho", "u", "P"), (field_se / n_samples).tolist()):
        writer.add_scalar(f"Loss/validate_{name}", v, ep_curr)
    writer.add_scalar("Feat/val_norm_mean", fn.mean().item(), ep_curr)
    writer.add_scalar("Feat/val_norm_max", fn.max().item(), ep_curr)
    writer.add_scalar("Out/val_pred_absmax", pred_absmax.item(), ep_curr)
    for k, v in mod_log.items():
        writer.add_scalar(k, v, ep_curr)



# ----------------------------------------------------------------------------------------------------


def HN_NFtrain_epoch(dataloader, 
                  model, hyper, hyper_type,
                  loss_fn, optimizer,
                  scheduler,
                  writer, ep_curr,
                  device = 'cpu'):

    model = model.to(device)
    hyper = hyper.to(device)
    model.train()
    hyper.train()

    loss = 0.0
    total_loss = torch.zeros((), device=device)
    n_samples = 0

    field_se = torch.zeros(3, device=device)          # NEW: per-field squared error
    hyper_grad_norms = []                                   # NEW: pre-clip gradient norms
    model_grad_norms = [] 
    pred_absmax = torch.zeros((), device=device)      # NEW: blow-up detector
    feat_norms, mod_log = [], {}                      # NEW: trunk-feature norms, modulation stats
    hook = hyper.trunk.register_forward_hook(         # NEW: captures ||f|| with no extra compute
        lambda m, i, o: feat_norms.append(o.detach().norm(dim=-1)))

    for batch, (X, y, z) in enumerate(dataloader):
        # X = (x, t)
        # y = (rho, u, P)
        # z = (t)

        coord = X[:, :1].to(device, non_blocking=True).unsqueeze(1)
        R = y.to(device, non_blocking=True)
        cond = z.unsqueeze(-1).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        #apply hypernetwork if needed
        if hyper_type.lower() == 'film':
            # apply hypernetwork
            film = hyper(cond)
            R_pred, __ = model(coord, film=film)
            mod = film
        elif hyper_type.lower() == 'lora':
            # apply hypernetwork
            lora = hyper(cond)
            R_pred, __ = model(coord, lora=lora)
            mod = lora
        else:
            raise ValueError(f"Hypernetwork type {hyper_type} not supported")

        R_pred = R_pred.squeeze(1) 

        loss = loss_fn(R_pred, R)

        # propogate gradients
        loss.backward()
        hyper_grad_norm = nn.utils.clip_grad_norm_(hyper.parameters(), grad_clip)
        hyper_grad_norms.append(hyper_grad_norm.detach())   
        model_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        model_grad_norms.append(model_grad_norm.detach())    
        # nn.utils.clip_grad_norm_([model.parameters(), hyper.parameters()], 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.detach() * R.shape[0]
        n_samples += R.shape[0]

        field_se += ((R_pred.detach() - R) ** 2).sum(0)                                   # NEW
        pred_absmax = torch.maximum(pred_absmax, R_pred.detach().abs().max())             # NEW
        if not mod_log:                                                                   # NEW
            mod_log = mod_stats(model, mod, hyper_type, "Mod/train")        

    hook.remove()

    epoch_loss = (total_loss / n_samples).item()
    print(f"loss: {epoch_loss:>7f}")
    writer.add_scalar("Loss/train", epoch_loss, ep_curr)

    # NEW: health logs
    gn_base = torch.stack(model_grad_norms)
    gn_hyper = torch.stack(hyper_grad_norms)
    fn = torch.cat(feat_norms)
    for name, v in zip(("rho", "u", "P"), (field_se / n_samples).tolist()):
        writer.add_scalar(f"Loss/train_{name}", v, ep_curr)

    writer.add_scalar("Grad/total_mean [base]", gn_base.mean().item(), ep_curr)
    writer.add_scalar("Grad/total_max [base]", gn_base.max().item(), ep_curr)
    writer.add_scalar("Grad/clip_frac [base]", (gn_base > grad_clip).float().mean().item(), ep_curr)
    writer.add_scalar("Grad/total_mean [hyper]", gn_hyper.mean().item(), ep_curr)
    writer.add_scalar("Grad/total_max [hyper]", gn_hyper.max().item(), ep_curr)
    writer.add_scalar("Grad/clip_frac [hyper]", (gn_hyper > grad_clip).float().mean().item(), ep_curr)

    writer.add_scalar("Feat/norm_mean", fn.mean().item(), ep_curr)
    writer.add_scalar("Feat/norm_max", fn.max().item(), ep_curr)
    writer.add_scalar("Out/pred_absmax", pred_absmax.item(), ep_curr)
    writer.add_scalar("LR/group0", optimizer.param_groups[0]["lr"], ep_curr)
    for k, v in mod_log.items():
        writer.add_scalar(k, v, ep_curr)


def HN_NFvalidate_epoch(dataloader, 
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

    field_se = torch.zeros(3, device=device)          # NEW
    pred_absmax = torch.zeros((), device=device)      # NEW
    feat_norms, mod_log = [], {}                      # NEW
    hook = hyper.trunk.register_forward_hook(         # NEW
        lambda m, i, o: feat_norms.append(o.detach().norm(dim=-1)))

    with torch.no_grad():
        for X, y, z in dataloader:
            coord = X[:, :1].to(device, non_blocking=True).unsqueeze(1)
            R = y.to(device, non_blocking=True)
            cond = z.unsqueeze(-1).to(device, non_blocking=True)

                    #apply hypernetwork if needed
            if hyper_type.lower() == 'film':
                # apply hypernetwork
                film = hyper(cond)
                R_pred, __ = model(coord, film=film)
                mod = film
            elif hyper_type.lower() == 'lora':
                # apply hypernetwork
                lora = hyper(cond)
                R_pred, __ = model(coord, lora=lora)
                mod = lora
            else:
                raise ValueError(f"Hypernetwork type {hyper_type} not supported")


            R_pred = R_pred.squeeze(1)

            loss = loss_fn(R_pred, R)
            total_loss += loss.detach() * R.shape[0]
            n_samples += R.shape[0]

            field_se += ((R_pred - R) ** 2).sum(0)                                            # NEW
            pred_absmax = torch.maximum(pred_absmax, R_pred.abs().max())                      # NEW
            if not mod_log:                                                                   # NEW
                mod_log = mod_stats(model, mod, hyper_type, "Mod/validate")                   # NEW
 
    hook.remove()               

    epoch_loss = (total_loss / n_samples).item()
    print('VALIDATION:')
    print(f"loss: {epoch_loss:>7f}")
    writer.add_scalar("Loss/validate", epoch_loss, ep_curr)

    # NEW: health logs
    fn = torch.cat(feat_norms)
    for name, v in zip(("rho", "u", "P"), (field_se / n_samples).tolist()):
        writer.add_scalar(f"Loss/validate_{name}", v, ep_curr)
    writer.add_scalar("Feat/val_norm_mean", fn.mean().item(), ep_curr)
    writer.add_scalar("Feat/val_norm_max", fn.max().item(), ep_curr)
    writer.add_scalar("Out/val_pred_absmax", pred_absmax.item(), ep_curr)
    for k, v in mod_log.items():
        writer.add_scalar(k, v, ep_curr)




# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
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
        self.IC = IC # (N_time*N_cell, 3) | (rho2_rho1, P2_P1, x0)

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        return self.coords[idx], self.flow[idx], self.IC[idx]

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

def apply_decay_mask(model, lr = 1e-3, weight_decay=1e-2):
    """
    Splits parameters into decaying and non-decaying groups.
    - Excludes: All 1D tensors (biases, 1D vectors).
    - Includes: Weight matrices (2D+ tensors, including LoRA A/B and FiLM projections).
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Exclude biases and any 1D parameter vectors
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay, 'lr': lr},
        {"params": no_decay_params, "weight_decay": 0.0, 'lr': lr}
    ]

    return optim_groups





# NEW: small helper, logs how far the modulation is from "no modulation", per layer
@torch.no_grad()
def mod_stats(model, mod, hyper_type, tag, n=64):
    s = {}
    if hyper_type.lower() == 'film':
        for l, (g, b) in enumerate(mod):                      # 0 = identity
            s[f"{tag}/gamma_dev_L{l}"] = (g[:n] - 1).norm(dim=-1).mean().item()
            s[f"{tag}/beta_L{l}"] = b[:n].norm(dim=-1).mean().item()
    else:
        for l, (A, B) in enumerate(mod):                      # relative size of the weight update
            dW = model.lora_scale * torch.matmul(B[:n], A[:n])
            s[f"{tag}/dW_rel_L{l}"] = (dW.flatten(1).norm(dim=1) / model.layers[l].linear.weight.norm()).mean().item()
    return s
 

