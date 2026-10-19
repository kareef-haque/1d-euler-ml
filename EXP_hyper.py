from neural_fields.hypernet_studies.models import INR, ModFiLM, ModLoRA, torch
from neural_fields.hypernet_studies.sample_exp_two_three import sample_Exp2, sample_Exp3, sample_Exp2_Baseline, sample_Exp3_Baseline
from neural_fields.hypernet_studies.run_samples import exp2_samples, exp3_samples, compare_exp2_exp3, rebuild_report

import yaml
import os
import re
import time

def do_da_train():
    start_time = time.time()
    sample_Exp2()
    sample_Exp3()
    sample_Exp2_Baseline()
    sample_Exp3_Baseline()

    end_time = time.time()

    elapsed_time = end_time - start_time

    print(f"Total execution time: {elapsed_time:.4f} seconds")


def exp2_diagnostics():
 
    from neural_fields.hypernet_studies.diagnostics import exp2_family, run, to_physical, Model, Baseline
 
 
    config = r'.\neural_fields\hypernet_studies\configs'
 
    models_saved = [r'models\exp2\base_FiLM_H0.pth', r'models\exp2\base_FiLM_H1.pth',
                    r'models\exp2\base_LoRA_H0_r4a4.pth', r'models\exp2\base_LoRA_H1_r4a4.pth']
 
    # seed pairs: same kind, same setup, different seed (FiLM x2, LoRA r4a4 x2)
    seed_pairs = [('FiLM_H0', 'FiLM_H1'), ('LoRA_H0_r4a4', 'LoRA_H1_r4a4')]
 
    input_features = 3      # hypernet input is still the raw (rho_ratio, p_ratio, x_split); only p_ratio varies now
 
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
 
 
    # builds the stored runs, the dense exact-solver grid and the hooks (cond / baseline input / exact solution).
    # NOTE: the runs listed here must be the NEW 1-D family (rho_ratio = 0.125, x_split = 0.5 fixed);
    # exp2_family asserts this. Change the index lists if your new data uses different ones.
    fam = exp2_family("generated_data", {"train": [-1, *range(20)], "validate": range(10), "test": range(15)},
                      t_stride=40)
 
 
    models = {}
    with open(os.path.join(config, 'model.yaml'), 'r') as file:
        config = yaml.safe_load(file)
 
 
    baseline_INR = INR(
        N_in=config["model"]["architecture"]["input_features"] + 3,
        N_hidden=config["model"]["architecture"]["hidden_features"],
        N_layers=config["model"]["architecture"]["hidden_layers"],
        N_out=config["model"]["architecture"]["output_features"],
    )
    baseline_weights = torch.load(r'models\baseline\exp2_baseline.pth')
    baseline_INR.load_state_dict(baseline_weights)
 
    baseline = Baseline('baseline', baseline_INR, to_physical=to_physical)
 
 
 
    for M in models_saved:
        weights = torch.load(M)
 
        model_name = os.path.basename(M)
        match_base = re.search(r"base_(.*?)\.pth", model_name)
        extracted_text = match_base.group(1) if match_base else None
 
        rank = 4
        alpha = 4.0
 
        if "LoRA" in extracted_text:
            match_lora = re.search(r"r(\d+)a(\d+)", extracted_text)
            if match_lora:
                rank = int(match_lora.group(1))
                alpha = float(match_lora.group(2))
 
 
        base = INR(
            N_in=config["model"]["architecture"]["input_features"],
            N_hidden=config["model"]["architecture"]["hidden_features"],
            N_layers=config["model"]["architecture"]["hidden_layers"],
            N_out=config["model"]["architecture"]["output_features"],
            lora_rank = rank,
            lora_alpha = alpha
        )
        base.load_state_dict(weights['base'])
 
        if "LoRA" in extracted_text:
            hyper = ModLoRA(
                cond_dim=input_features,
                trunk_hidden=config["hypernetwork"]["trunk_architecture"]["hidden_features"],
                trunk_layers=config['hypernetwork']["trunk_architecture"]["hidden_layers"],
                trunk_out=config["hypernetwork"]["trunk_architecture"]["output_features"],
                inr = base
            )
            type = 'lora'
        else:
            hyper = ModFiLM(
                cond_dim=input_features,
                trunk_hidden=config["hypernetwork"]["trunk_architecture"]["hidden_features"],
                trunk_layers=config['hypernetwork']["trunk_architecture"]["hidden_layers"],
                trunk_out=config["hypernetwork"]["trunk_architecture"]["output_features"],
                inr = base
            )
            type = 'film'
 
        hyper.load_state_dict(weights['hyper'])
 
        # EXP2: the base is frozen (only the hypernet counts as trained)
        models[extracted_text] = Model(extracted_text, base, hyper, type, to_physical=to_physical, base_trained=False)
 
 
 
    results = run(fam, models, baseline, seed_pairs=seed_pairs, out_dir=r'./diagnostics/exp2', device=device)
 
 
def exp3_diagnostics():
 
    from neural_fields.hypernet_studies.diagnostics import exp3_family, run, to_physical, Model, Baseline
 
 
    config = r'.\neural_fields\hypernet_studies\configs'
 
    models_saved = [r'models\exp3\base_FiLM_H0.pth', r'models\exp3\base_FiLM_H1.pth',
                    r'models\exp3\base_LoRA_H0_r4a4.pth', r'models\exp3\base_LoRA_H1_r4a4.pth']
 
    seed_pairs = [('FiLM_H0', 'FiLM_H1'), ('LoRA_H0_r4a4', 'LoRA_H1_r4a4')]
 
    input_features = 1      # hypernet input is t (normalised to 0..1)
 
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
 
 
    # t_stride=10 (was 40) so that each 8 % val/test block holds ~16 snapshots
    fam = exp3_family('./generated_data/train/train_-1.npy', t_stride=10)
 
 
    models = {}
    with open(os.path.join(config, 'model.yaml'), 'r') as file:
        config = yaml.safe_load(file)
 
 
    baseline_INR = INR(
        N_in=config["model"]["architecture"]["input_features"],
        N_hidden=config["model"]["architecture"]["hidden_features"],
        N_layers=config["model"]["architecture"]["hidden_layers"],
        N_out=config["model"]["architecture"]["output_features"],
    )
    baseline_weights = torch.load(r'models\baseline\exp3_baseline.pth')
    baseline_INR.load_state_dict(baseline_weights)
 
    baseline = Baseline('baseline', baseline_INR, to_physical=to_physical)
 
 
 
    for M in models_saved:
        weights = torch.load(M)
 
        model_name = os.path.basename(M)
        match_base = re.search(r"base_(.*?)\.pth", model_name)
        extracted_text = match_base.group(1) if match_base else None
 
        rank = 4
        alpha = 4.0
 
        if "LoRA" in extracted_text:
            match_lora = re.search(r"r(\d+)a(\d+)", extracted_text)
            if match_lora:
                rank = int(match_lora.group(1))
                alpha = float(match_lora.group(2))
 
 
        base = INR(
            N_in=config["model"]["architecture"]["input_features"] - 1,
            N_hidden=config["model"]["architecture"]["hidden_features"],
            N_layers=config["model"]["architecture"]["hidden_layers"],
            N_out=config["model"]["architecture"]["output_features"],
            lora_rank = rank,
            lora_alpha = alpha
        )
        base.load_state_dict(weights['base'])
 
        if "LoRA" in extracted_text:
            hyper = ModLoRA(
                cond_dim=input_features,
                trunk_hidden=config["hypernetwork"]["trunk_architecture"]["hidden_features"],
                trunk_layers=config['hypernetwork']["trunk_architecture"]["hidden_layers"],
                trunk_out=config["hypernetwork"]["trunk_architecture"]["output_features"],
                inr = base
            )
            type = 'lora'
        else:
            hyper = ModFiLM(
                cond_dim=input_features,
                trunk_hidden=config["hypernetwork"]["trunk_architecture"]["hidden_features"],
                trunk_layers=config['hypernetwork']["trunk_architecture"]["hidden_layers"],
                trunk_out=config["hypernetwork"]["trunk_architecture"]["output_features"],
                inr = base
            )
            type = 'film'
 
        hyper.load_state_dict(weights['hyper'])
 
        # EXP3: base and hypernet were trained together
        models[extracted_text] = Model(extracted_text, base, hyper, type, to_physical=to_physical, base_trained=True)
 
 
 
    results = run(fam, models, baseline, seed_pairs=seed_pairs, out_dir=r'./diagnostics/exp3', device=device)


if __name__ == '__main__':

    # do_da_train()

    # exp2_diagnostics()

    # exp3_diagnostics()

    exp2_samples()
    exp3_samples()
    compare_exp2_exp3() 


