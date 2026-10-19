# Run script: exp2_samples() ("Init. Cond.") and exp3_samples() ("Time") each measure the four samples and draw the figures.
# EDIT THE PATHS in `samples` to match where your samples' checkpoints live; the structure (4 hypernetwork checkpoints + 1 baseline per
# sample) is what matters. replot(out_root) redraws the figures from the saved results.
from neural_fields.hypernet_studies.models import INR, ModFiLM, ModLoRA

import os
import re
import yaml
import torch


def load_models(model_paths, config, input_features, base_inputs, base_trained):
    """Build {name: Model} from a list of checkpoint paths (same loading code as before).
    base_inputs = N_in of the base INR (EXP2: config input_features; EXP3: config input_features - 1);
    base_trained = False for EXP2 (frozen base), True for EXP3."""
    from neural_fields.hypernet_studies.diagnostics import Model, to_physical

    models = {}
    for M in model_paths:
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
            N_in=base_inputs,
            N_hidden=config["model"]["architecture"]["hidden_features"],
            N_layers=config["model"]["architecture"]["hidden_layers"],
            N_out=config["model"]["architecture"]["output_features"],
            lora_rank = rank,
            lora_alpha = alpha
        )
        base.load_state_dict(weights['base'])

        hyper_class = ModLoRA if "LoRA" in extracted_text else ModFiLM
        hyper = hyper_class(
            cond_dim=input_features,
            trunk_hidden=config["hypernetwork"]["trunk_architecture"]["hidden_features"],
            trunk_layers=config['hypernetwork']["trunk_architecture"]["hidden_layers"],
            trunk_out=config["hypernetwork"]["trunk_architecture"]["output_features"],
            inr = base
        )
        hyper.load_state_dict(weights['hyper'])

        models[extracted_text] = Model(extracted_text, base, hyper, 'lora' if "LoRA" in extracted_text else 'film',
                                       to_physical=to_physical, base_trained=base_trained)

    return models


def load_baseline(path, config, n_inputs):
    from neural_fields.hypernet_studies.diagnostics import Baseline, to_physical

    inr = INR(
        N_in=n_inputs,
        N_hidden=config["model"]["architecture"]["hidden_features"],
        N_layers=config["model"]["architecture"]["hidden_layers"],
        N_out=config["model"]["architecture"]["output_features"],
    )
    inr.load_state_dict(torch.load(path))

    return Baseline('baseline', inr, to_physical=to_physical)


def run_samples(fam, samples, config_dir, input_features, base_inputs, base_trained, baseline_inputs, out_root, device):
    """Measure every sample, then draw the figures of this experiment (aggregated over the samples). fam is built ONCE by the caller (the
    exact-solver grids are the slow part).
    Files under out_root:
        A1_..., B2_..., C1_..., C2_..., D1_..., D4_..., F3_... .png   the figures (mean +- std over the samples)
        snapshots/<sample>.png                                        exact vs predicted solutions of each sample
        results.pkl                                                   everything run() measured, so replot() needs no models"""
    from neural_fields.hypernet_studies.diagnostics import run
    from neural_fields.hypernet_studies.diag_plots import plot_all, save_results

    with open(os.path.join(config_dir, 'model.yaml'), 'r') as file:
        config = yaml.safe_load(file)

    results = {}
    for sample_name, spec in samples.items():
        print(f"\n######## {sample_name} ########")
        models = load_models(spec["models"], config, input_features, base_inputs, base_trained)
        baseline = load_baseline(spec["baseline"], config, baseline_inputs)
        results[sample_name] = run(fam, models, baseline, device=device)

    save_results(results, os.path.join(out_root, "results.pkl"))                  # first: a failing figure must not lose the (slow) measurements
    plot_all(results, out_root)

    return results


def replot(out_root):
    """Redraw the figures of one experiment from out_root/results.pkl without loading any model (also works on results written by the
    earlier, larger run())."""
    from neural_fields.hypernet_studies.diag_plots import plot_all, load_results

    return plot_all(load_results(os.path.join(out_root, "results.pkl")), out_root)


def exp2_samples():
    from neural_fields.hypernet_studies.diagnostics import exp2_family

    config_dir = r'.\neural_fields\hypernet_studies\configs'
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    # EDIT: one entry per sample; each has its own base (inside the checkpoints) and its own baseline
    samples = {f"sample_{i}": {"models": [rf'models\sample{i}\exp2\base_FiLM_H0.pth', rf'models\sample{i}\exp2\base_FiLM_H1.pth',
                                          rf'models\sample{i}\exp2\base_LoRA_H0_r4a4.pth', rf'models\sample{i}\exp2\base_LoRA_H1_r4a4.pth'],
                               "baseline": rf'models\sample{i}\baseline\exp2_baseline.pth'} for i in (1, 2, 3, 4)}

    fam = exp2_family("generated_data", {"train": [-1, *range(20)], "validate": range(5), "test": range(5)}, t_stride=40)

    with open(os.path.join(config_dir, 'model.yaml'), 'r') as file:
        n_xt = yaml.safe_load(file)["model"]["architecture"]["input_features"]

    # EXP2: hypernet input = (rho_ratio, p_ratio, x_split) -> 3; base = (x, t); baseline = (x, t, rho_ratio, p_ratio, x_split)
    return run_samples(fam, samples, config_dir, input_features=3, base_inputs=n_xt, base_trained=False,
                       baseline_inputs=n_xt + 3, out_root=r'./diagnostics/exp2_samples', device=device)


def exp3_samples():
    from neural_fields.hypernet_studies.diagnostics import exp3_family

    config_dir = r'.\neural_fields\hypernet_studies\configs'
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    samples = {                                                             # EDIT
        f"sample_{i}": {"models": [rf'models\sample{i}\exp3\base_FiLM_H0.pth', rf'models\sample{i}\exp3\base_FiLM_H1.pth',
                                   rf'models\sample{i}\exp3\base_LoRA_H0_r4a4.pth', rf'models\sample{i}\exp3\base_LoRA_H1_r4a4.pth'],
                        "baseline": rf'models\sample{i}\baseline\exp3_baseline.pth'} for i in (1, 2, 3, 4)}

    fam = exp3_family('./generated_data/train/train_-1.npy', t_stride=10)

    with open(os.path.join(config_dir, 'model.yaml'), 'r') as file:
        n_xt = yaml.safe_load(file)["model"]["architecture"]["input_features"]

    # EXP3: hypernet input = t -> 1; base = x (n_xt - 1); baseline = (x, t)
    return run_samples(fam, samples, config_dir, input_features=1, base_inputs=n_xt - 1, base_trained=True,
                       baseline_inputs=n_xt, out_root=r'./diagnostics/exp3_samples', device=device)