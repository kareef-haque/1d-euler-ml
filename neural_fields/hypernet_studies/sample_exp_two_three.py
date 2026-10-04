from neural_fields.hypernet_studies.train import train_base, train_hyper, train_EndtoEnd
from neural_fields.hypernet_studies.models import INR

import os
import gc
import yaml
import torch 



def sample_Exp2():
    '''
    Experiment to test hypernetwork ability to modulate through function space
        - Vary initial Conditions
    '''
    torch.set_float32_matmul_precision('high')
    
    save_path = r'./models'
    os.makedirs(save_path, exist_ok=True)

    exp2_dir= os.path.join(save_path, 'exp2')
    os.makedirs(exp2_dir, exist_ok=True)
    save_path = exp2_dir

    # base = train_base()
    # base_state = base.state_dict() # Cache the state dict to use for later sweeps
    # flush_memory()


    # FiLM1 = train_hyper(neural_field=base,
    #                     input_features=3,
    #                     hyper_type='FiLM')

    # checkpoint = {
    #     "base": base.state_dict(),
    #     "hyper": FiLM1.state_dict()
    #     }
    # torch.save(checkpoint, 
    #             os.path.join(save_path, f'base_FiLM_H1.pth'))

    # del FiLM1
    # flush_memory()



    
    # FiLM2 = train_hyper(neural_field=base,
    #                     input_features=3,
    #                     hyper_type='FiLM')
    # checkpoint = {
    #     "base": base.state_dict(),
    #     "hyper": FiLM2.state_dict()
    #     }
    # torch.save(checkpoint, 
    #             os.path.join(save_path, f'base_FiLM_H2.pth'))

    # del FiLM2
    # flush_memory()



    # #r = 4, alpha = 4
    # LoRA1 = train_hyper(neural_field=base,
    #                     input_features=3,
    #                     hyper_type='LoRA')
    # checkpoint = {
    #             "base": base.state_dict(),
    #             "hyper": LoRA1.state_dict()
    #             }
    # torch.save(checkpoint, 
    #             os.path.join(save_path, f'base_LoRA_H1_r4a4.pth'))


    # del LoRA1
    # flush_memory()
    
    # #r = 4, alpha = 4
    # LoRA0 = train_hyper(neural_field=base,
    #                     input_features=3,
    #                     hyper_type='LoRA')
    # checkpoint = {
    #             "base": base.state_dict(),
    #             "hyper": LoRA0.state_dict()
    #             }
    # torch.save(checkpoint, 
    #             os.path.join(save_path, f'base_LoRA_H0_r4a4.pth'))

    # del LoRA0
    # flush_memory()

    # # We can now completely delete the original base model since we have its state_dict
    # del base 
    # flush_memory()

    chombo = torch.load(r'models\exp2\base_LoRA_H0_r4a4.pth')
    base_state = chombo['base']


    #r = 1, alpha = 1
    base_Lora2 = copy_BaseNet_for_LoRA_Sweep(source = base_state,
                                                rank = 1, alpha = 1.0)
    LoRA2 = train_hyper(neural_field=base_Lora2,
                        input_features=3,
                        hyper_type='LoRA')
    checkpoint = {
                "base": base_state,
                "hyper": LoRA2.state_dict()
                }
    torch.save(checkpoint, 
                os.path.join(save_path, f'base_LoRA_H2_r1a1.pth'))
    del LoRA2, base_Lora2
    flush_memory()
    
    #r = 4, alpha = 16
    base_Lora3 = copy_BaseNet_for_LoRA_Sweep(source = base_state,
                                                rank = 4, alpha = 16.0)
    LoRA3 = train_hyper(neural_field=base_Lora3,
                        input_features=3,
                        hyper_type='LoRA')
    checkpoint = {
                "base": base_state,
                "hyper": LoRA3.state_dict()
                }
    torch.save(checkpoint, 
                os.path.join(save_path, f'base_LoRA_H3_r4a16.pth'))
    del LoRA3, base_Lora3
    flush_memory()



def sample_Exp3():

    '''
    Experiment to test hypernetwork ability to modulate through function space
        - use a hypernetwork to study relation between weight dynamics and solution temporal dynamics
    '''
    torch.set_float32_matmul_precision('high')
    
    save_path = r'./models'
    os.makedirs(save_path, exist_ok=True)

    exp3_dir= os.path.join(save_path, 'exp3')
    os.makedirs(exp3_dir, exist_ok=True)
    save_path = exp3_dir

    base, FiLM1 = train_EndtoEnd(input_features=1,
                                 hyper_type = 'FiLM',)

    checkpoint = {
        "base": base.state_dict(),
        "hyper": FiLM1.state_dict()
        }
    torch.save(checkpoint, 
                os.path.join(save_path, f'base_FiLM_H1.pth'))
    
    del base, FiLM1
    flush_memory()

    
    base, FiLM2 = train_EndtoEnd(input_features=1,
                                hyper_type = 'FiLM',)

    checkpoint = {
        "base": base.state_dict(),
        "hyper": FiLM2.state_dict()
        }
    torch.save(checkpoint, 
                os.path.join(save_path, f'base_FiLM_H2.pth'))

    del base, FiLM2
    flush_memory()


    #r = 4, alpha = 4
    base, LoRA1 = train_EndtoEnd(input_features=1,
                                 hyper_type = 'LoRA',
                                 lora_rank=4,
                                 lora_alpha=4.0)

    checkpoint = {
        "base": base.state_dict(),
        "hyper": LoRA1.state_dict()
        }
    torch.save(checkpoint, 
                os.path.join(save_path, f'base_LoRA_H1_r4a4.pth'))

    del base, LoRA1
    flush_memory()
    
    #r = 4, alpha = 4
    base, LoRA0 = train_EndtoEnd(input_features=1,
                                 hyper_type = 'LoRA',
                                 lora_rank=4,
                                 lora_alpha=4.0)

    checkpoint = {
        "base": base.state_dict(),
        "hyper": LoRA0.state_dict()
        }
    torch.save(checkpoint, 
                os.path.join(save_path, f'base_LoRA_H0_r4a4.pth'))
    del base, LoRA0
    flush_memory()
    
    #r = 1, alpha = 1
    base, LoRA2 = train_EndtoEnd(input_features=1,
                                 hyper_type = 'LoRA',
                                 lora_rank=1,
                                 lora_alpha=1.0)

    checkpoint = {
        "base": base.state_dict(),
        "hyper": LoRA2.state_dict()
        }
    torch.save(checkpoint, 
                os.path.join(save_path, f'base_LoRA_H2_r1a1.pth'))

    del base, LoRA2
    flush_memory()
    
    #r = 4, alpha = 16
    base, LoRA3 = train_EndtoEnd(input_features=1,
                                 hyper_type = 'LoRA',
                                 lora_rank=4,
                                 lora_alpha=4.0)

    checkpoint = {
        "base": base.state_dict(),
        "hyper": LoRA3.state_dict()
        }
    torch.save(checkpoint, 
                os.path.join(save_path, f'base_LoRA_H3_r4a15.pth'))

    del base, LoRA3
    flush_memory()



def copy_BaseNet_for_LoRA_Sweep(source, rank, alpha, 
                                config =r'.\neural_fields\hypernet_studies\configs'):

    with open(os.path.join(config, 'model.yaml'), 'r') as file:
        config = yaml.safe_load(file)
    EulerNF = INR(
            N_in=config["model"]["architecture"]["input_features"],
            N_hidden=config["model"]["architecture"]["hidden_features"],
            N_layers=config["model"]["architecture"]["hidden_layers"],
            N_out=config["model"]["architecture"]["output_features"],
            lora_rank = rank,
            lora_alpha = alpha
        )

    weights = source
    EulerNF.load_state_dict(weights)
    return EulerNF


# Helper function to clear GPU cache
def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()
