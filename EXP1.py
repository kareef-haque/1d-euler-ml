from neural_fields.capability_test.diagnostics import visualize_model
from neural_fields.capability_test.models import INR, SIREN, WIRE
from neural_fields.preprocess import SodPreprocessor
import torch




if __name__ == '__main__':

    train, test = SodPreprocessor()
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