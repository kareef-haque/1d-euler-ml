'''
Neural Field Definition
- For use in spatiotemporal modeling of the 1D Euler Equations
    - Specifically for variatons on the Sod Shock Tube Problem

IMPLICIT NEURAL REPRESENTATION
- Three different variations will be tested [all vary only the activation funciton]
    - Base (sigma = ReLU)
    - SIREN (sigma = sine)
    - WIRE (sigma = gabor wavelet)

- Implemention Notes
    - One Defined model, with activation function being modular
    - Other Modular Aspects defined during initalization
        - Layers
        - Hidden features
        - Input/Output features
        '''

import torch
import torch.nn as nn
import numpy as np

class INR(nn.Module):
    """
    An Implicit Neural Representation (INR) designed for spatiotemporal signal modeling.

    Attributes:
        N_in (int): Input dimensionality per time step
        N_hidden_features (int): Number of features in hidden layers
        N_hidden_layers (int): Number of hidden layers
        N_out (int): Output dimensionality
    """
    def __init__(self, N_in, 
                 N_hidden_features,
                 N_hidden_layers, 
                 N_out,
                 ):
        super().__init__()

        # Create a list to hold all our custom layer objects.
        layers = []

        # Input Layer: Maps input features (N_in) to hidden features (N_hidden_features).
        layers.append(nn.Linear(N_in, N_hidden_features))
        # Non-linear activation function: Adds non-linearity by taking the exponential of the input values minus 1.
        layers.append(nn.ReLU())

        # Hidden Layers: Repeats for each layer. Maps hidden features to themselves.
        for _ in range(N_hidden_layers):
            layers.append(nn.Linear(N_hidden_features, N_hidden_features))
            layers.append(nn.ReLU())

        # Output Layer: Maps final hidden features to output features (N_out).
        layers.append(nn.Linear(N_hidden_features, N_out))

        # Stack all layer objects into a single Sequential object.
        self.neural_field = nn.Sequential(*layers)

    def forward(self, x):
        # Perform the forward pass through the entire neural network.
        x = x.to(torch.float32)
        x = x.clone().detach().requires_grad_(True)
        return self.neural_field(x), x



# ---------------------------------------------------------------------------------------------------
# -----------------------------------------------SIREN-----------------------------------------------
# ---------------------------------------------------------------------------------------------------


class SIREN(nn.Module):
    """
    An Implicit Neural Representation (INR) designed for spatiotemporal signal modeling.
    - Utilizing Sine activation function to better capture high frequencies

    CITE:
        Implicit Neural Representations with Periodic Activation Functions
        Vincent Sitzmann, Julien N. P. Martel, Alexander W. Bergman, David B. Lindell, Gordon Wetzstein
        https://doi.org/10.48550/arXiv.2006.09661

    Attributes:
        N_in (int): Input dimensionality per time step
        N_hidden_features (int): Number of features in hidden layers
        N_hidden_layers (int): Number of hidden layers
        N_out (int): Output dimensionality
        omega_0 (float): Frequency of the sine activation function
        first_omega_0 (float): Frequency of the sine activation function for the first layer

    """

    def __init__(self, N_in, 
                    N_hidden_features,
                    N_hidden_layers, 
                    N_out,
                    omega_0,
                    first_omega_0,
                    final_linear=False
                    ):
        super().__init__()

        self.layers = []
        self.first_omega_0 = first_omega_0
        self.omega_0 = omega_0

        self.layers.append(SineLayer(N_in=N_in, 
                                     N_out=N_hidden_features,
                                     omega_0 = first_omega_0,
                                     is_first=True)
                                     )

        for _ in range(N_hidden_layers):
            self.layers.append(SineLayer(N_in=N_hidden_features, 
                                         N_out=N_hidden_features,
                                         omega_0 = omega_0)
                                         )

        if final_linear:
            layer_final = nn.Linear(N_hidden_features, N_out)

            with torch.no_grad():
                layer_final.weight.uniform_(-np.sqrt(6/N_hidden_features) / self.omega_0,
                                             np.sqrt(6/N_hidden_features) / self.omega_0)

        else:
            layer_final = SineLayer(N_in=N_hidden_features, 
                                    N_out=N_out,
                                    omega_0 = omega_0
                                    )
            
        self.layers.append(layer_final)

        self.neural_field = nn.Sequential(*self.layers)

    def forward(self, x):
        x = x.to(torch.float32)
        x = x.clone().detach().requires_grad_(True) # for allowing the taking of derivatives w.r.t to input?
        return self.neural_field(x), x


class SineLayer(nn.Module):
    """
    An Implicit Neural Representation (INR) designed for spatiotemporal signal modeling.
    - Utilizing Sine activation function to better capture high frequencies

    IMPLMENTATION OF A SINE LAYER
        - Integrate Layer and sine activation function as one

    CITE:
        Implicit Neural Representations with Periodic Activation Functions
        Vincent Sitzmann, Julien N. P. Martel, Alexander W. Bergman, David B. Lindell, Gordon Wetzstein
        https://doi.org/10.48550/arXiv.2006.09661

    Attributes:
        N_in (int): Input dimensionality
        N_out (int): Output dimensionality
        omega_0 (float): Frequency of the sine activation function
        is_first (bool): impacts weight initalization
        bias (bool): i dunno

    """

    def __init__(self, N_in, 
                 N_out,
                 omega_0,
                 is_first = False,
                 bias=True
                 ):
        super().__init__()

        self.omega_0 = omega_0

        self.linear = nn.Linear(N_in, N_out, bias=bias)
        self.init_weights(is_first, N_in)

    def init_weights(self, is_first, N_in):
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1/N_in, 
                                             1/N_in)
            else:
                self.linear.weight.uniform_(-np.sqrt(6/N_in) / self.omega_0,
                                             np.sqrt(6/N_in) / self.omega_0)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


    
# ----------------------------------------------------------------------------------------------------
# ------------------------------------------------WIRE------------------------------------------------
# ----------------------------------------------------------------------------------------------------


class WIRE(nn.Module):
    """
    An Implicit Neural Representation (INR) designed for spatiotemporal signal modeling.
    - Utilizing Gabor Wavelet activation function to better capture high frequencies

    CITE:
        WIRE: Wavelet Implicit Neural Representations
        Vishwanath Saragadam, Daniel LeJeune, Jasper Tan, Guha Balakrishnan, Ashok Veeraraghavan, Richard G. Baraniuk
        https://doi.org/10.48550/arXiv.2301.05187

    Attributes:
        N_in (int): Input dimensionality per time step
        N_hidden_features (int): Number of features in hidden layers
        N_hidden_layers (int): Number of hidden layers
        N_out (int): Output dimensionality
        omega_0 (float): Frequency of the Gabor wavelet
        s_0 (float): Spread of the Gabor wavelet

    """

    def __init__(self, N_in, 
                    N_hidden_features,
                    N_hidden_layers, 
                    N_out,
                    omega_0,
                    s_0,
                    first_omega_0,
                    first_s_0, 
                    final_linear=False
                    ):
        super().__init__()

        self.layers = []
        self.first_omega_0 = first_omega_0
        self.first_s_0 = first_s_0
        self.omega_0 = omega_0
        self.s_0 = s_0
        
        self.layers.append(WIRELayer(N_in=N_in, 
                                     N_out=N_hidden_features,
                                     omega_0 = first_omega_0,
                                     s_0 = first_s_0,
                                     is_first=True)
                                     )

        for _ in range(N_hidden_layers):
            self.layers.append(WIRELayer(N_in=N_hidden_features, 
                                         N_out=N_hidden_features,
                                         omega_0 = omega_0,
                                         s_0 = s_0)
                                         )

        if final_linear:
            layer_final = nn.Linear(N_hidden_features, N_out)

            with torch.no_grad():
                layer_final.weight.uniform_(-np.sqrt(6/N_hidden_features*self.omega_0),
                                             np.sqrt(6/N_hidden_features*self.omega_0))

        else:
            layer_final = WIRELayer(N_in=N_hidden_features, 
                                    N_out=N_out,
                                    omega_0 = omega_0,
                                    s_0 = s_0
                                         )
            
        self.layers.append(layer_final)

        self.neural_field = nn.Sequential(*self.layers)

    def forward(self, x):
        x = x.to(torch.float32)
        x = x.clone().detach().requires_grad_(True) # for allowing the taking of derivatives w.r.t to input?
        return self.neural_field(x), x


class WIRELayer(nn.Module):
    """
    An Implicit Neural Representation (INR) designed for spatiotemporal signal modeling.
    - Utilizing Gabor Wavelet activation function to better capture high frequencies

    CITE:
        WIRE: Wavelet Implicit Neural Representations
        Vishwanath Saragadam, Daniel LeJeune, Jasper Tan, Guha Balakrishnan, Ashok Veeraraghavan, Richard G. Baraniuk
        https://doi.org/10.48550/arXiv.2301.05187

    Attributes:
        N_in (int): Input dimensionality per time step
        N_out (int): Output dimensionality
        omega_0 (float): Frequency of the Gabor wavelet
        s_0 (float): Spread of the Gabor wavelet

    """

    def __init__(self, N_in, 
                 N_out,
                 omega_0,
                 s_0,
                 is_first = False,
                 bias=True
                 ):
        super().__init__()

        self.omega_0 = omega_0
        self.s_0 = s_0

        self.linear = nn.Linear(N_in, N_out, bias=bias)
        self.init_weights(is_first, N_in)

    def init_weights(self, is_first, N_in):
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1/N_in, 
                                             1/N_in)
            else:
                self.linear.weight.uniform_(-np.sqrt(6/(N_in*self.omega_0)),
                                             np.sqrt(6/(N_in*self.omega_0)))

    def forward(self, x):
        # utilizes the imaginary part of the complex Gabor Wavelet
        return torch.sin(self.omega_0 * self.linear(x))*torch.exp(-(self.s_0 * self.linear(x))**2)

