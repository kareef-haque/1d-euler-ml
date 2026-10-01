'''
NumPy bridge so the classical solver can use a trained NFV flux.
Call with the state padded by exactly g = a // 2 ghost cells per side: (3, N + 2g) -> (3, N + 1)
'''
import os
import numpy as np
import torch

from euler1d.neural.model import NFVModel
from euler1d.neural import stepper


class NFVFlux:
    def __init__(self, model, dtype=torch.float64):
        if isinstance(model, (str, os.PathLike)):
            model = NFVModel.load(model, dtype)
        if model.b != 1:
            raise NotImplementedError('the solver adapter supports b = 1 only')
        self.model = model.to(dtype).eval()
        self.dtype = dtype
        self.g = model.g
        self.gamma = model.extra.get('gamma', 1.4)

    def __call__(self, Q_in, dx=1.0, gamma=1.4):
        x = torch.from_numpy(np.ascontiguousarray(Q_in)).to(self.dtype)[None]
        with torch.no_grad():
            F = stepper.flux(self.model, [x])[0]
        return F.numpy()