'''
Fixed global scales (not learned, not state dependent).
Q is divided by QS before the network; the network output is multiplied by FS.
'''
import numpy as np

RHO_REF = 1.0
A_REF = 300.0

QS = np.array([RHO_REF, RHO_REF * A_REF, RHO_REF * A_REF**2])
FS = np.array([RHO_REF * A_REF, RHO_REF * A_REF**2, RHO_REF * A_REF**3])