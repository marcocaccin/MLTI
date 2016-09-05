import os

import numpy as np

# from ase.atoms import Atoms
# from ase.io import read, write
from ase.units import GPa, J, m, kB, fs, Ang
import quippy 
from matscipy.neighbours import neighbour_list
from quippy import Potential

k_spring = 0.35
initial_strain = 0.10
crack_seed_length = 1./3 # fraction of total length
strain_ramp_length = 15
vacuum = 20
relax_fmax = 0.01
pre_optim = True
relax_slab = True

calc = Potential("IP TS", param_filename='ts_params.xml')
