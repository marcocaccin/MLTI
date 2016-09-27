#! /usr/bin/env python
from __future__ import division, print_function

import numpy as np

import ase.units as units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.utils.geometry import find_mic

from ase_addon import SlowGrowthBondLength, Hookean

import quippy
from quippy.io import AtomsWriter

import os
import sys
import argparse

# ***** Parse input parameters *****
parser = argparse.ArgumentParser(description='Input parameters for MD initialisation.')
# must include
parser.add_argument('-f','--config_filename', help='Input xyz structure', type=str)
parser.add_argument('--tipatoms', nargs=2, help='tip atoms indices', type=int)
# nice to have
parser.add_argument('-k','--k_spring', help='spring constant', type=float, default=-1)
parser.add_argument('-warmf','--warm_start_file', help='xyz containing initial momenta', type=str, default=None)
parser.add_argument('-nT','--n_steps', help='Thermalisation timesteps', type=int, default=3000)
# no need to touch
parser.add_argument('-e','--initial_strain', help='strain', type=float, default=0.1)
parser.add_argument('-T','--sim_T', help='Simulation temperature (K)', type=float, default=300.0)
parser.add_argument('-speed','--bondlength_speed', help='Bondlength speed in TI', type=float, default=5e-4)
parser.add_argument('-dt','--timestep', help='Timestep (NB: time base units are not fs!)', type=float, default=1.0*units.fs)
parser.add_argument('-dt_traj','--traj_interval', help='Trajectory write frequency', type=int, default=10)
# ***** End input parameters *****

params = parser.parse_args()

calc = quippy.Potential('IP TS', param_filename="ts_params.xml")
tipatoms = params.tipatoms

folder, config_filename = os.path.split(params.config_filename)
os.chdir(folder)
at = quippy.AtomsList(config_filename, start=-1)[0]

# automatically set spring constant to default Gc value if not manually set
if params.k_spring == -1:
    if 'c' in folder:
        params.k_spring = 0.35
    elif 'a' in folder:
        params.k_spring = 0.80
        print("Inferred k_spring %f" % params.k_spring)
    else:
        print("Please specify the spring constant.")
        sys.exit(1)


print("Running MD with following parameters:")
print(params)

# ***** Decide names of outputs *****
basename = config_filename.strip('.xyz')

jjj = 0
while (os.path.exists('md-%s-%03d.xyz' % (basename, jjj)) or
       os.path.exists('db-%s-%03d.csv' % (basename, jjj))):
    jjj += 1
traj_name = 'md-%s-%03d.xyz' % (basename, jjj)
db_name = 'db-%s-%03d.csv' % (basename, jjj)
log_name = 'md-%s-%03d.log' % (basename, jjj)

# ***** Initialise Atoms object *****
if not at.has_property('momenta'):
    print("Resetting momenta to desired temperature")
    MaxwellBoltzmannDistribution(at, 2.0*params.sim_T*units.kB)

try:
    clamp_mask = at.get_array('clamp_mask')
    fix_line_mask = at.get_array('fix_line_mask')
    initial_positions = at.get_array('pos_orig')
except:
    at_help = quippy.Atoms('crack.xyz')
    clamp_mask = at_help.get_array('clamp_mask')
    # fix_line_mask = at_help.get_array('fix_line_mask')
    initial_positions = at_help.get_array('pos_orig')
    at_help = None

at.set_calculator(calc)

# set atoms order
_, dists = find_mic(initial_positions - initial_positions[tipatoms[0]], at.get_cell())
at_order = np.argsort(dists)[:120]


# ***** Setup constraints *****
# springs = [Hookean(a1=i, a2=[0,-1,0,initial_positions[i,1]], k=k_spring, rt=0) for i in np.where(clamp_mask)[0]]
springs = [Hookean(a1=i, a2=initial_positions[i], k=params.k_spring, rt=0) for i in np.where(clamp_mask)[0]]

from ase.constraints import FixedLine
left = initial_positions[:,0].min()
right = initial_positions[:,0].max()
fix_line_mask = np.logical_or(initial_positions[:,0] > right - 5., initial_positions[:,0] < left + 5.)
fix_line_idx = np.where(fix_line_mask)[0]
fix_line = [FixedLine(i, np.array([0, 1, 0])) for i in fix_line_idx]

# apply constraints
at.set_constraint(springs + fix_line)

# general cleanup of Atoms object
for k in at.arrays.keys():
    if k not in ['species', 'positions', 'momenta', 'numbers']:
        at.set_array(k, None)
for k in at.info.keys():
    if k not in ['energy', 'strain']:
        del at.info[k]      


# ***** Setup and run MD *****

# Initialise the dynamical system
initial_momenta = at.get_momenta()
dynamics = Langevin(at, params.timestep, params.sim_T * units.kB, 8e-4, logfile = log_name)


# Save frames to the trajectory every `traj_interval` time steps
out = AtomsWriter(traj_name)
def trajectory_write(at):
    atout = at.copy()
    for k in atout.arrays.keys():
        if k not in ['species', 'positions', 'momenta', 'numbers', 'force']:
            atout.set_array(k, None)
    for k in atout.info.keys():
        if k not in ['energy','time','temperature','pbc','Lattice']:
            del atout.info[k]
    at.info['energy'] = at.get_potential_energy()
    out.write(atout)


dynamics.attach(trajectory_write, params.traj_interval,
                at) # Save trajectory
