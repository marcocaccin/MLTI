#! /usr/bin/env python
from common_initialiser import *
from ase.constraints import FixBondLength

# warm_start_file = '../c0.35/crack-300K.xyz'

# conserve the weak bond
fix_dist = FixBondLength(*params.tipatoms)

# apply constraints
other_constraints = at.constraints
at.set_constraint(other_constraints + [fix_dist])


init_temp = 300.
temperature_step = (params.sim_T - init_temp) / params.n_steps
def temperature_break(at, dynamics):
    KinEng = at.get_kinetic_energy()
    n_steps = dynamics.get_number_of_steps()
    T_target = init_temp + n_steps * temperature_step
    KinEng_target = 1.5 * at.get_number_of_atoms() * units.kB * T_target
    at.set_momenta(KinEng_target / KinEng * at.get_momenta())
    print("Adjust momenta...")
    return


if params.warm_start_file is not None:
    at_help = quippy.AtomsList(params.warm_start_file, start=-1)[0]
    p = at_help.get_momenta()
    at.set_momenta(p)
    at_help = None
    
dynamics.attach(temperature_break, 12,
                at,
                dynamics) # Print MD info on screen

# Run dynamics at constant reaction coordinate to equilibrate system
dynamics.run(params.n_steps)

# remove temperature_break call
del dynamics.observers[-1]

# Run dynamics at constant reaction coordinate to equilibrate system
dynamics.run(params.n_steps)


at.write('crack-300K.xyz')
