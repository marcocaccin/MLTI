#! /usr/bin/env python
from __future__ import print_function, division

from common_initialiser import *

def write_db(atoms, db_name):
    with open(db_name, 'ab') as db_file:
        cnstr = atoms.constraints[-1]
        d = cnstr.distance
        # constraint force is not contained yet in cnstr for some reason
        forces = atoms.get_array('force')
        at_dummy = atoms.copy()
        cnstr.adjust_forces(at_dummy, forces)        
        f = np.dot(cnstr.constraint_force, cnstr.direction)
        print("%.5f, %.5f" % (np.abs(d), f), file=db_file)
    return

dist_initial = 3
dist_final = 7.2
run_forth = True
run_back = False

other_constraints = at.constraints

if run_forth:
    # ***** Setup constraints *****
    fix_dist = GentlyDriftBondLength(*params.tipatoms, bond_speed = 0, direction = params.tipatoms)
    fix_dist.adjust_direction(at)

    # find rate of bond length increase for the current simulation
    dist_at = np.abs(np.dot(at.positions[params.tipatoms[0]] - at.positions[params.tipatoms[1]], fix_dist.direction))
    bondlength_span = dist_initial - dist_at
    n_steps = int(bondlength_span / params.bondlength_speed)
    
    fix_dist.bond_speed = np.sign(n_steps) * params.bondlength_speed
    
    # apply constraints
    at.set_constraint(other_constraints + [fix_dist])
    
    # Run dynamics reversing reaction coordinate to dist_initial angstrom
    dynamics.atoms.set_momenta(initial_momenta)
    dynamics.run(np.abs(n_steps))
    
db_file = open(db_name, 'wb')
print("eps, k, i0 , i1", file=db_file)
print("%.3f, %.2f, %d, %d" % (params.initial_strain, params.k_spring, params.tipatoms[0], params.tipatoms[1]), file=db_file)
db_file.close()
dynamics.attach(write_db, params.traj_interval, at, db_name)

if run_forth:
    try:
        import os
        home = os.path.expanduser("~")
        os.rename(traj_name, '%s/.Trash/%s' % (home, traj_name))
    except:
        pass

    # find rate of bond length increase for the current simulation
    bondlength_span = dist_final - dist_initial
    n_steps = int(bondlength_span / params.bondlength_speed)
    # ***** Setup constraints *****
    fix_dist = GentlyDriftBondLength(*params.tipatoms, bond_speed = np.sign(n_steps) * params.bondlength_speed, direction = params.tipatoms)
    # apply constraints
    at.set_constraint(other_constraints + [fix_dist])
    
    # Run dynamics until tipatoms are at dist_intermediate
    dynamics.run(np.abs(n_steps))


if run_back:
    # find rate of bond length increase for the current simulation
    bondlength_span = dist_initial - dist_final
    n_steps = int(bondlength_span / params.bondlength_speed)
    # ***** Setup constraints *****
    fix_dist = GentlyDriftBondLength(*params.tipatoms, bond_speed = np.sign(n_steps) * params.bondlength_speed, direction = params.tipatoms)
    # apply constraints
    at.set_constraint(other_constraints + [fix_dist])
    
    # Run dynamics until tipatoms are at dist_intermediate
    dynamics.run(np.abs(n_steps))
