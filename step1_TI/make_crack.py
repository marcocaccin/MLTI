#! /usr/bin/env python

import numpy as np

import ase.io
import ase.units as units
from matscipy.fracture_mechanics.crack import thin_strip_displacement_y
import quippy
import sys

import params
from minim import constrained_minim

def shrink_cell(atoms):
    """
    Shrink Atoms cell to match min-max in every direction
    """
    r = atoms.get_positions()
    mins = [pos.min() for pos in r.T]
    maxs = [pos.max() for pos in r.T]
    atoms.set_cell([b - a for a, b in zip(mins, maxs)])
    return


def center_atoms(atoms, vacuum=None, axis=(0, 1, 2)):
    """                                               
    (1) Removes existing vacuum, (2) adds specified vacuum and (3) centers system along specified directions.
    Substitute for the screwed up quippy Atoms.center()
    """
    
    if vacuum is not None:
        shrink_cell(atoms)
        atoms.cell += np.eye(3)*vacuum
    
    axis_mask = [True if ax in axis else False for ax in range(3)]
    atoms.positions -= atoms.positions.mean(axis=0) * axis_mask
    return 


calc = quippy.Potential('IP TS', param_filename="ts_params.xml")


# Load atomic configuration
fld = sys.argv[1].strip()
params.os.chdir(fld)

crack_slab = quippy.Atoms('slab.xyz')
print('Unit slab with %d atoms per unit cell:' % len(crack_slab))
print(crack_slab.cell)
print('')


in_put = raw_input('Perform dry system generation (guess configuration)?')
GUESS_RUN = (in_put in ['y', 'Y', 'yes'])
if GUESS_RUN:
    print("Performing dry run...")
else:
    print("Performing full system construction...")


# open up the cell along x and y by introducing some vacuum
center_atoms(crack_slab, vacuum=params.vacuum)

# centre the slab on the origin
crack_slab.positions[:, 0] -= crack_slab.positions[:, 0].mean()
crack_slab.positions[:, 1] -= crack_slab.positions[:, 1].mean()
crack_slab.positions[:, 1] += 1e-1 # small nudge to give asymmetry in case atoms lie exactly halfway

right, top, _ = crack_slab.positions.max(axis=0)
left, bottom, _ = crack_slab.positions.min(axis=0)
orig_width = right - left
orig_height = top - bottom

crack_seed_length = params.crack_seed_length * orig_width
# subtract 5.3*N to crack_seed_length to reduce crack length of N bonds
crack_seed_length -= (5.3 * 0) 
print(('Made slab with %d atoms, original width and height: %.1f x %.1f A^2' %
       (len(crack_slab), orig_width, orig_height)))

# ****** Apply initial strain ramp *****

strain = params.initial_strain

displacement = thin_strip_displacement_y(
    crack_slab.positions[:, 0],
    crack_slab.positions[:, 1],
    strain,
    left + crack_seed_length,
    left + crack_seed_length +
    params.strain_ramp_length)

if GUESS_RUN:
    crack_slab.positions[:, 1] += displacement

    # ******** Save output file ********** 
    center_atoms(crack_slab, vacuum=params.vacuum)
    pretty = ase.atoms.Atoms(crack_slab.get_atomic_numbers(), crack_slab.get_positions(), cell=crack_slab.get_cell())
    pretty.set_array('index', np.arange(len(pretty)))
    pretty.write('crack_guess.xyz', format='extxyz')
else:
    # we need to check that the non-strained non-notched slab is relaxed already!!!
    from ase.optimize.precon import LBFGS
    crack_slab.set_calculator(calc)
    LBFGS(crack_slab).run(fmax=params.relax_fmax)
    crack_slab.write("slab.xyz")


    swap = np.loadtxt('swap_topbottom_atoms.csv', dtype='int')
    displacement[swap] = -displacement[swap]
    crack_slab.positions[:, 1] += displacement

    # make all positions positive
    crack_slab.set_positions(crack_slab.get_positions() - crack_slab.get_positions().min(axis=0) + 5*np.ones(3))

    try:
        delete_atoms = np.loadtxt('delete_atoms.csv', dtype='int')
        delete_atoms = np.atleast_1d(delete_atoms)
        crack_slab.set_array('orig_index', np.arange(len(crack_slab)))
        crack_slab.remove_atoms(atom_indices=(delete_atoms + 1))
    except Exception as err:
        print(err)
        pass

    # Save all calculated materials properties inside the Atoms object
    crack_slab.info['OrigWidth'] = orig_width
    crack_slab.info['OrigHeight'] = orig_height
    crack_slab.info['cell_origin'] = -np.diag(crack_slab.cell)/2.0
    crack_slab.info['strain'] = strain
    crack_slab.info['is_cracked'] = False

    print('Applied initial load: strain=%.4f' % strain)

    # ***** Relaxation of crack slab  *****

    # optionally, relax the slab, keeping top and bottom rows clamped
    if params.relax_slab:
        initial_positions = crack_slab.get_positions()
        crack_slab.set_array('pos_orig', initial_positions)
        # ase.io.write('crack.xyz', crack_slab, format='extxyz')
        crack_slab.write('crack.xyz', format='extxyz')

        for iteration in range(2):
            crack_slab.set_calculator(calc)
            constrained_minim(crack_slab, 
                              method='LBFGS', 
                              vacuum=20.0, 
                              k_spring=params.k_spring, 
                              clamp_mask=crack_slab.get_array('clamp_mask'), 
                              initial_positions=initial_positions,
                              fmax=params.relax_fmax,
                              spring='point')
            # for some reason, do it twice
        
    # ******** Save output file **********

    # Save results in extended XYZ format, including extra properties and info
    print('Writing crack slab to file "crack.xyz"')
    crack_slab.write('crack.xyz', format='extxyz')
