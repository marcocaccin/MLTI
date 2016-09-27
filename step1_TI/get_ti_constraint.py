#!/usr/bin/env python
from __future__ import print_function

import sys
import os
import numpy as np
from quippy import AtomsReader
from ase.utils.geometry import find_mic
from ase_addon import SlowGrowthBondLength


def calculate_constraint(at, tipatoms):
    f = at.get_array('force')
    cnstr = SlowGrowthBondLength(*tipatoms, bond_speed=0., direction=tipatoms)
    cnstr.adjust_forces(at, f)
    
    p1, p2 = at.positions[tipatoms]
    d, _ = find_mic(np.array([p2 - p1]), at._cell, pbc=False)
    d = d[0]
    d = np.dot(d, cnstr.direction)
    f_cnstr = np.dot(cnstr.get_constraint_force(), cnstr.direction)
    return np.abs(d), f_cnstr


if __name__ == "__main__":


    # MAIN
    input_file = sys.argv[1]
    atlist = AtomsReader(input_file)

    tipatoms = [int(sys.argv[2]), int(sys.argv[3])]
    
    name = input_file.strip('.xyz')
    fff = open('%s-constraint.csv' % name, 'wb')
    for i, at in enumerate(atlist):
        d, f_cnstr = calculate_constraint(at, tipatoms)
        print("%.5f, %.5f" % (np.abs(d), f_cnstr), file=fff)
        print("Step: %06d | %.5f, %.5f" % (i, np.abs(d), f_cnstr))
    fff.close()
