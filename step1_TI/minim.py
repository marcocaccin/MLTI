import os
import sys

import numpy as np

from ase.calculators.calculator import Calculator
from ase.optimize.precon import LBFGS
from ase.optimize import FIRE
from ase.optimize import LBFGS as vanillaLBFGS

from ase.constraints import FixedLine, FixAtoms

from quippy import Atoms, Potential
from ase_addon import Hookean


class SumCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, calc1, calc2, *args, **kwargs):
        self.calc1 = calc1
        self.calc2 = calc2
        Calculator.__init__(self, *args, **kwargs)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        if system_changes:
            self.results = {}
        if 'energy' in properties and 'energy' not in self.results:
            e1 = self.calc1.get_potential_energy(atoms)
            e2 = self.calc2.get_potential_energy(atoms)
            self.results['energy'] = e1 + e2
        if 'forces' in properties and 'forces' not in self.results:
            f1 = self.calc1.get_forces(atoms)
            f2 = self.calc2.get_forces(atoms)
            self.results['forces'] = f1 + f2

class ConstraintCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, constraint, *args, **kwargs):
        self.constraint = constraint
        Calculator.__init__(self, *args, **kwargs)

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        if system_changes:
            self.results = {}
        if 'energy' in properties and 'energy' not in self.results:
            self.results['energy'] = 0.0
            for constraint in self.constraint:
                self.results['energy'] += constraint.adjust_potential_energy(atoms)
        if 'forces' in properties and 'forces' not in self.results:
            self.results['forces'] = np.zeros((len(atoms), 3))
            for constraint in self.constraint:
                constraint.adjust_forces(atoms, self.results['forces'])


def constrained_minim(atoms, method='LBFGS', vacuum=20.0, k_spring=0.6, clamp_mask=None, initial_positions=None, fmax=0.05, spring='point'):
    c = np.ptp(atoms.positions, axis=0)
    c += vacuum
    atoms.set_cell(np.diag(c))

    if clamp_mask is None:
        at_help = Atoms('crack.xyz')
        clamp_mask = at_help.get_array('clamp_mask')
    if initial_positions is None:
        at_help = Atoms('crack.xyz')
        at_help.get_array('pos_orig')
    
    if spring == 'plane':
        springs = [Hookean(a1=i,
                           a2=[0, -1, 0, initial_positions[i,1]],
                           k=k_spring, rt=0)
                   for i in np.where(clamp_mask)[0]]
    else:
        springs = [Hookean(a1=i,
                           a2=initial_positions[i],
                           k=k_spring, rt=0)
                   for i in np.where(clamp_mask)[0]]
        
    from quippy.io import AtomsWriter
    out = AtomsWriter("traj-relax_structure.xyz")
    trajectory_write = lambda: out.write(atoms)

    if method == "LBFGS":
        pot = atoms.calc
        cc = ConstraintCalculator(springs)
        sc = SumCalculator(pot, cc)

        left, bottom, _ = initial_positions.min(axis=0)
        fix_line_mask = (initial_positions[:,0] < left + 6.5)
        fix_line = FixAtoms(mask = fix_line_mask)
        atoms.set_array('fix_line_mask', fix_line_mask)
        atoms.set_constraint(fix_line)
        atoms.set_calculator(sc)
        opt = LBFGS(atoms, use_armijo=False)
        
    elif method == "FIRE":
        right, top, _ = initial_positions.max(axis=0)
        left, bottom, _ = initial_positions.min(axis=0)
        fix_line_mask = np.logical_or(initial_positions[:,0] > right - 6.5, initial_positions[:,0] < left + 6.5)
        fix_line_idx = np.where(fix_line_mask)[0]
        fix_line = [FixedLine(i, np.array([0,1,0])) for i in fix_line_idx]
        atoms.set_array('fix_line_mask', fix_line_mask)
        
        atoms.set_constraint(fix_line + springs)
    
        opt = FIRE(atoms, dtmax=2.5, maxmove=0.5)

    elif method == "mix":
        # do a first opt by keeping vertical edges fixed, then make them move along y
        right, top, _ = initial_positions.max(axis=0)
        left, bottom, _ = initial_positions.min(axis=0)
        fix_line_mask = np.logical_or(initial_positions[:,0] > right - 6.5, initial_positions[:,0] < left + 6.5)
        fix_line_idx = np.where(fix_line_mask)[0]
        
        pot = atoms.calc
        cc = ConstraintCalculator(springs)
        sc = SumCalculator(pot, cc)

        atoms.set_constraint(FixAtoms(mask=fix_line_mask))
        atoms.set_calculator(sc)
        LBFGS(atoms, use_armijo=False).run(fmax=0.2)

        
        fix_line = [FixedLine(i, np.array([0,1,0])) for i in fix_line_idx]
        atoms.set_array('fix_line_mask', fix_line_mask)
        
        atoms.set_constraint(fix_line + springs)
        atoms.set_calculator(pot)
        opt = vanillaLBFGS(atoms)
        
    else:
        print("Method not understood")
        return

    opt.attach(trajectory_write, interval=5)
    opt.run(fmax=fmax)
    return

if __name__ == '__main__':
    
    pot = Potential('IP TS', param_filename="../ts_params.xml")
    in_file = sys.argv[1]
    out_file = os.path.splitext(in_file)[0] + '_relaxed.xyz'
    atoms = Atoms(in_file)
    minim_precon(atoms)
    atoms.write(out_file)
