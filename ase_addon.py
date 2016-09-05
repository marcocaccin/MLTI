from __future__ import division

import numpy as np
from ase.constraints import FixConstraint
from ase.utils.geometry import find_mic
from ase.constraints import Hookean

class GentlyDriftBondLength(FixConstraint):
    """Constraint object for fixing a bond length."""
    def __init__(self, a1, a2, bond_speed=0., direction=None):
        """Linearly tweak distance between atoms with indices a1 and a2. If mic is
        True, follows the minimum image convention to keep constant the
        shortest distance between a1 and a2 in any periodic direction.
        atoms only needs to be supplied if mic=True.
        Bond speed is measured in Angstrom per timestep
        """
        self.indices = [a1, a2]
        self.constraint_force = None
        self.bond_speed = bond_speed
        # self.fourindices = None
        if direction is None:
            self.direction = None # standard distance constraint
        elif len(direction) == 2:
            self.adjust = True
            self.direction = 'adjust' # self-adjusting direction of constraint
        else:
            self.direction = np.array(direction) / np.linalg.norm(direction) # fixed direction of constraint

    def adjust_direction(self, atoms):
        if not self.adjust:
            return
        p1, p2 = atoms.positions[self.indices]
        direction, _ = find_mic(np.array([p2 - p1]), atoms._cell, pbc=False)
        direction = direction[0]                                                       
        direction[2] = 0.
        direction /= np.linalg.norm(direction)
        self.direction = direction
        
    def adjust_positions(self, atoms, new):
        p1, p2 = atoms.positions[self.indices]
        d_old, p_old = find_mic(np.array([p2 - p1]), atoms._cell, pbc=False)
        d_old = d_old[0]
        
        q1, q2 = new[self.indices]
        d_new, p_new = find_mic(np.array([q2 - q1]), atoms._cell, pbc=False)
        d_new = d_new[0]
        
        if self.direction is None:
            # amount to put back distance to old value
            delta_d = 0.5 * d_new * (p_old - p_new) / p_new
            # desired increase amount
            sbit = 0.5 * d_new / p_new * self.bond_speed
        else:
            self.adjust_direction(atoms)
            p_old = np.abs(np.dot(d_old, self.direction))
            p_new = np.dot(d_new, self.direction)
            swapsign = np.sign(p_new)
            length_new = np.abs(p_new)
            # amount to put back distance to old value by resizing distance vector d_new
            delta_d = 0.5 * (p_old - p_new) * self.direction * swapsign
            # desired increase amount
            sbit = 0.5 * self.direction * self.bond_speed * swapsign

        new[self.indices] = (q1 - delta_d - sbit, q2 + delta_d + sbit)
        self.distance = p_old + self.bond_speed
        
    def adjust_forces(self, atoms, forces):
        d = np.subtract.reduce(atoms.positions[self.indices])
        d, p = find_mic(np.array([d]), atoms._cell, pbc=False)
        d = d[0]
        if self.direction is None:
            f = 0.5 * d * np.dot(np.subtract.reduce(forces[self.indices]), d) / p**2
        else:
            if self.adjust:
                self.adjust_direction(atoms)
                f = 0.5 * np.dot(np.subtract.reduce(forces[self.indices]), self.direction) * self.direction
            else:
                f = 0.5 * np.dot(np.subtract.reduce(forces[self.indices]), self.direction) * np.sign(np.dot(d, self.direction)) * self.direction
        self.constraint_force = f
        forces[self.indices] += (-f, f)
    
    def index_shuffle(self, atoms, ind):
        """Shuffle the indices of the two atoms in this constraint"""
        newa = [-1, -1]  # Signal error
        for new, old in slice2enlist(ind, len(atoms)):
            for i, a in enumerate(self.indices):
                if old == a:
                    newa[i] = new
        if newa[0] == -1 or newa[1] == -1:
            raise IndexError('Constraint not part of slice')
        self.indices = newa
    
    def get_constraint_force(self, atoms=None):
        """Return the (scalar) force required to maintain the constraint"""
        return self.constraint_force
    
    def __repr__(self):
        return 'FixBondLength(%d, %d)' % tuple(self.indices)
    
    def todict(self):
        return {'name': 'FixBondLength',
                'kwargs': {'a1': self.indices[0], 'a2': self.indices[1]}}


# EDITED VERSION OF Hookean CONSTRAINT
def adjust_potential_energy(self, atoms):
    """Returns the difference to the potential energy due to an active                                                                     
    constraint. (That is, the quantity returned is to be added to the                                                                      
    potential energy.)"""
    positions = atoms.positions
    if self._type == 'plane':
        A, B, C, D = self.plane
        x, y, z = positions[self.index]
        d = ((A * x + B * y + C * z + D) /
             np.sqrt(A**2 + B**2 + C**2))
        return 0.5 * self.spring * d**2

    if self._type == 'two atoms':
        p1, p2 = positions[self.indices]
    elif self._type == 'point':
        p1 = positions[self.index]
        p2 = self.origin
    displace = p2 - p1
    bondlength = np.linalg.norm(displace)
    if bondlength > self.threshold:
        return 0.5 * self.spring * (bondlength - self.threshold)**2
    else:
        return 0.


def adjust_forces(self, atoms, forces):
    positions = atoms.positions
    if self._type == 'plane':
        A, B, C, D = self.plane
        x, y, z = positions[self.index]
        d = ((A * x + B * y + C * z + D) /
             np.sqrt(A**2 + B**2 + C**2))
        magnitude = self.spring * d
        direction = - np.array((A, B, C)) / np.linalg.norm((A, B, C))
        forces[self.index] += direction * magnitude
        return
    if self._type == 'two atoms':
        p1, p2 = positions[self.indices]
    elif self._type == 'point':
        p1 = positions[self.index]
        p2 = self.origin
    displace = p2 - p1
    bondlength = np.linalg.norm(displace)
    if bondlength > self.threshold:
        magnitude = self.spring * (bondlength - self.threshold)
        direction = displace / np.linalg.norm(displace)
        if self._type == 'two atoms':
            forces[self.indices[0]] += direction * magnitude
            forces[self.indices[1]] -= direction * magnitude
        else:
            forces[self.index] += direction * magnitude

setattr(Hookean, 'adjust_potential_energy', adjust_potential_energy)
setattr(Hookean, 'adjust_forces', adjust_forces)


