# MIT License
#
# Copyright (c) 2019 Lucas K Wagner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This file may have been modified by Bytedance Inc. ("Bytedance Modifications").
# All Bytedance Modifications are Copyright 2022 Bytedance Inc.

# This file may have been modified by Shengdu Chai.
# Modifications Copyright (c) 2025 Shengdu Chai

from itertools import starmap

import numpy as np
from pyscf.pbc.gto import Cell as PyscfCell


def get_supercell_kpts(supercell):
    """Gets k-points in supercell that map to primitive cell reciprocal space.

    Args:
        supercell: PySCF Cell object of simulation supercell.

    Returns:
        Array of k-points in Cartesian coordinates (2π/Bohr).

    """
    Sinv = np.linalg.inv(supercell.S).T
    u = [0, 1]
    unit_box = np.stack([x.ravel() for x in np.meshgrid(*[u] * 3, indexing="ij")]).T
    unit_box_ = np.dot(unit_box, supercell.S.T)
    xyz_range = np.stack([f(unit_box_, axis=0) for f in (np.amin, np.amax)]).T
    kptmesh = np.meshgrid(*list(starmap(np.arange, xyz_range)), indexing="ij")
    possible_kpts = np.dot(np.stack([x.ravel() for x in kptmesh]).T, Sinv)
    in_unit_box = (possible_kpts >= 0) * (possible_kpts < 1 - 1e-12)
    select = np.where(np.all(in_unit_box, axis=1))[0]
    reclatvec = np.linalg.inv(supercell.original_cell.lattice_vectors()).T * 2 * np.pi
    return np.dot(possible_kpts[select], reclatvec)


def get_supercell_copies(latvec, S):
    """Get atomic positions in supercell from lattice vectors and supercell matrix.

    Args:
        latvec: Lattice vectors of the primitive cell
        S: Supercell transformation matrix

    Returns:
        Array of atomic positions in the supercell
    """
    Sinv = np.linalg.inv(S).T
    u = [0, 1]
    unit_box = np.stack([x.ravel() for x in np.meshgrid(*[u] * 3, indexing="ij")]).T
    unit_box_ = np.dot(unit_box, S)
    xyz_range = np.stack([f(unit_box_, axis=0) for f in (np.amin, np.amax)]).T
    mesh = np.meshgrid(*list(starmap(np.arange, xyz_range)), indexing="ij")
    possible_pts = np.dot(np.stack([x.ravel() for x in mesh]).T, Sinv.T)
    in_unit_box = (possible_pts >= 0) * (possible_pts < 1 - 1e-12)
    select = np.where(np.all(in_unit_box, axis=1))[0]
    return np.linalg.multi_dot((possible_pts[select], S, latvec))


def get_supercell(cell, S, sym_type="minimal") -> PyscfCell:
    """Generates supercell from primitive cell.

    Args:
        cell: PySCF Cell object representing primitive cell.
        S: Shape (3, 3) supercell transformation matrix.
        sym_type: Symmetry type ('minimal', 'fcc', 'bcc', 'hexagonal').

    Returns:
        PySCF Cell object for QMC simulation supercell.

    """
    import pyscf.pbc

    scale = np.abs(int(np.round(np.linalg.det(S))))
    superlattice = np.dot(S, cell.lattice_vectors())
    Rpts = get_supercell_copies(cell.lattice_vectors(), S)
    atom = []
    for name, xyz in cell._atom:
        atom.extend([(name, xyz + R) for R in Rpts])
    supercell = pyscf.pbc.gto.Cell()
    supercell.a = np.asarray(superlattice, dtype=cell.a.dtype)
    supercell.atom = atom
    supercell.ecp = cell.ecp
    supercell.basis = cell.basis
    supercell.exp_to_discard = cell.exp_to_discard
    supercell.unit = "Bohr"
    supercell.spin = cell.spin * scale
    supercell.build()
    supercell.original_cell = cell
    supercell.S = S
    supercell.scale = scale
    supercell.output = None
    supercell.stdout = None
    supercell.space_group_symmetry = True
    supercell = set_symmetry_lat(supercell, sym_type)
    return supercell


def set_symmetry_lat(supercell, sym_type="minimal"):
    """Attaches symmetry-adapted lattice vectors to simulation cell.

    Args:
        supercell: PySCF supercell object.
        sym_type: Crystal symmetry type ('minimal', 'fcc', 'bcc', 'hexagonal').

    Returns:
        Supercell with BV and AV attributes set according to symmetry.

    """
    prim_bv = supercell.original_cell.reciprocal_vectors()
    sim_bv = supercell.reciprocal_vectors()
    if sym_type == "minimal":
        mat = np.eye(3)
    elif sym_type == "fcc":
        mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    elif sym_type == "bcc":
        mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, -1, 0], [1, 0, -1], [0, 1, -1]])
    elif sym_type == "hexagonal":
        mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, 0]])
    else:
        mat = np.eye(3)

    prim_bv = mat @ prim_bv
    sim_bv = mat @ sim_bv

    prim_av = np.linalg.pinv(prim_bv).T
    sim_av = np.linalg.pinv(sim_bv).T
    supercell.BV = np.asarray(sim_bv, dtype=supercell.a.dtype)
    supercell.AV = np.asarray(sim_av, dtype=supercell.a.dtype)
    supercell.original_cell.BV = np.asarray(prim_bv, dtype=supercell.a.dtype)
    supercell.original_cell.AV = np.asarray(prim_av, dtype=supercell.a.dtype)
    return supercell


def get_k_indices(cell, mf, kpts, tol=1e-6):
    """Given a list of kpts, return inds such that mf.kpts[inds] is a list of kpts equivalent to the input list."""
    kdiffs = mf.kpts[None] - kpts[:, None]
    frac_kdiffs = np.dot(kdiffs, cell.lattice_vectors().T) / (2 * np.pi)
    kdiffs = np.mod(frac_kdiffs + 0.5, 1) - 0.5
    return np.nonzero(np.linalg.norm(kdiffs, axis=-1) < tol)[1]


def convert_simulation_cell_to_unit_cell(cell):
    """Convert simulation cell to unit cell with fractional coordinates.

    Args:
        cell: PySCF cell object representing the simulation cell

    Returns:
        PySCF cell object with unit lattice vectors and fractional coordinates
    """
    import pyscf.pbc

    inv_lattice = np.linalg.inv(np.asarray(cell.lattice_vectors()))
    atom = []
    for name, xyz in cell._atom:
        atom.extend([(name, np.dot(xyz, inv_lattice))])
    unit_cell = pyscf.pbc.gto.Cell()
    unit_cell.a = np.eye(3, dtype=cell.a.dtype)
    unit_cell.atom = atom
    unit_cell.ecp = cell.ecp
    unit_cell.basis = cell.basis
    unit_cell.exp_to_discard = cell.exp_to_discard
    unit_cell.unit = "Bohr"
    unit_cell.spin = cell.spin
    unit_cell.space_group_symmetry = True
    unit_cell.verbose = 0
    unit_cell.build()
    return unit_cell


def get_klist(cell, twist):
    """Get k-point list for cell with twist boundary conditions.

    Args:
        cell: PySCF cell object
        twist: Twist vector for boundary conditions

    Returns:
        List of k-points for each spin channel
    """
    kpts = get_supercell_kpts(cell)
    kpts = kpts + np.dot(np.linalg.inv(cell.a), np.mod(twist, 1.0)) * 2 * np.pi
    spins = cell.nelec
    klist = [np.tile(kpts, (spin, 1)).astype(cell.a.dtype) for spin in spins]
    return klist
