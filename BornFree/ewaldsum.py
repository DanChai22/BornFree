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

# This file may have been modified by Bytedance Inc. (“Bytedance Modifications”).
# All Bytedance Modifications are Copyright 2022 Bytedance Inc.

# This file may have been modified by Shengdu Chai.
# Modifications Copyright (c) 2025 Shengdu Chai

import logging
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from pyscf.pbc.gto import Cell as PyscfCell

from BornFree import distance
from BornFree.base_config import CrystalLatticeConfig
from BornFree.network import network_block

logger = logging.getLogger(__name__)


def get_lattice_displacements(latvec: Array) -> Array:
    """Generates lattice displacement vectors for real-space Ewald summation.

    Args:
        latvec: Lattice vectors matrix of shape (3, 3) in Bohr.

    Returns:
        Displacement vectors of shape (27, 3) in Bohr.

    """
    u = jnp.asarray([-1, 0, 1])
    unit_box = jnp.stack([x.ravel() for x in jnp.meshgrid(*[u] * 3, indexing="ij")]).T
    unit_box = unit_box + 1e-10 * jax.random.uniform(jax.random.PRNGKey(0), shape=unit_box.shape)
    return jnp.asarray(jnp.dot(unit_box, latvec))


def calculate_ion_ion_real(
    atoms_pos: Array, atom_charges: Array, dist: Any, alpha: float, displacements: Array
) -> float:
    """Calculate the real-space part of the ion-ion interaction.

    Args:
        atoms_pos: Atomic positions of shape (natom*3,) in Bohr.
        atom_charges: Atomic charges of shape (natom,).
        dist: Distance calculator for periodic boundary conditions.
        alpha: Ewald parameter.
        displacements: Lattice displacement vectors of shape (n_cells, 3) in Bohr.

    Returns:
        Ion-ion interaction energy in Hartree.

    """
    # Real space part
    if len(atom_charges) == 1:
        ii_real = 0
    else:
        ion_distances = dist.dist_matrix(atoms_pos)
        rvec = ion_distances[None, :, :, :] + displacements[:, None, None, :]  # rvec shape (n unit cell, nion, nion, 3)
        r = jnp.linalg.norm(rvec, axis=-1)  # r shape (n unit cell, nion, nion)
        charge_ij = atom_charges[..., None] * atom_charges[None, ...]  # charge_ij shape (nion, nion)
        ii_real = jnp.sum(jnp.triu(charge_ij * jax.lax.erfc(alpha * r) / r, k=1))
    return ii_real


def calculate_ion_electron_real(
    atoms_pos: Array,
    elec_pos: Array,
    atom_charges: Array,
    dist: Any,
    alpha: float,
    displacements: Array,
) -> float:
    """Calculate the real-space part of the ion-electron interaction.

    Args:
        atoms_pos: Atomic positions of shape (natom*3,) in Bohr.
        elec_pos: Electron positions of shape (nelec*3,) in Bohr.
        atom_charges: Atomic charges of shape (natom,).
        dist: Distance calculator for periodic boundary conditions.
        alpha: Ewald parameter.
        displacements: Lattice displacement vectors of shape (n_cells, 3) in Bohr.

    Returns:
        Ion-electron interaction energy in Hartree.

    """
    ei_distances = dist.dist_i(atoms_pos, elec_pos)
    r = ei_distances[:, :, None, :] + displacements
    r = jnp.linalg.norm(r, axis=-1)
    ei_cij = jnp.sum(jax.lax.erfc(alpha * r) / r, axis=-1)
    ei_real_separated = jnp.sum(-atom_charges[None, :] * ei_cij)
    return ei_real_separated


def calculate_electron_electron_real(
    elec_pos: Array, dist: Any, alpha: float, nelec: int, displacements: Array
) -> float:
    """Calculate the real-space part of the electron-electron interaction.

    Args:
        elec_pos: Electron positions of shape (nelec*3,) in Bohr.
        dist: Distance calculator for periodic boundary conditions.
        alpha: Ewald parameter.
        nelec: Number of electrons.
        displacements: Lattice displacement vectors of shape (n_cells, 3) in Bohr.

    Returns:
        Electron-electron interaction energy in Hartree.

    """
    # Real space electron-electron part
    ee_real_separated = jnp.array(0.0)
    if nelec > 1:
        ee_distances = dist.dist_matrix(elec_pos)
        rvec = (
            ee_distances[None, :, :, :] + displacements[:, None, None, :]
        )  # rvec shape (n unit cell, nelec, nelec, 3)
        r = jnp.linalg.norm(rvec, axis=-1)
        ee_real_separated = jnp.sum(jnp.triu(jax.lax.erfc(alpha * r) / r, k=1))

    return ee_real_separated


def calculate_reciprocal(
    elec_pos: Array,
    atom_pos: Array,
    atom_charges: Array,
    natom: int,
    nelec: int,
    gpoints: Array,
    gweight: Array,
    omega: float,
) -> tuple[float, float, float]:
    """Calculate reciprocal-space components of Coulomb interactions.

    Args:
        elec_pos: Electron positions of shape (nelec*3,) in Bohr.
        atom_pos: Atomic positions of shape (natom*3,) in Bohr.
        atom_charges: Atomic charges of shape (natom,).
        natom: Number of atoms.
        nelec: Number of electrons.
        gpoints: Reciprocal lattice vectors of shape (n_gpoints, 3).
        gweight: Weight factors of shape (n_gpoints,).
        omega: Cell volume.

    Returns:
        Tuple of (ee_recip, ei_recip, ii_recip) energies in Hartree.

    """
    # Reciprocal space electron-electron part
    gweight = gweight / omega
    e_GdotR = jnp.matmul(elec_pos.reshape(nelec, -1), jnp.transpose(gpoints))
    sum_e_sin = jnp.sin(e_GdotR).sum(axis=0)
    sum_e_cos = jnp.cos(e_GdotR).sum(axis=0)
    ee_recip = jnp.dot(sum_e_sin**2 + sum_e_cos**2, gweight)
    # Reciprocal space electron-ion part
    GdotR = jnp.dot(gpoints, jnp.asarray(atom_pos.reshape(natom, -1).T))
    ion_exp = jnp.dot(jnp.exp(1j * GdotR), atom_charges)
    coscos_sinsin = -ion_exp.real * sum_e_cos - ion_exp.imag * sum_e_sin
    ei_recip = 2 * jnp.dot(coscos_sinsin, gweight)
    ii_recip = jnp.dot(gweight, jnp.abs(ion_exp) ** 2)
    return ee_recip, ei_recip, ii_recip


def calculate_constants(atom_charges: Array, alpha: float, omega: float, nelec: int) -> tuple[float, float, float]:
    """Calculate constant correction terms for Ewald summation.

    Args:
        atom_charges: Atomic charges of shape (natom,).
        alpha: Ewald parameter.
        omega: Cell volume.
        nelec: Number of electrons.

    Returns:
        Tuple of (ii_const, ei_const, ee_const) in Hartree.

    """
    charge_sum = jnp.sum(atom_charges)
    charge_sum2 = jnp.sum(atom_charges**2)

    ijconst = -jnp.pi / (2 * omega * alpha**2)
    squareconst = -alpha / jnp.sqrt(jnp.pi)

    ii_const = charge_sum**2 * ijconst + charge_sum2 * squareconst
    ei_const = -nelec * charge_sum * ijconst * 2
    ee_const = nelec**2 * ijconst + nelec * squareconst

    return ii_const, ei_const, ee_const


def select_big(gpts: Array, recvec: Array, alpha: float) -> tuple[Array, Array]:
    """Filter significant G-points with weight > 1e-12.

    Args:
        gpts: Integer grid points of shape (n_gpts, 3).
        recvec: Reciprocal lattice vectors of shape (3, 3).
        alpha: Ewald parameter.

    Returns:
        Tuple of (gpoints, gweight) after filtering.

    """
    gpts = jnp.array(gpts)
    gpoints = 2 * jnp.pi * jnp.matmul(recvec.T, gpts.reshape(gpts.shape[0], -1)).T
    gsquared = jnp.sum(gpoints * gpoints, axis=-1)
    gweight = 4 * jnp.pi * jnp.exp(-gsquared / (4 * alpha**2)) / gsquared
    bigweight = gweight > 1e-12
    return gpoints[bigweight], gweight[bigweight]


def calculate_gpoints_and_weights(gpts: Array, recvec: Array, alpha: float) -> tuple[Array, Array]:
    """Calculate G-points and weights without filtering.

    Args:
        gpts: Integer grid points of shape (n_gpts, 3).
        recvec: Reciprocal lattice vectors of shape (3, 3).
        alpha: Ewald parameter.

    Returns:
        Tuple of (gpoints, gweight).

    """
    gpts = jnp.array(gpts)
    gpoints = jnp.matmul(recvec.T, gpts.reshape(gpts.shape[0], -1)).T * 2 * jnp.pi
    gsquared = jnp.sum(gpoints * gpoints, axis=-1)
    gweight = 4 * jnp.pi * jnp.exp(-gsquared / (4 * alpha**2)) / gsquared
    return gpoints, gweight


class EwaldSum_nvt_fixed:
    """Ewald summation for NVT ensemble with fixed lattice.

    Attributes:
        nelec: Number of electrons.
        atom_coords: Atomic positions (natom*3,) in Bohr.
        atom_charges: Atomic charges (natom,).
        natom: Number of atoms.
        latvec: Lattice vectors (3, 3) in Bohr.
        dist: Distance calculator.
        omega: Unit cell volume in Bohr³.
        displacements: Lattice displacements for real-space sum.
        alpha: Ewald parameter.
        gpoints: Reciprocal lattice vectors.
        gweight: G-point weights.

    """

    def __init__(self, cell: PyscfCell, ewald_gmax: int = 25, nlatvec: int = 1):
        """Initialize Ewald summation for NVT ensemble.

        Args:
            cell: PySCF Cell object.
            ewald_gmax: Maximum G-point index.
            nlatvec: Real-space neighbor range.

        """
        self.nelec = sum(cell.nelec)
        self.atom_coords = cell.atom_coords().ravel()
        self.atom_charges = cell.atom_charges()
        self.natom = cell.natm
        self.latvec = cell.lattice_vectors()
        self.dist = distance.MinimalImageDistance(self.latvec)
        self.omega = jnp.linalg.det(self.latvec)
        self.set_lattice_displacements(nlatvec)
        self.set_up_reciprocal_ewald_sum(ewald_gmax)

    def set_lattice_displacements(self, nlatvec: int):
        """Generate lattice displacement vectors for real-space summation.

        Args:
            nlatvec: Neighbor range in each direction.

        Sets:
            self.displacements: Shape ((2*nlatvec+1)³, 3).

        """
        XYZ = jnp.meshgrid(*[jnp.arange(-nlatvec, nlatvec + 1)] * 3, indexing="ij")
        xyz = jnp.stack(XYZ, axis=-1).reshape((-1, 3))
        self.displacements = jnp.asarray(jnp.dot(xyz, self.latvec))

    def set_up_reciprocal_ewald_sum(self, ewald_gmax: int):
        """Set up reciprocal-space grid and determine alpha parameter.

        Args:
            ewald_gmax: Maximum G-point index.

        Sets:
            self.alpha: Ewald parameter.
            self.gpoints: Filtered G-point vectors.
            self.gweight: G-point weights.

        """
        recvec = jnp.linalg.inv(self.latvec).T

        # Determine alpha
        smallestheight = jnp.amin(1 / jnp.linalg.norm(recvec, axis=1))
        self.alpha = 5.0 / smallestheight
        logger.info("Setting Ewald alpha to %s", self.alpha.item())

        # Determine G points to include in reciprocal Ewald sum
        gptsXpos = jnp.meshgrid(
            jnp.arange(1, ewald_gmax + 1),
            *[jnp.arange(-ewald_gmax, ewald_gmax + 1)] * 2,
            indexing="ij",
        )
        zero = jnp.asarray([0])
        gptsX0Ypos = jnp.meshgrid(
            zero,
            jnp.arange(1, ewald_gmax + 1),
            jnp.arange(-ewald_gmax, ewald_gmax + 1),
            indexing="ij",
        )
        gptsX0Y0Zpos = jnp.meshgrid(zero, zero, jnp.arange(1, ewald_gmax + 1), indexing="ij")
        gs = zip(*[select_big(x, recvec, self.alpha) for x in (gptsXpos, gptsX0Ypos, gptsX0Y0Zpos)])
        self.gpoints, self.gweight = [jnp.concatenate(x, axis=0) for x in gs]

    def energy(self, configs: Array) -> tuple[float, float, float]:
        """Calculate Coulomb interaction energies.

        Args:
            configs: Electron positions of shape (nelec*3,).

        Returns:
            Tuple of (ee, ei, ii) energies in Hartree.

        """
        ee_real = calculate_electron_electron_real(configs, self.dist, self.alpha, self.nelec, self.displacements)
        ei_real = calculate_ion_electron_real(
            self.atom_coords,
            configs,
            self.atom_charges,
            self.dist,
            self.alpha,
            self.displacements,
        )
        ii_real = calculate_ion_ion_real(
            self.atom_coords,
            self.atom_charges,
            self.dist,
            self.alpha,
            self.displacements,
        )
        ee_recip, ei_recip, ii_recip = calculate_reciprocal(
            configs,
            self.atom_coords,
            self.atom_charges,
            self.natom,
            self.nelec,
            self.gpoints,
            self.gweight,
            self.omega,
        )
        ii_const, ei_const, ee_const = calculate_constants(self.atom_charges, self.alpha, self.omega, self.nelec)
        ee = ee_real + ee_recip + ee_const
        ei = ei_real + ei_recip + ei_const
        ii = ii_real + ii_recip + ii_const
        return ee, ei, ii


class EwaldSum_nvt_quantum(EwaldSum_nvt_fixed):
    """Ewald summation for NVT ensemble with quantum nuclei and electrons."""

    def __init__(self, cell: PyscfCell, ewald_gmax: int = 25, nlatvec: int = 1):
        """Initialize Ewald summation for quantum NVT ensemble.

        Args:
            cell: PySCF Cell object.
            ewald_gmax: Maximum G-point index.
            nlatvec: Real-space neighbor range.

        """
        super().__init__(cell, ewald_gmax, nlatvec)

    def energy(self, configs: Array) -> tuple[float, float, float]:
        """Calculate Coulomb energies with quantum nuclei and electrons.

        Args:
            configs: Combined ion and electron positions of shape
                ((natom+nelec)*3,). Format: [ion_pos | elec_pos].

        Returns:
            Tuple of (ee, ei, ii) energies in Hartree.

        """
        elec_pos = configs[self.natom * 3 :]
        atom_pos = configs[: self.natom * 3]
        ee_real = calculate_electron_electron_real(elec_pos, self.dist, self.alpha, self.nelec, self.displacements)
        ei_real = calculate_ion_electron_real(
            atom_pos,
            elec_pos,
            self.atom_charges,
            self.dist,
            self.alpha,
            self.displacements,
        )
        ii_real = calculate_ion_ion_real(atom_pos, self.atom_charges, self.dist, self.alpha, self.displacements)
        ee_recip, ei_recip, ii_recip = calculate_reciprocal(
            elec_pos,
            atom_pos,
            self.atom_charges,
            self.natom,
            self.nelec,
            self.gpoints,
            self.gweight,
            self.omega,
        )
        ii_const, ei_const, ee_const = calculate_constants(self.atom_charges, self.alpha, self.omega, self.nelec)
        ee = ee_real + ee_recip + ee_const
        ei = ei_real + ei_recip + ei_const
        ii = ii_real + ii_recip + ii_const
        return ee, ei, ii


class EwaldSum_npt_quantum(EwaldSum_nvt_quantum):
    """Ewald summation for NPT ensemble with variable lattice.

    Attributes:
        lattice_config: Lattice configuration.

    """

    def __init__(
        self,
        cell: PyscfCell,
        lattice_config: CrystalLatticeConfig,
        ewald_gmax: int = 25,
        nlatvec: int = 1,
    ):
        """Initialize Ewald summation for NPT ensemble.

        Args:
            cell: PySCF Cell object.
            lattice_config: Lattice configuration.
            ewald_gmax: Maximum G-point index.
            nlatvec: Unused, kept for compatibility.

        """
        super().__init__(cell, ewald_gmax, nlatvec)
        self.lattice_config = lattice_config

    def set_up_reciprocal_ewald_sum(self, ewald_gmax: int):
        """Store G-point grid as integer indices for dynamic lattice.

        Args:
            ewald_gmax: Maximum G-point index.

        Sets:
            self.gptsXpos, self.gptsX0Ypos, self.gptsX0Y0Zpos: G-point grids.

        """
        # Determine G points to include in reciprocal Ewald sum
        self.gptsXpos = jnp.meshgrid(
            jnp.arange(1, ewald_gmax + 1),
            *[jnp.arange(-ewald_gmax, ewald_gmax + 1)] * 2,
            indexing="ij",
        )
        zero = jnp.asarray([0])
        self.gptsX0Ypos = jnp.meshgrid(
            zero,
            jnp.arange(1, ewald_gmax + 1),
            jnp.arange(-ewald_gmax, ewald_gmax + 1),
            indexing="ij",
        )
        self.gptsX0Y0Zpos = jnp.meshgrid(zero, zero, jnp.arange(1, ewald_gmax + 1), indexing="ij")

    def get_gpts_and_weights(self, alpha: float, recvec: Array) -> tuple[Array, Array]:
        """Convert G-point indices to vectors and compute weights.

        Args:
            alpha: Ewald parameter.
            recvec: Reciprocal lattice vectors of shape (3, 3).

        Returns:
            Tuple of (gpoints, gweight).

        """
        gs = zip(*[
            calculate_gpoints_and_weights(x, recvec, alpha) for x in (self.gptsXpos, self.gptsX0Ypos, self.gptsX0Y0Zpos)
        ])
        return [jnp.concatenate(x, axis=0) for x in gs]

    def energy(self, cellpar: Array, configs: Array) -> tuple[Array, Array, Array]:
        """Calculate Coulomb energies with dynamic lattice.

        Args:
            cellpar: Lattice parameters of shape (n_params,).
            configs: Ion and electron positions of shape ((natom+nelec)*3,).

        Returns:
            Tuple of (ee, ei, ii) energies in Hartree.

        """
        # configs shape [(Natom+Nelec)*3]
        latvec = network_block.get_jacobian(cellpar, self.lattice_config)  # shape (3,3)
        dist = distance.MinimalImageDistance_Dynamic(latvec)
        displacements = get_lattice_displacements(latvec)
        omega = jnp.linalg.det(latvec)
        recvec = jnp.linalg.inv(latvec).T  # 2 pi smaller compared to the pyscf.reciprocal_vectors()
        smallestheight = jnp.amin(1 / jnp.linalg.norm(recvec, axis=1))
        alpha = 5.0 / smallestheight
        gpoints, gweight = self.get_gpts_and_weights(alpha, recvec)
        elec_pos = configs[self.natom * 3 :]
        atom_pos = configs[: self.natom * 3]
        ee_real = calculate_electron_electron_real(elec_pos, dist, alpha, self.nelec, displacements)
        ei_real = calculate_ion_electron_real(atom_pos, elec_pos, self.atom_charges, dist, alpha, displacements)
        ii_real = calculate_ion_ion_real(atom_pos, self.atom_charges, dist, alpha, displacements)
        ee_recip, ei_recip, ii_recip = calculate_reciprocal(
            elec_pos,
            atom_pos,
            self.atom_charges,
            self.natom,
            self.nelec,
            gpoints,
            gweight,
            omega,
        )
        ii_const, ei_const, ee_const = calculate_constants(self.atom_charges, alpha, omega, self.nelec)
        ee = ee_real + ee_recip + ee_const
        ei = ei_real + ei_recip + ei_const
        ii = ii_real + ii_recip + ii_const
        return (
            ee.astype(configs.dtype),
            ei.astype(configs.dtype),
            ii.astype(configs.dtype),
        )
