# Copyright (c) 2025 Shengdu Chai
#
# Licensed under the Apache License, Version 2.0.

"""Estimators for computing material properties like RDF and XRD patterns.

This module provides classes for calculating radial distribution functions (RDF)
and X-ray diffraction (XRD) patterns from atomic configurations.
"""

import jax
import jax.numpy as jnp
import numpy as np
from ase import Atoms
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.ase import AseAtomsAdaptor

from BornFree.network import network_block
from BornFree.utils.units import bohr2angstrom

vmap_enforce_pbc = jax.vmap(network_block.enforce_pbc, in_axes=(None, 0), out_axes=0)


class get_rdf:
    """Radial distribution function calculator with periodic boundary conditions.

    Args:
        n_bins: Number of histogram bins for RDF calculation.

    """

    def __init__(self, n_bins):
        """Initialize RDF calculator.

        Args:
            n_bins: Number of bins for the radial distribution function
        """
        self.n_bins = n_bins

    def find_nearest_distance(self, rij, Ls):
        """Finds nearest distance between particles under PBC.

        Args:
            rij: Shape (3,) displacement vector.
            Ls: Shape (3, 3) lattice vectors.

        Returns:
            Nearest distance considering periodic images.

        """
        inv_Ls = jnp.linalg.inv(Ls)
        rij = rij - jnp.dot(rij, inv_Ls).round() @ Ls
        dij = jnp.linalg.norm(rij)
        return dij

    def rdf(self, x, y, Ls):
        """Computes radial distribution function for particle pairs.

        Args:
            x: Shape (N, 3) particle positions.
            y: Shape (N, 3) particle positions.
            Ls: Shape (3, 3) lattice vectors.

        Returns:
            Tuple of (r, gr) where r is bin centers and gr is normalized RDF.

        """
        assert x.shape == y.shape
        n = x.shape[0]
        L = jnp.linalg.norm(Ls)
        v = jnp.linalg.det(Ls)

        i, j = jnp.triu_indices(n, k=1)
        rij = (x[:, None, :] - y[None, :, :])[i, j]
        vmap_find_nearest_distance = jax.vmap(
            self.find_nearest_distance, in_axes=(0, None), out_axes=0
        )
        dij = vmap_find_nearest_distance(rij, Ls)  # (n*(n-1)/2)

        hist, bin_edges = jnp.histogram(
            dij.reshape(-1), range=[0, 0.9 * L], bins=self.n_bins
        )
        r = (bin_edges[:-1] + bin_edges[1:]) / 2
        dr = bin_edges[1] - bin_edges[0]
        gr = hist / dij.size / (4 * jnp.pi * r**2 * dr / v)

        return r, gr


class get_xrd:
    """X-ray diffraction pattern calculator.

    Args:
        wavelength: X-ray wavelength in Angstroms.
        two_theta_range: Tuple of (min, max) 2θ angles in degrees.
        step: Step size for 2θ sampling.
        gamma: Lorentzian broadening parameter.

    """

    def __init__(self, wavelength=0.2, two_theta_range=(5, 20), step=0.01, gamma=0.01):
        """Initialize XRD calculator.

        Args:
            wavelength: X-ray wavelength in angstroms
            two_theta_range: Range of 2θ angles in degrees
            step: Step size for 2θ angles
            gamma: Lorentzian broadening parameter
        """
        self.wavelength = wavelength
        self.two_theta_range = two_theta_range
        self.step = step
        self.gamma = gamma
        self.xrd_calculator = XRDCalculator(
            wavelength=wavelength, debye_waller_factors=None, symprec=0
        )

    def load_atoms(self, xp, Ls, i):
        """Loads atomic configuration into ASE Atoms object.

        Args:
            xp: Shape (batch_size, natm, 3) positions in Bohr.
            Ls: Lattice vectors in Bohr.
            i: Configuration index to extract.

        Returns:
            ASE Atoms object with PBC applied.

        """
        xp = vmap_enforce_pbc(Ls, xp)[0]
        xp = bohr2angstrom(xp)
        Ls = bohr2angstrom(Ls)
        # create Atoms object by the shape of xp
        atoms = Atoms("H" + str(xp.shape[1]), positions=xp[i], cell=Ls, pbc=[1, 1, 1])
        return atoms

    def lorentz(self, E, E0, gamma):
        """Lorentzian broadening function."""
        return (1 / jnp.pi) * (gamma / ((E - E0) ** 2 + gamma**2))

    def compute_average_xrd_pymatgen(self, xp, Ls, n_samples=100):
        """Computes ensemble-averaged XRD pattern using Pymatgen.

        Args:
            xp: Shape (batch_size, natm, 3) atomic positions.
            Ls: Lattice vectors.
            n_samples: Number of configurations to average.

        Returns:
            Tuple of (two_theta, averaged_intensity).

        """
        two_theta_values = jnp.arange(
            self.two_theta_range[0], self.two_theta_range[1], self.step
        )
        total_intensity = jnp.zeros_like(two_theta_values)

        for i in range(n_samples):
            # Convert ASE Atoms object to Pymatgen structure
            atoms = self.load_atoms(xp, Ls, i)
            atoms.symbols = ["H"] * len(atoms)
            structure = AseAtomsAdaptor.get_structure(atoms.repeat((1, 1, 1)))
            # Compute XRD pattern
            xrd_pattern = self.xrd_calculator.get_pattern(
                  structure, two_theta_range=self.two_theta_range
              )

            def intensity_func(x, y):
                return y * self.lorentz(two_theta_values, x, self.gamma)

            vmap_intensity_func = jax.vmap(intensity_func, in_axes=(0, 0), out_axes=0)
            intensities = jnp.sum(
                vmap_intensity_func(xrd_pattern.x, xrd_pattern.y), axis=0
            )
            # Add to the total intensities
            total_intensity += intensities

        # Average the intensities over the number of configurations
        averaged_intensity = total_intensity / n_samples

        return two_theta_values, averaged_intensity

    def two_theta_to_d_spacing(self, two_theta):
        """Converts 2θ angle to d-spacing using Bragg's law.

        Args:
            two_theta: 2θ angles in degrees.

        Returns:
            d-spacing in Angstroms.

        """
        theta = np.radians(two_theta / 2)
        d_spacing = self.wavelength / (2 * np.sin(theta))
        return d_spacing

    def compute_xrd(self, xp, Ls, n_samples=1):
        """Computes normalized XRD pattern with d-spacing.

        Args:
            xp: Atomic positions from ensemble.
            Ls: Lattice vectors.
            n_samples: Number of configurations to average.

        Returns:
            Tuple of (d_spacing, normalized_intensity).

        """
        two_theta, averaged_intensity = self.compute_average_xrd_pymatgen(
            xp, Ls, n_samples=n_samples
        )

        d_spacing = self.two_theta_to_d_spacing(two_theta)

        averaged_intensity /= averaged_intensity.max() / 100
        return d_spacing, averaged_intensity
