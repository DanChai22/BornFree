# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file may have been modified by Bytedance Inc. (“Bytedance Modifications”).
# All Bytedance Modifications are Copyright 2022 Bytedance Inc.

# This file may have been modified by Shengdu Chai.
# Modifications Copyright (c) 2025 Shengdu Chai

import logging

import jax
import jax.numpy as jnp
from folx import forward_laplacian
from folx.api import FwdJacobian, FwdLaplArray
from pyscf.pbc.gto import Cell as PyscfCell

from BornFree import constants, ewaldsum
from BornFree.base_config import CrystalLatticeConfig
from BornFree.network import network_block
from BornFree.utils.units import gpa2habohr3

logger = logging.getLogger(__name__)


class BaseLocalEnergy:
    """Base class for all local energy calculators.

    This provides common methods and properties used across all the various
    local energy calculator implementations.
    """

    def __init__(self, simulation_cell: PyscfCell, mode="for", partition_number=3):
        """Initialize the base local energy calculator.

        Args:
            simulation_cell: PySCF cell object with system information
            mode: Kinetic energy evaluation method
            partition_number: Number of partitions for 'partition' mode

        """
        self.simulation_cell = simulation_cell
        self.mode = mode
        self.partition_number = partition_number

    @property
    def dtype(self):
        """Return the precision used in the simulation cell."""
        return self.simulation_cell.a.dtype

    def local_kinetic_energy_real_imag(self, f):
        """Calculate local kinetic energy using real and imaginary parts."""
        pass

    def local_kinetic_energy_real_imag_hessian(self, f):
        """Calculate local kinetic energy with Hessian using real and imaginary parts."""
        pass

    def local_kinetic_energy_folx(self, f):
        """Calculate local kinetic energy using forward Laplacian differentiation."""
        pass

    def local_kinetic_energy_partition(self, f):
        """Calculate local kinetic energy using partition method."""
        pass

    def _initialize_mass_array(self, is_deuterium=False):
        """Initialize mass array for quantum calculations.

        Args:
            is_deuterium: Whether to use deuterium (D) mass instead of hydrogen (H)

        Returns:
            Array of masses for all particles (nuclei + electrons)

        """
        ne = sum(self.simulation_cell.nelec) * 3

        if is_deuterium:
            logger.info("Using mass of D (deuterium).")
            mass_factor = 2.0
        else:
            logger.info("Using mass of H (hydrogen).")
            mass_factor = 1.0

        # Nuclear masses in atomic units
        mass_array = mass_factor * jnp.repeat(
            self.simulation_cell.atom_mass_list() * constants.ATOM_MASS,
            3,
        ).astype(self.dtype)

        # Concatenate nuclear masses with electron masses (=1.0 in atomic units)
        return jnp.concatenate((mass_array, jnp.ones(ne, dtype=mass_array.dtype)))

    def _select_kinetic_energy_method(self, f):
        """Select the appropriate kinetic energy method based on mode.

        Args:
            f: Function that returns log𝜓

        Returns:
            Function that computes kinetic energy

        Raises:
            ValueError: If an unrecognized mode is specified

        """
        if self.mode == "for":
            return self.local_kinetic_energy_real_imag(f)
        elif self.mode == "hessian" and hasattr(
            self, "local_kinetic_energy_real_imag_hessian"
        ):
            return self.local_kinetic_energy_real_imag_hessian(f)
        elif self.mode == "partition":
            return self.local_kinetic_energy_partition(f)
        elif self.mode == "folx":
            return self.local_kinetic_energy_folx(f)
        else:
            raise ValueError(f"Unrecognized laplacian evaluation mode: {self.mode}")

    def _make_kinetic_energy_fn(self, ke_ri_fn):
        """Create a kinetic energy function that sums real and imaginary parts.

        Args:
            ke_ri_fn: Function that returns [real_part, imag_part] of kinetic energy

        Returns:
            Function that computes total kinetic energy by summing components

        """

        def kinetic_energy(params, positions):
            """Compute total kinetic energy."""
            return sum(ke_ri_fn(params, positions))

        return kinetic_energy

    def _make_gradient_closures(self, f, params):
        """Create gradient functions and their closures for real and imaginary parts.

        This is a helper method to avoid repeating the common pattern of creating
        gradients for real and imaginary components of the wavefunction.

        Args:
            f: Function that returns log𝜓 (complex-valued)
            params: Parameters to fix in the closure

        Returns:
            Tuple of (grad_f_real_closure, grad_f_imag_closure) where each closure
            takes only positions as input

        """
        grad_f_real = jax.grad(lambda p, y: f(p, y).real, argnums=1)
        grad_f_imag = jax.grad(lambda p, y: f(p, y).imag, argnums=1)

        def grad_f_real_closure(y):
            return grad_f_real(params, y)

        def grad_f_imag_closure(y):
            return grad_f_imag(params, y)

        return grad_f_real_closure, grad_f_imag_closure

    def _format_kinetic_result(self, real_part, imag_part, factor=0.5):
        """Format kinetic energy result as [real, imag*1j] with a multiplicative factor.

        Args:
            real_part: Real component of kinetic energy
            imag_part: Imaginary component of kinetic energy
            factor: Multiplicative factor (default 0.5 for standard kinetic energy)

        Returns:
            List of [real_scaled, imag_scaled*1j]

        """
        return [-factor * real_part, -factor * 1j * imag_part]


class LocalEnergy(BaseLocalEnergy):
    """Handles local energy calculations for solid-state systems.

    This class provides methods for computing the local energy of a quantum system,
    including both kinetic and potential energy terms. It supports various methods
    for computing the kinetic energy, trading off between computational efficiency
    and memory usage.
    """

    def __init__(
        self, simulation_cell, mode="for", partition_number=3, nuclear_treatment="fixed"
    ):
        """Initializes the LocalEnergy calculator.

        Args:
            simulation_cell: PySCF cell object with system information
            mode: Kinetic energy evaluation method
            partition_number: Number of partitions for 'partition' mode
            nuclear_treatment: How to handle nuclei ('fixed')

        """
        super().__init__(simulation_cell, mode, partition_number)
        self.nuclear_treatment = nuclear_treatment

    def local_kinetic_energy_real_imag(self, f):
        """Computes kinetic energy by evaluating real and imaginary parts separately.

        Uses forward/backward differentiation to compute the Laplacian of log𝜓
        divided by 𝜓. This method is memory efficient but potentially slower.

        Args:
            f: Function that returns log𝜓 for given parameters and positions

        Returns:
            Function that computes kinetic energy given parameters and positions

        """

        def _lapl_over_f(params, x):
            ne = x.shape[-1]
            eye = jnp.eye(ne)
            grad_f_real_closure, grad_f_imag_closure = self._make_gradient_closures(
                f, params
            )

            def _body_fun(i, val):
                primal_real, tangent_real = jax.jvp(
                    grad_f_real_closure, (x,), (eye[i],)
                )
                primal_imag, tangent_imag = jax.jvp(
                    grad_f_imag_closure, (x,), (eye[i],)
                )
                kine_real = (
                    val[0] + tangent_real[i] + primal_real[i] ** 2 - primal_imag[i] ** 2
                )
                kine_imag = (
                    val[1] + tangent_imag[i] + 2 * primal_real[i] * primal_imag[i]
                )
                return [kine_real, kine_imag]

            result = jax.lax.fori_loop(0, ne, _body_fun, [0.0, 0.0])
            return self._format_kinetic_result(result[0], result[1])

        return _lapl_over_f

    def local_kinetic_energy_real_imag_hessian(self, f):
        """Computes kinetic energy using Hessian-based evaluation.

        Uses JAX's Hessian computation for parallel evaluation of the Laplacian.
        This method is faster but requires more memory.

        Args:
            f: Function that returns log𝜓

        Returns:
            Function that computes kinetic energy

        """

        def _lapl_over_f(params, x):
            grad_f_real = jax.grad(lambda p, y: f(p, y).real, argnums=1)
            grad_f_imag = jax.grad(lambda p, y: f(p, y).imag, argnums=1)
            hessian_f_real = jax.hessian(lambda p, y: f(p, y).real, argnums=1)
            hessian_f_imag = jax.hessian(lambda p, y: f(p, y).imag, argnums=1)
            v_grad_f_real = grad_f_real(params, x)
            v_grad_f_imag = grad_f_imag(params, x)
            real_kinetic = (
                jnp.trace(
                    hessian_f_real(params, x),
                )
                + jnp.sum(v_grad_f_real**2)
                - jnp.sum(v_grad_f_imag**2)
            )
            imag_kinetic = jnp.trace(
                hessian_f_imag(params, x),
            ) + jnp.sum(2 * v_grad_f_real * v_grad_f_imag)

            return self._format_kinetic_result(real_kinetic, imag_kinetic)

        return _lapl_over_f

    def local_kinetic_energy_folx(self, f):
        """Computes kinetic energy using optimized forward differentiation.

        Uses the folx package which implements an optimized version of forward
        mode automatic differentiation for Laplacian computation. This method
        is typically the fastest among all available methods.

        Args:
            f: Function that returns log𝜓

        Returns:
            Function that computes kinetic energy

        """

        def _lapl_over_f(params, x):
            def func(flat_x):
                return f(params, flat_x)

            fwd_f = forward_laplacian(func)
            output = fwd_f(x)
            return [-0.5 * ((output.jacobian.dense_array**2).sum() + output.laplacian)]

        return _lapl_over_f

    def local_kinetic_energy_partition(self, f):
        """Computes kinetic energy using a partitioned approach.

        This method divides the computation into partitions to balance between
        memory usage and computational efficiency. The number of partitions can
        be adjusted through the partition_number attribute.

        Args:
            f: Function that returns log𝜓

        Returns:
            Function that computes kinetic energy

        """
        vjvp = jax.vmap(jax.jvp, in_axes=(None, None, 0))

        def _lapl_over_f(params, x):
            n = x.shape[0]
            eye = jnp.eye(n)
            grad_f_closure_real, grad_f_closure_imag = self._make_gradient_closures(
                f, params
            )

            eyes = jnp.asarray(jnp.array_split(eye, self.partition_number))

            def _body_fun(val, e):
                primal_real, tangent_real = vjvp(grad_f_closure_real, (x,), (e,))
                primal_imag, tangent_imag = vjvp(grad_f_closure_imag, (x,), (e,))
                return val, ([primal_real, primal_imag], [tangent_real, tangent_imag])

            _, (plist, tlist) = jax.lax.scan(_body_fun, None, eyes)
            primal = [primal.reshape((-1, primal.shape[-1])) for primal in plist]
            tangent = [tangent.reshape((-1, tangent.shape[-1])) for tangent in tlist]

            real_kinetic = (
                jnp.trace(tangent[0])
                + jnp.trace(primal[0] ** 2).sum()
                - jnp.trace(primal[1] ** 2).sum()
            )
            imag_kinetic = (
                jnp.trace(tangent[1]) + jnp.trace(2 * primal[0] * primal[1]).sum()
            )
            return [-0.5 * real_kinetic, -0.5 * 1j * imag_kinetic]

        return _lapl_over_f

    def local_ewald_energy(self):
        """Creates a function to compute the Ewald energy.

        The Ewald energy includes all electrostatic interactions in the periodic
        system:
        - Electron-electron repulsion
        - Electron-ion attraction
        - Ion-ion repulsion

        Returns:
            Function that computes Ewald energy for given electron positions

        """
        ewald = ewaldsum.EwaldSum_nvt_fixed(self.simulation_cell)

        def _local_ewald_energy(x):
            energy = ewald.energy(x)
            return energy

        return _local_ewald_energy

    def local_energy_separate(self, f):
        """Creates a function to compute the total local energy.

        Combines kinetic and potential energy terms to compute the total local
        energy E_L = H𝜓/𝜓. The kinetic energy computation method is determined
        by the mode attribute.

        Args:
            f: Function that returns log𝜓

        Returns:
            Function that computes total local energy

        """
        ke_ri = self._select_kinetic_energy_method(f)
        ke = self._make_kinetic_energy_fn(ke_ri)
        ew = self.local_ewald_energy()

        def _local_energy(params, x):
            kinetic = ke(params, x)
            ewald_ee, ewald_ei, ewald_ii = ew(x)
            return kinetic, ewald_ee, ewald_ei, ewald_ii

        return _local_energy


class BaseQuantumKineticEnergy(BaseLocalEnergy):
    """Base class for quantum kinetic energy calculations with mass corrections.

    This class consolidates common logic for computing kinetic energies of quantum
    systems (both NVT and NPT). Subclasses must implement coordinate transformation
    methods to handle different ensemble-specific metrics.
    """

    def __init__(
        self, simulation_cell, mode="for", is_deuterium=False, partition_number=3
    ):
        """Initialize the quantum kinetic energy calculator.

        Args:
            simulation_cell: PySCF cell object with system information
            mode: Kinetic energy evaluation method ('for', 'hessian', 'partition', 'folx')
            is_deuterium: Whether to use deuterium mass instead of hydrogen mass
            partition_number: Number of partitions for 'partition' mode

        """
        super().__init__(simulation_cell, mode, partition_number)
        self.ne = sum(self.simulation_cell.nelec) * 3
        self.natom = self.simulation_cell.natm
        self.mass_array = self._initialize_mass_array(is_deuterium)

    def _get_coordinate_transform_matrices(self, params, x):
        """Get coordinate transformation matrices for kinetic energy computation.

        This method must be implemented by subclasses to provide ensemble-specific
        coordinate transformations (identity for NVT, metric tensor for NPT).

        Args:
            params: Network parameters (may contain cell parameters for NPT)
            x: Position array

        Returns:
            Tuple of (eye_matrix, transform_matrix):
                - eye_matrix: Identity or metric-transformed identity for JVP
                - transform_matrix: Matrix for transforming gradients (or None for NVT)

        """
        raise NotImplementedError("Subclasses must implement coordinate transforms")

    def _transform_gradient(self, gradient, transform_matrix):
        """Apply coordinate transformation to gradient.

        Args:
            gradient: Gradient vector to transform
            transform_matrix: Transformation matrix (None for identity transform)

        Returns:
            Transformed gradient

        """
        if transform_matrix is None:
            return gradient
        return jnp.matmul(transform_matrix.T, gradient)

    def local_kinetic_energy_cross(self, f1, f2, particle):
        """Compute cross kinetic energy terms between two wavefunctions.

        Evaluates the cross kinetic energy contribution between two different
        wavefunctions (e.g., atomic and electronic components in Born-Oppenheimer
        corrections).

        Args:
            f1: Function returning log(ψ₁) for the first wavefunction
            f2: Function returning log(ψ₂) for the second wavefunction
            particle: Particle type to compute over, either "elec" or "atom"

        Returns:
            Function that computes cross kinetic energy given parameters and positions

        Raises:
            ValueError: If particle is not "elec" or "atom"

        """

        def _compute_cross_kinetic(params, x):
            eye, inv_j = self._get_coordinate_transform_matrices(params, x)
            grad_f1_real_closure, grad_f1_imag_closure = self._make_gradient_closures(
                f1, params
            )
            grad_f2_real_closure, grad_f2_imag_closure = self._make_gradient_closures(
                f2, params
            )

            def _body_fun(i, val):
                primal_real1, _ = jax.jvp(grad_f1_real_closure, (x,), (eye[i],))
                primal_imag1, _ = jax.jvp(grad_f1_imag_closure, (x,), (eye[i],))
                primal_real2, _ = jax.jvp(grad_f2_real_closure, (x,), (eye[i],))
                primal_imag2, _ = jax.jvp(grad_f2_imag_closure, (x,), (eye[i],))

                # Apply coordinate transformation
                primal_real1 = self._transform_gradient(primal_real1, inv_j)
                primal_imag1 = self._transform_gradient(primal_imag1, inv_j)
                primal_real2 = self._transform_gradient(primal_real2, inv_j)
                primal_imag2 = self._transform_gradient(primal_imag2, inv_j)

                kine_real = (
                    val[0]
                    + (
                        primal_real1[i] * primal_real2[i]
                        - primal_imag1[i] * primal_imag2[i]
                    )
                    / self.mass_array[i]
                )
                kine_imag = (
                    val[1]
                    + (
                        primal_real1[i] * primal_imag2[i]
                        + primal_real2[i] * primal_imag1[i]
                    )
                    / self.mass_array[i]
                )
                return [kine_real, kine_imag]

            n_dims = x.shape[-1]
            if particle == "elec":
                result = jax.lax.fori_loop(
                    self.natom * 3, n_dims, _body_fun, [0.0, 0.0]
                )
                return self._format_kinetic_result(result[0], result[1], factor=1.0)
            elif particle == "atom":
                result = jax.lax.fori_loop(0, self.natom * 3, _body_fun, [0.0, 0.0])
                return self._format_kinetic_result(result[0], result[1], factor=1.0)
            else:
                raise ValueError(
                    f"Unrecognized particle type: {particle}. Must be 'elec' or 'atom'."
                )

        return _compute_cross_kinetic

    def local_kinetic_energy_real_imag_atom(self, f):
        """Compute atomic kinetic energy contributions for Born-Oppenheimer corrections.

        Args:
            f: Function that returns log𝜓

        Returns:
            Function that computes atomic kinetic energy

        """

        def _compute_atomic_laplacian(params, x):
            eye, inv_j = self._get_coordinate_transform_matrices(params, x)
            grad_f_real_closure, grad_f_imag_closure = self._make_gradient_closures(
                f, params
            )

            def _body_fun(i, val):
                primal_real, tangent_real = jax.jvp(
                    grad_f_real_closure, (x,), (eye[i],)
                )
                primal_imag, tangent_imag = jax.jvp(
                    grad_f_imag_closure, (x,), (eye[i],)
                )

                # Apply coordinate transformation
                primal_real = self._transform_gradient(primal_real, inv_j)
                primal_imag = self._transform_gradient(primal_imag, inv_j)

                kine_real = (
                    val[0]
                    + (tangent_real[i] + primal_real[i] ** 2 - primal_imag[i] ** 2)
                    / self.mass_array[i]
                )
                kine_imag = (
                    val[1]
                    + (tangent_imag[i] + 2 * primal_real[i] * primal_imag[i])
                    / self.mass_array[i]
                )
                return [kine_real, kine_imag]

            result = jax.lax.fori_loop(0, self.natom * 3, _body_fun, [0.0, 0.0])
            return self._format_kinetic_result(result[0], result[1])

        return _compute_atomic_laplacian

    def local_kinetic_energy_real_imag(self, f):
        """Compute kinetic energy by evaluating real and imaginary parts separately.

        Uses forward/backward differentiation to compute the Laplacian sequentially.
        This method is memory efficient but potentially slower for large systems.

        Args:
            f: Function that returns log𝜓

        Returns:
            Function that computes kinetic energy given parameters and positions

        """

        def _compute_laplacian(params, x):
            eye, inv_j = self._get_coordinate_transform_matrices(params, x)
            grad_f_real_closure, grad_f_imag_closure = self._make_gradient_closures(
                f, params
            )

            def _body_fun(i, val):
                primal_real, tangent_real = jax.jvp(
                    grad_f_real_closure, (x,), (eye[i],)
                )
                primal_imag, tangent_imag = jax.jvp(
                    grad_f_imag_closure, (x,), (eye[i],)
                )

                # Apply coordinate transformation
                primal_real = self._transform_gradient(primal_real, inv_j)
                primal_imag = self._transform_gradient(primal_imag, inv_j)

                kine_real = (
                    val[0]
                    + (tangent_real[i] + primal_real[i] ** 2 - primal_imag[i] ** 2)
                    / self.mass_array[i]
                )
                kine_imag = (
                    val[1]
                    + (tangent_imag[i] + 2 * primal_real[i] * primal_imag[i])
                    / self.mass_array[i]
                )
                return [kine_real, kine_imag]

            n_dims = x.shape[-1]
            result = jax.lax.fori_loop(0, n_dims, _body_fun, [0.0, 0.0])
            return self._format_kinetic_result(result[0], result[1])

        return _compute_laplacian

    def local_kinetic_energy_partition(self, f):
        """Compute kinetic energy using a partitioned approach.

        Divides the computation into partitions to balance memory usage and
        computational efficiency.

        Args:
            f: Function that returns log𝜓

        Returns:
            Function that computes kinetic energy

        """
        vjvp = jax.vmap(jax.jvp, in_axes=(None, None, 0))

        def _compute_laplacian_partitioned(params, x):
            eye, inv_j = self._get_coordinate_transform_matrices(params, x)
            grad_f_closure_real, grad_f_closure_imag = self._make_gradient_closures(
                f, params
            )

            eyes = jnp.asarray(jnp.array_split(eye, self.partition_number))

            def _body_fun(val, e):
                primal_real, tangent_real = vjvp(grad_f_closure_real, (x,), (e,))
                primal_imag, tangent_imag = vjvp(grad_f_closure_imag, (x,), (e,))
                return val, ([primal_real, primal_imag], [tangent_real, tangent_imag])

            _, (plist, tlist) = jax.lax.scan(_body_fun, None, eyes)

            # Apply coordinate transformation if needed
            if inv_j is not None:
                primal = [
                    jnp.matmul(primal.reshape((-1, primal.shape[-1])), inv_j)
                    for primal in plist
                ]
            else:
                primal = [primal.reshape((-1, primal.shape[-1])) for primal in plist]

            tangent = [tangent.reshape((-1, tangent.shape[-1])) for tangent in tlist]

            real_kinetic = (
                jnp.trace(tangent[0] / self.mass_array)
                + jnp.trace(primal[0] ** 2 / self.mass_array).sum()
                - jnp.trace(primal[1] ** 2 / self.mass_array).sum()
            )
            imag_kinetic = (
                jnp.trace(tangent[1] / self.mass_array)
                + jnp.trace(2 * primal[0] * primal[1] / self.mass_array).sum()
            )
            return [-0.5 * real_kinetic, -0.5 * 1j * imag_kinetic]

        return _compute_laplacian_partitioned

    def local_energy_atom_kinetic_diff(self, f1, f2, f3):
        """Generate function to compute Born-Oppenheimer kinetic energy differences.

        Computes the kinetic energy differences between total, atomic, and
        electronic wavefunctions for Born-Oppenheimer corrections.

        Args:
            f1: Function returning log of total wavefunction
            f2: Function returning log of atomic wavefunction
            f3: Function returning log of electronic wavefunction

        Returns:
            Function that computes all kinetic energy components

        Raises:
            ValueError: If mode is not 'for' (only sequential mode supported)

        """
        if self.mode == "for":
            ke_ri_atom_total = self.local_kinetic_energy_real_imag_atom(f1)
            ke_ri_atom_atom = self.local_kinetic_energy_real_imag_atom(f2)
            ke_ri_atom_elec = self.local_kinetic_energy_real_imag_atom(f3)
            ke_ri_atom_cross = self.local_kinetic_energy_cross(f2, f3, particle="atom")
        else:
            raise ValueError(
                f"Unrecognized laplacian evaluation mode '{self.mode}' for atom kinetic differences. "
                f"Only 'for' mode is supported for this operation."
            )

        ke_atom_total = self._make_kinetic_energy_fn(ke_ri_atom_total)
        ke_atom_atom = self._make_kinetic_energy_fn(ke_ri_atom_atom)
        ke_atom_elec = self._make_kinetic_energy_fn(ke_ri_atom_elec)
        ke_atom_cross = self._make_kinetic_energy_fn(ke_ri_atom_cross)

        def _compute_kinetic_differences(params, x):
            kinetic_atom_total = ke_atom_total(params, x)
            kinetic_atom_atom = ke_atom_atom(params, x)
            kinetic_atom_elec = ke_atom_elec(params, x)
            kinetic_atom_cross = ke_atom_cross(params, x)

            return (
                kinetic_atom_total,
                kinetic_atom_atom,
                kinetic_atom_elec,
                kinetic_atom_cross,
            )

        return _compute_kinetic_differences


class LocalEnergy_quantum(BaseQuantumKineticEnergy):
    """Handles local energy calculations for quantum nuclei in NVT ensemble.

    This class provides methods for computing the local energy when nuclei are
    treated quantum mechanically with fixed cell parameters (NVT ensemble).
    """

    def __init__(
        self, simulation_cell, mode="for", is_deuterium=False, partition_number=3
    ):
        """Initialize the local energy function for quantum nuclei in NVT ensemble.

        Args:
            simulation_cell: PySCF object of simulation cell
            mode: Specify the evaluation style of local energy:
                'for' - calculates the laplacian of each electron one by one (slow, saves GPU memory)
                'hessian' - calculates the laplacian in a highly parallelized mode (fast, requires more GPU memory)
                'partition' - calculate the laplacian in a moderate way
                'folx' - calculate the laplacian using forward mode AD
            is_deuterium: Whether to use deuterium mass instead of hydrogen mass
            partition_number: Only used if 'partition' mode is employed.
                partition_number must be divisible by (dim * number of electrons).
                The smaller the faster, but requires more memory.

        """
        super().__init__(simulation_cell, mode, is_deuterium, partition_number)

    def _get_coordinate_transform_matrices(self, params, x):
        """Get identity transformation for NVT (Cartesian coordinates).

        Args:
            params: Network parameters (unused for NVT)
            x: Position array

        Returns:
            Tuple of (eye_matrix, None) - no transformation needed for NVT

        """
        ne = x.shape[-1]
        eye = jnp.eye(ne, dtype=x.dtype)
        return eye, None

    def local_kinetic_energy_folx(self, f):
        """Compute kinetic energy using optimized forward differentiation (folx).

        This method uses the folx package for efficient Laplacian computation
        with mass corrections for NVT ensemble.

        Args:
            f: Function that returns log𝜓

        Returns:
            Function that computes kinetic energy

        """

        def _compute_folx_laplacian(params, x):
            ne = x.shape[-1]
            eyes = jnp.eye(ne, dtype=x.dtype) / jnp.sqrt(
                self.mass_array.reshape([-1, 1])
            )

            def func(flat_x):
                return f(params, flat_x)

            fwd_f = forward_laplacian(func)
            x_input = FwdLaplArray(x, FwdJacobian(eyes), jnp.zeros_like(x))
            output = fwd_f(x_input)
            return [-0.5 * ((output.jacobian.dense_array**2).sum() + output.laplacian)]

        return _compute_folx_laplacian

    def local_ewald_energy(self):
        """Generate function to compute Ewald energy for quantum NVT ensemble.

        Returns:
            Function that computes Ewald energy components (ee, ei, ii)

        """
        ewald = ewaldsum.EwaldSum_nvt_quantum(self.simulation_cell)

        def _compute_ewald_energy(x):
            return ewald.energy(x)

        return _compute_ewald_energy

    def local_energy_separate(self, f):
        """Generate function to compute the total local energy.

        Combines kinetic and Ewald energy terms for NVT ensemble.

        Args:
            f: Function that returns log𝜓

        Returns:
            Function that computes total local energy components

        """
        kinetic_energy_ri = self._select_kinetic_energy_method(f)
        kinetic_energy = self._make_kinetic_energy_fn(kinetic_energy_ri)
        ewald_energy = self.local_ewald_energy()

        def _compute_local_energy(params, x):
            kinetic = kinetic_energy(params, x)
            ewald_ee, ewald_ei, ewald_ii = ewald_energy(x)
            return kinetic, ewald_ee, ewald_ei, ewald_ii

        return _compute_local_energy


class LocalEnthalpy_quantum(BaseQuantumKineticEnergy):
    """Quantum version of local enthalpy calculations with pressure terms for NPT ensemble.

    This class extends quantum kinetic energy calculations to include pressure-volume
    work terms and curvilinear coordinate transformations for variable cell simulations.
    """

    def __init__(
        self,
        simulation_cell,
        target_pressure=100.0,
        mode="for",
        lattice_config: CrystalLatticeConfig | None = None,
        is_deuterium=False,
        partition_number=3,
    ):
        """Initialize the quantum local enthalpy calculator for NPT ensemble.

        Args:
            simulation_cell: PySCF cell object with system information
            target_pressure: Target pressure in GPa
            mode: Kinetic energy evaluation method
            lattice_config: How to parameterize the lattice
            is_deuterium: Whether to use deuterium mass instead of hydrogen mass
            partition_number: Number of partitions for 'partition' mode
            folx_sparsity_threshold: Sparsity threshold for folx forward Laplacian

        """
        super().__init__(simulation_cell, mode, is_deuterium, partition_number)
        self.target_pressure = target_pressure
        self.lattice_config = lattice_config

    def get_inv_j(self, cellpar):
        """Get the inverse Jacobian for coordinate transformation.

        Args:
            cellpar: Cell parameters

        Returns:
            Transposed inverse Jacobian matrix

        """
        inv_j = network_block.get_inv_jacobian(cellpar, self.lattice_config)
        return inv_j.T

    def get_inv_metric(self, cellpar):
        """Get the inverse metric tensor for curvilinear coordinates.

        Args:
            cellpar: Cell parameters

        Returns:
            Inverse metric tensor G^{-1} = J^{-1} (J^{-1})^T

        """
        inv_j = self.get_inv_j(cellpar)
        inv_metric = jnp.matmul(inv_j, inv_j.T)
        return inv_metric

    def _get_coordinate_transform_matrices(self, params, x):
        """Get metric tensor transformation for NPT (curvilinear coordinates).

        Args:
            params: Network parameters containing cell parameters
            x: Position array

        Returns:
            Tuple of (metric_eye, inv_j) for coordinate transformation

        """
        ne = x.shape[-1] // 3
        inv_metric = self.get_inv_metric(params["cell"])
        eye = jnp.kron(jnp.eye(ne, dtype=x.dtype), inv_metric)
        inv_j = jnp.kron(jnp.eye(ne, dtype=x.dtype), self.get_inv_j(params["cell"]))
        return eye, inv_j

    def local_kinetic_energy_folx(self, f):
        """Compute kinetic energy using optimized forward differentiation for NPT.

        This method uses the folx package with curvilinear coordinate transformations
        and configurable sparsity threshold.

        Args:
            f: Function that returns log𝜓

        Returns:
            Function that computes kinetic energy

        """

        def _compute_folx_laplacian(params, x):
            ne = x.shape[-1] // 3
            eye = jnp.kron(
                jnp.eye(ne, dtype=x.dtype), self.get_inv_j(params["cell"]).T
            ) / jnp.sqrt(self.mass_array.reshape([-1, 1]))

            def func(flat_x):
                return f(params, flat_x)

            fwd_f = forward_laplacian(func, sparsity_threshold=6)
            x_input = FwdLaplArray(x, FwdJacobian(eye), jnp.zeros_like(x))
            output = fwd_f(x_input)
            return [-0.5 * ((output.jacobian.dense_array**2).sum() + output.laplacian)]

        return _compute_folx_laplacian

    def local_ewald_energy(self):
        """Generate function to compute Ewald energy with pressure-volume term for NPT.

        Returns:
            Function that computes Ewald energy components (ee, ei, ii) and PV term

        """
        ewald = ewaldsum.EwaldSum_npt_quantum(self.simulation_cell, self.lattice_config)

        def _compute_ewald_energy_with_pv(cellpar, x):
            r, jac = network_block.convert_to_simulation_cell(
                cellpar, x, self.lattice_config
            )
            ee, ei, ii = ewald.energy(cellpar, r)
            pv = gpa2habohr3(self.target_pressure) * jnp.linalg.det(jac)
            return ee, ei, ii, pv

        return _compute_ewald_energy_with_pv

    def local_energy_separate(self, f):
        """Generate function to compute the total local enthalpy for NPT ensemble.

        Combines kinetic energy, Ewald energy, and pressure-volume work term.

        Args:
            f: Function that returns log𝜓

        Returns:
            Function that computes total local enthalpy components

        """
        kinetic_energy_ri = self._select_kinetic_energy_method(f)
        kinetic_energy = self._make_kinetic_energy_fn(kinetic_energy_ri)
        ewald_energy = self.local_ewald_energy()

        def _compute_local_enthalpy(params, x):
            kinetic = kinetic_energy(params, x)
            ewald_ee, ewald_ei, ewald_ii, pv = ewald_energy(params["cell"], x)
            return kinetic, ewald_ee, ewald_ei, ewald_ii, pv

        return _compute_local_enthalpy


def make_BO_kin(
    total_network,
    atom_network,
    elec_network,
    simulation_cell,
    mode="for",
    lattice_config: CrystalLatticeConfig | None = None,
    is_deuterium=False,
    partition_number=3,
    ensemble="nvt",
):
    """Create a function to compute kinetic energy differences between wavefunctions.

    This unified interface creates a function that computes the kinetic energy
    differences between the total, atomic, and electronic wavefunctions.

    Args:
        total_network: Function for total wavefunction
        atom_network: Function for atomic wavefunction
        elec_network: Function for electronic wavefunction
        simulation_cell: PySCF cell object with system information
        mode: Kinetic energy evaluation method
        lattice_config: How to parameterize the lattice (NPT only)
        is_deuterium: Whether to use isotope average mass
        partition_number: Number of partitions for 'partition' mode
        ensemble: 'NVT' or 'NPT'

    Returns:
        Function that computes the kinetic energy differences

    """
    if ensemble == "NPT":
        el_class = LocalEnthalpy_quantum(
            simulation_cell=simulation_cell,
            mode=mode,
            lattice_config=lattice_config,
            is_deuterium=is_deuterium,
            partition_number=partition_number,
        )
    elif ensemble == "NVT":
        el_class = LocalEnergy_quantum(
            simulation_cell=simulation_cell,
            mode=mode,
            is_deuterium=is_deuterium,
            partition_number=partition_number,
        )
    else:
        raise ValueError(f"Unsupported ensemble: {ensemble}. Use 'NVT' or 'NPT'.")

    el_fun = el_class.local_energy_atom_kinetic_diff(
        total_network, atom_network, elec_network
    )
    batch_local_energy = jax.vmap(el_fun, in_axes=(None, 0), out_axes=0)

    @constants.pmap
    def expec_kinetic_dif(params, data):
        ke_t, ke_a, ke_e, ke_c = batch_local_energy(params, data)
        mean_ke_t = jnp.mean(ke_t)
        mean_ke_a = jnp.mean(ke_a)
        mean_ke_e = jnp.mean(ke_e)
        mean_ke_c = jnp.mean(ke_c)
        pmean_ke_t = constants.pmean(mean_ke_t)
        pmean_ke_a = constants.pmean(mean_ke_a)
        pmean_ke_e = constants.pmean(mean_ke_e)
        pmean_ke_c = constants.pmean(mean_ke_c)
        return pmean_ke_t, pmean_ke_a, pmean_ke_e, pmean_ke_c

    return expec_kinetic_dif
