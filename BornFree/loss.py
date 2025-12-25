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
import operator
from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp
import kfac_jax
from jax import Array
from pyscf.pbc.gto import Cell as PyscfCell

from BornFree import base_config, constants, hamiltonian
from BornFree.base_config import CrystalLatticeConfig

logger = logging.getLogger(__name__)


@chex.dataclass
class AuxiliaryLossData:
    """Auxiliary data from loss function evaluation.

    This dataclass stores various energy components and statistics computed
    during the loss function evaluation for monitoring and analysis.

    Attributes:
        variance: Variance of the local energy.
        local_energy: Local energy values for each sample in the batch.
        imaginary: Imaginary part of the mean local energy.
        kinetic: Kinetic energy component.
        ewald: Total Ewald energy (sum of ee, ei, ii components).
        ewald_ee: Electron-electron Ewald interaction energy.
        ewald_ei: Electron-ion Ewald interaction energy.
        ewald_ii: Ion-ion Ewald interaction energy.
        pv: Pressure-volume work term (for NPT ensemble).

    """

    variance: Array
    local_energy: Array
    imaginary: Array
    kinetic: Array
    ewald: Array
    ewald_ee: Array
    ewald_ei: Array
    ewald_ii: Array
    pv: Array


def clip_local_energy_func(diff, clip_local_energy, clip_type="complex"):
    """Clips local energy values to improve training stability.

    Implements two types of clipping:
        1. Complex: Clips the magnitude while preserving phase
        2. Real: Clips real and imaginary parts separately

    Args:
        diff: Local energy differences to clip. Shape: (batch_size,)
        clip_local_energy: Clipping threshold. If ≤ 0, no clipping is performed.
        clip_type: Type of clipping to use. Options are 'complex' for polar
            coordinate clipping or 'real' for Cartesian coordinate clipping.
            Defaults to 'complex'.

    Returns:
        Array: Clipped energy differences with same shape as input.

    """
    if clip_local_energy <= 0.0:
        return diff

    if clip_type == "complex":
        # Compute radius and phase in polar coordinates
        radius = jnp.abs(diff)
        phase = jnp.angle(diff)

        # Compute statistics for clipping
        radius_std = constants.pmean(radius.std())
        radius_median = constants.pmean(jnp.median(radius))

        # Clip radius while preserving phase
        clip_radius = jnp.clip(
            radius,
            radius_median - radius_std * clip_local_energy,
            radius_median + radius_std * clip_local_energy,
        )
        return clip_radius * jnp.exp(1j * phase)

    elif clip_type == "real":
        # Compute mean absolute values for real and imaginary parts
        threshold_real = constants.pmean(jnp.mean(jnp.abs(diff.real)))
        threshold_imag = constants.pmean(jnp.mean(jnp.abs(diff.imag)))

        # Clip real and imaginary parts separately
        clipped_real = jnp.clip(
            diff.real,
            -clip_local_energy * threshold_real,
            clip_local_energy * threshold_real,
        )
        clipped_imag = jnp.clip(
            diff.imag,
            -clip_local_energy * threshold_imag,
            clip_local_energy * threshold_imag,
        )
        return clipped_real + 1j * clipped_imag

    else:
        raise ValueError(
            f"Unrecognized clip type: '{clip_type}'. Must be 'complex' or 'real'."
        )


def create_frozen_mask(params_pytree, *keys_to_freeze):
    """Create a PyTree mask for freezing specific parameter keys during training.

    This function generates a boolean mask with the same structure as the input
    parameter PyTree. Array parameters are marked as trainable (True) by default,
    except for those specified in keys_to_freeze which are set to False.

    Args:
        params_pytree: A PyTree containing model parameters with the same
            structure as the training parameters.
        *keys_to_freeze: Variable number of string keys to freeze. These keys
            should exist at the top level of the params_pytree dictionary.

    Returns:
        dict: A PyTree mask with the same structure as params_pytree. JAX Arrays
            corresponding to frozen keys are marked False, all others are True.

    """
    # Create initial mask: True for all JAX arrays, False otherwise
    trainable_mask = jax.tree_util.tree_map(
        lambda x: isinstance(x, Array), params_pytree
    )

    # If no keys to freeze or not a dict, return the default mask
    if not keys_to_freeze or not isinstance(params_pytree, dict):
        return trainable_mask

    # Create mutable copy and freeze specified keys
    frozen_mask = dict(trainable_mask)
    for key in keys_to_freeze:
        if key in frozen_mask:
            # Recursively set all values under this key to False
            frozen_mask[key] = jax.tree_util.tree_map(
                lambda _: False, params_pytree[key]
            )
        else:
            logger.warning(
                "Key '%s' not found in params PyTree. Available keys: %s",
                key,
                list(params_pytree.keys()),
            )

    return frozen_mask


def _compute_loss_and_aux(
    local_energy_components: tuple, include_pv: bool = False
) -> tuple[Array, AuxiliaryLossData]:
    """Compute loss and auxiliary data from local energy components.

    Shared logic between NVT and NPT ensembles for computing the final loss
    and collecting auxiliary data.

    Args:
        local_energy_components: Tuple of (kinetic, ewald_ee, ewald_ei, ewald_ii)
            or (kinetic, ewald_ee, ewald_ei, ewald_ii, pv) for NPT.
        include_pv: Whether the tuple includes a PV term (NPT ensemble).

    Returns:
        Tuple of (loss, aux_data) where loss is a scalar and aux_data contains
        variance, energy components, etc.

    """
    if include_pv:
        kinetic_energy, ewald_ee, ewald_ei, ewald_ii, pv_term = local_energy_components
        ewald_total = ewald_ee + ewald_ei + ewald_ii + pv_term
    else:
        kinetic_energy, ewald_ee, ewald_ei, ewald_ii = local_energy_components
        ewald_total = ewald_ee + ewald_ei + ewald_ii
        pv_term = jnp.zeros_like(ewald_total)

    # Compute total local energy
    local_energy = kinetic_energy + ewald_total
    mean_local_energy = jnp.mean(local_energy)

    # Compute loss and variance across devices
    pmean_loss = constants.pmean(mean_local_energy)
    variance = constants.pmean(
        jnp.mean(jnp.abs(local_energy) ** 2) - jnp.abs(mean_local_energy.real) ** 2
    )
    loss = pmean_loss.real

    aux_data = AuxiliaryLossData(
        variance=variance,
        local_energy=local_energy,
        imaginary=pmean_loss.imag,
        kinetic=kinetic_energy,
        ewald=ewald_total,
        ewald_ee=ewald_ee,
        ewald_ei=ewald_ei,
        ewald_ii=ewald_ii,
        pv=pv_term,
    )

    return loss, aux_data


def _compute_jvp_gradient(
    batch_network: Callable,
    primals: tuple,
    tangents: tuple,
    clip_diff: Array,
) -> Array:
    """Compute gradient dot product using JVP for variance reduction.

    Shared logic for computing the gradient term in both NVT and NPT ensembles.

    Args:
        batch_network: Batched network function.
        primals: Tuple of (params, data).
        tangents: Tuple of (params_tangent, data_tangent).
        clip_diff: Clipped local energy differences.

    Returns:
        Scalar gradient dot product.

    """
    psi_primal, psi_tangent = jax.jvp(batch_network, primals, tangents)
    conj_psi_tangent = jnp.conjugate(psi_tangent)
    conj_psi_primal = jnp.conjugate(psi_primal)

    # Register with KFAC
    kfac_jax.register_normal_predictive_distribution(conj_psi_primal.real[:, None])

    return jnp.mean((clip_diff * conj_psi_tangent).real)


def make_loss_nvt(
    network: Callable,
    batch_network: Callable,
    simulation_cell,
    clip_local_energy: float = 5.0,
    clip_type: str = "real",
    mode: str = "for",
    nuclear_treatment: str = "quantum",
    is_deuterium: bool = False,
    partition_number: int = 3,
) -> Callable[[dict, Array], tuple[Array, AuxiliaryLossData]]:
    """Creates a loss function for NVT ensemble wavefunction optimization.

    This function constructs a loss function suitable for variational Monte Carlo
    optimization of the wavefunction. It supports various modes of operation and
    includes stability improvements through local energy clipping.

    Args:
        network: Unbatched network function computing log|ψ|.
        batch_network: Batched version of network function.
        simulation_cell: PySCF cell object with system information.
        clip_local_energy: Local energy clipping threshold. Defaults to 5.0.
        clip_type: Type of energy clipping. Options are 'real' for Cartesian
            clipping or 'complex' for polar clipping. Defaults to 'real'.
        mode: Method for computing local energy. Options are 'for' for sequential
            (memory efficient), 'hessian' for parallel (faster but memory intensive),
            or 'partition' for hybrid approach. Defaults to 'for'.
        nuclear_treatment: Type of quantum model. Options are 'quantum' for Neural
            Quantum Embedding or 'fixed' for fixed nuclei. Defaults to 'quantum'.
        is_deuterium: Whether to average over isotope configurations.
            Defaults to False.
        partition_number: Number of partitions for 'partition' mode. Must divide
            (dim * n_electrons). Defaults to 3.

    Returns:
        Callable: A function that computes loss and auxiliary data given parameters
            and electron positions.

    """
    # Create appropriate local energy calculator based on nuclear treatment
    if nuclear_treatment == "quantum":
        logger.info("is_deuterium: %s", is_deuterium)
        el_class = hamiltonian.LocalEnergy_quantum(
            simulation_cell=simulation_cell,
            mode=mode,
            is_deuterium=is_deuterium,
            partition_number=partition_number,
        )
    elif nuclear_treatment == "fixed":
        el_class = hamiltonian.LocalEnergy(
            simulation_cell=simulation_cell,
            mode=mode,
            partition_number=partition_number,
            nuclear_treatment=nuclear_treatment,
        )
    else:
        raise ValueError(
            f"Unknown nuclear_treatment: '{nuclear_treatment}'. "
            f"Must be 'quantum' or 'fixed'."
        )

    el_fun = el_class.local_energy_separate(network)
    batch_local_energy = jax.vmap(el_fun, in_axes=(None, 0), out_axes=0)

    @jax.custom_jvp
    def total_energy(params, data):
        """Compute total energy and auxiliary data for NVT ensemble.

        Args:
            params: Dictionary of model parameters including network weights.
            data: Batch of electron coordinates with shape
                [batch_size, n_electrons * n_dim].

        Returns:
            tuple: A tuple (loss, aux_data) where:
                - loss (float): Real part of mean local energy (scalar).
                - aux_data (AuxiliaryLossData): Contains variance, energies, etc.

        """
        # Compute energy components and aggregate into loss and aux data
        energy_components = batch_local_energy(params, data)
        return _compute_loss_and_aux(energy_components, include_pv=False)

    @total_energy.defjvp
    def total_energy_jvp(primals, tangents):
        """Custom Jacobian-vector product for total energy with variance reduction.

        Args:
            primals: Tuple of (params, data) - inputs to total_energy.
            tangents: Tuple of (params_tangent, data_tangent) - tangent vectors.

        Returns:
            tuple: A tuple (primals_out, tangents_out) where:
                - primals_out: (loss, aux_data) from forward pass.
                - tangents_out: (gradient_dot_product, aux_data).

        """
        loss, aux_data = total_energy(*primals)

        # Compute clipped energy differences for variance reduction
        diff = aux_data.local_energy - loss
        clip_diff = clip_local_energy_func(diff, clip_local_energy, clip_type)

        # Compute gradient using shared JVP logic
        tangents_dot = _compute_jvp_gradient(
            batch_network, primals, tangents, clip_diff
        )

        return (loss, aux_data), (tangents_dot, aux_data)

    return total_energy


def make_loss_npt(
    network: Callable,
    batch_network: Callable,
    simulation_cell,
    clip_local_energy: float = 5.0,
    clip_type: str = "real",
    mode: str = "for",
    nuclear_treatment: str = "quantum",
    is_deuterium: bool = False,
    partition_number: int = 3,
    target_pressure: float = 100.0,
    lattice_config: CrystalLatticeConfig | None = None,
) -> Callable[[dict, Array], tuple[Array, AuxiliaryLossData]]:
    """Creates a loss function for NPT ensemble wavefunction optimization.

    Generates a loss function for variational Monte Carlo optimization with constant
    pressure (NPT ensemble). The loss includes the Gibbs free energy G = E + PV.

    Args:
        network: Unbatched logdet function of wavefunction.
        batch_network: Batched logdet function of wavefunction.
        simulation_cell: PySCF object of simulation cell.
        clip_local_energy: Clip window width of local energy. Defaults to 5.0.
        clip_type: Specify the clip style. Options are 'real' for Cartesian style
            clipping or 'complex' for polar style clipping. Defaults to 'real'.
        mode: Specify the evaluation style of local energy. Options are 'for' to
            calculate the Laplacian of each electron one by one (slow but saves GPU
            memory), 'hessian' for highly parallelized mode (fast but requires more
            GPU memory), or 'partition' for a moderate approach. Defaults to 'for'.
        nuclear_treatment: Type of nuclear treatment. Currently not used.
            Defaults to 'quantum'.
        is_deuterium: Whether to average over isotope configurations.
            Defaults to False.
        partition_number: Number of partitions for 'partition' mode. Must be
            divisible by (dim * number of electrons). The smaller the faster, but
            requires more memory. Defaults to 3.
        target_pressure: Target pressure in GPa. Defaults to 100.0.
        lattice_config: Configuration for lattice parameterization.
            Defaults to CrystalLatticeConfig(mode="angle").

    Returns:
        Callable: The loss function that computes Gibbs free energy and auxiliary data.

    """
    del nuclear_treatment
    logger.info("is_deuterium: %s", is_deuterium)
    el_class = hamiltonian.LocalEnthalpy_quantum(
        simulation_cell=simulation_cell,
        target_pressure=target_pressure,
        mode=mode,
        lattice_config=lattice_config,
        is_deuterium=is_deuterium,
        partition_number=partition_number,
    )
    el_fun = el_class.local_energy_separate(network)

    batch_local_energy = jax.vmap(el_fun, in_axes=(None, 0), out_axes=0)

    @jax.custom_jvp
    def total_gibbs(params, data):
        """Compute Gibbs free energy (enthalpy) and auxiliary data for NPT ensemble.

        The Gibbs free energy includes the PV work term: G = E + PV

        Args:
            params: Dictionary of model parameters including network weights and
                cell parameters.
            data: Batch of electron coordinates with shape
                [batch_size, n_electrons * n_dim].

        Returns:
            tuple: A tuple (loss, aux_data) where:
                - loss (float): Real part of mean Gibbs free energy (scalar).
                - aux_data (AuxiliaryLossData): Contains variance, energies,
                    PV term, etc.

        """
        # Compute energy components including PV term and aggregate
        energy_components = batch_local_energy(params, data)
        return _compute_loss_and_aux(energy_components, include_pv=True)

    @total_gibbs.defjvp
    def total_gibbs_jvp(primals, tangents):
        """Custom Jacobian-vector product for Gibbs free energy with cell freezing.

        The cell parameters are frozen during gradient computation to ensure
        stability in lattice optimization.

        Args:
            primals: Tuple of (params, data) - inputs to total_gibbs.
            tangents: Tuple of (params_tangent, data_tangent) - tangent vectors.

        Returns:
            tuple: A tuple (primals_out, tangents_out) where:
                - primals_out: (loss, aux_data) from forward pass.
                - tangents_out: (gradient_dot_product, aux_data).

        """
        params, data = primals
        params_tangent, data_tangent = tangents

        # Freeze cell parameters to prevent unstable lattice updates
        frozen_mask = create_frozen_mask(params_tangent, "cell")
        masked_params_tangent = jax.tree_util.tree_map(
            operator.mul, params_tangent, frozen_mask
        )
        masked_tangent = (masked_params_tangent, data_tangent)

        # Compute loss and clipped energy differences
        loss, aux_data = total_gibbs(params, data)
        diff = aux_data.local_energy - loss
        clip_diff = clip_local_energy_func(diff, clip_local_energy, clip_type)

        tangents_dot = 2.0 * _compute_jvp_gradient(
            batch_network, primals, masked_tangent, clip_diff
        )

        return (loss, aux_data), (tangents_dot, aux_data)

    return total_gibbs


def setup_evaluate_loss(
    cfg: base_config.BornFreeConfig,
    networks: dict[str, Callable],
    batched_networks: dict[str, Callable],
    simulation_cell: PyscfCell,
) -> Callable:
    """Creates the loss evaluation function based on the ensemble type.

    Args:
        cfg: The configuration object for the simulation.
        networks: A dictionary of the base network functions.
        batched_networks: A dictionary of the batched network functions.
        simulation_cell: The PySCF cell object.

    Returns:
        The loss evaluation function.

    Raises:
        ValueError: If the nuclear treatment is not supported.

    """
    logger.info("Setting up loss evaluation...")

    # Check for supported nuclear treatments generally
    if cfg.nuclear_treatment not in ["fixed", "quantum"]:
        raise ValueError(f"Unsupported nuclear treatment: {cfg.nuclear_treatment}")

    if cfg.ensemble == "NVT":
        return make_loss_nvt(
            network=networks["log"].apply,
            batch_network=batched_networks["log"],
            simulation_cell=simulation_cell,
            clip_local_energy=cfg.optim.clip_el,
            clip_type=cfg.optim.clip_type,
            mode=cfg.optim.laplacian_mode,
            nuclear_treatment=cfg.nuclear_treatment,
            is_deuterium=cfg.crystal.is_deuterium,
            partition_number=cfg.optim.partition_number,
        )
    elif cfg.ensemble == "NPT":
        if cfg.nuclear_treatment != "quantum":
            raise ValueError(
                f"NPT ensemble currently only supports 'quantum' nuclear treatment, got {cfg.nuclear_treatment}"
            )

        return make_loss_npt(
            network=networks["log"].apply,
            batch_network=batched_networks["log"],
            simulation_cell=simulation_cell,
            clip_local_energy=cfg.optim.clip_el,
            clip_type=cfg.optim.clip_type,
            mode=cfg.optim.laplacian_mode,
            nuclear_treatment=cfg.nuclear_treatment,
            is_deuterium=cfg.crystal.is_deuterium,
            partition_number=cfg.optim.partition_number,
            target_pressure=cfg.target_pressure,
            lattice_config=cfg.crystal.lattice,
        )
    else:
        raise ValueError(f"Unsupported ensemble type: {cfg.ensemble}")
