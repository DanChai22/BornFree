# Copyright (c) 2025 Shengdu Chai
#
# Licensed under the Apache License, Version 2.0.

import logging
from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
from jax import Array

from BornFree import constants, distance
from BornFree.loss import AuxiliaryLossData

logger = logging.getLogger(__name__)


@chex.dataclass
class AnnealingState:
    """State of the cell parameters.

    Attributes:
        params: A dictionary of parameters for the neural network and the system.
        key: JAX random number generator key.
        enthalpy: The enthalpy of the system.
        aux_data: Auxiliary data returned by the loss function.
        num_accepts: The number of accepted MCMC moves.

    """

    params: dict
    key: chex.PRNGKey
    enthalpy: Any
    aux_data: AuxiliaryLossData
    num_accepts: int | Array


@chex.dataclass
class MCMCState:
    """State of the MCMC chain.

    Attributes:
        data: The current MCMC configurations (electron and/or nuclei positions).
            Shape: (batch_size, ndim).
        key: JAX random number generator key.
        logprob: The log probability of the current state.
            Shape: (batch_size,).
        num_accepts: The number of accepted MCMC moves.
            Can be either integer or array for multiple types of moves.

    """

    data: Array
    key: chex.PRNGKey
    logprob: Array
    num_accepts: int | Array


@chex.dataclass
class MCMCConfig:
    """Configuration for MCMC sampling.

    Attributes:
        batch_per_device: Number of parallel chains per device.
        latvec: The lattice vectors defining the simulation cell.
            Shape: (3, 3).
        natom: The number of atoms in the system.
        steps: The number of MCMC steps to perform.
        iterations: Number of iterations for multi-step methods.
        atoms: The atomic positions. Shape: (natoms, 3).
        importance_sampling: Function for computing drift velocity (optional).

    """

    batch_per_device: int
    latvec: Array
    natom: int
    steps: int = 10
    iterations: int = 10
    atoms: Array | None = None
    importance_sampling: Callable | None = None
    current_temp: float | None = 1000.0
    annealing_steps: int | None = 10


def _log_prob_gaussian(x: Array, mu: Array, sigma: Array) -> Array:
    """Calculates the log probability of a multivariate Gaussian with diagonal covariance.

    Used for proposal distributions in MCMC and for initial state generation.

    Args:
        x: Positions to evaluate
            Shape: (batch, nelectron, 1, ndim)
        mu: Means of Gaussian distribution
            Shape: Same as x or broadcastable to x
        sigma: Standard deviations
            Shape: Same as x or broadcastable to x

    Returns:
        Log probabilities
            Shape: (batch, nelectron, 1, 1)

    """
    numer = jnp.sum(-0.5 * ((x - mu) ** 2) / (sigma**2), axis=[1, 2, 3])
    denom = x.shape[-1] * jnp.sum(jnp.log(sigma), axis=[1, 2, 3])
    return numer - denom


def _harmonic_mean(x: Array, atoms: Array) -> Array:
    """Calculates the harmonic mean of electron-nuclear distances.

    Used for adaptive proposal widths and initial state generation.

    Args:
        x: Electron positions
            Shape: (batch, nelectrons, 1, ndim)
        atoms: Nuclear positions
            Shape: (natoms, ndim)

    Returns:
        Harmonic mean distances
            Shape: (batch, nelectrons, 1, 1)

    """
    ae = x - atoms[None, ...]
    r_ae = jnp.linalg.norm(ae, axis=-1, keepdims=True)
    return 1.0 / jnp.mean(1.0 / r_ae, axis=-2, keepdims=True)


def limdrift(g: Array, cutoff: float = 1.0) -> Array:
    """Limits the drift velocity for stable importance sampling.

    Implements the algorithm from Umrigar et al. (1993) to prevent
    divergent drift velocities in regions of small wavefunction amplitude.

    Args:
        g: Gradient of log|𝜓|
            Shape: (..., ndim)
        cutoff: Maximum allowed drift magnitude
            Default: 1.0 atomic units

    Returns:
        Limited drift vector with same shape as input

    """
    gnorm = jnp.linalg.norm(g, axis=-1, keepdims=True)
    gunit = g / (gnorm + 1e-10)
    return gunit * gnorm / (1 + gnorm / cutoff)


def propose_new_geometry(params: dict, key: chex.PRNGKey, move_width: float) -> dict:
    """Proposes a new geometry by adding random noise to cell parameters.

    Args:
        params: A dictionary of parameters for the neural network and the system.
        key: JAX random number generator key.
        move_width: The standard deviation of the Gaussian noise used for proposals.

    Returns:
        Proposed parameters with perturbed cell.

    """
    # Deep copy to avoid modifying original parameters
    proposed_params = jax.tree_util.tree_map(lambda x: x, params)

    cell_params = proposed_params["cell"]
    noise = move_width * jax.random.normal(key, shape=cell_params.shape)
    proposed_cell = cell_params + noise
    proposed_cell = constants.pmean(proposed_cell)
    proposed_params["cell"] = proposed_cell
    return proposed_params


class MetropolisHastings:
    """Metropolis-Hastings MCMC implementation with specialized moves.

    Provides methods for:
    1. Standard all-particle updates
    2. Atom-only updates
    3. Electron-only updates

    Each method properly handles periodic boundary conditions and
    maintains detailed balance.
    """

    @staticmethod
    def accept(
        x1: Array,
        x2: Array,
        lp_1: Array,
        lp_2: Array,
        ratio: Array,
        key: chex.PRNGKey,
        num_accepts: int,
    ) -> tuple[Array, Any, Array, int]:
        """Executes the Metropolis-Hastings accept/reject step.

        Implements the standard MH acceptance criterion:
        P(accept) = min(1, |𝜓(x')|²/|𝜓(x)|²)

        Args:
            x1: The current state.
                Shape: (batch_size, n_particles * ndim).
            x2: The proposed state.
                Shape: Same as x1.
            lp_1: The log probability of the current state.
                Shape: (batch_size,).
            lp_2: The log probability of the proposed state.
                Shape: Same as lp_1.
            ratio: The log acceptance ratio log(|𝜓(x')|²/|𝜓(x)|²).
                Shape: Same as lp_1.
            key: JAX random number generator key.
            num_accepts: The number of accepted MCMC moves.

        Returns:
            Tuple containing:
            - New state (accepted or unchanged)
            - New RNG key
            - New log probability
            - Updated acceptance count

        """
        key, subkey = jax.random.split(key)
        rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
        cond = ratio > rnd
        x_new = jnp.where(cond[..., None], x2, x1)
        lp_new = jnp.where(cond, lp_2, lp_1)
        num_accepts += jnp.sum(cond)
        return x_new, key, lp_new, num_accepts

    @staticmethod
    def accept_atom(
        x1: Array,
        x2: Array,
        lp_1: Array,
        lp_2: Array,
        ratio: Array,
        key: chex.PRNGKey,
        num_accepts: int,
        natom: int,
    ) -> tuple[Array, Any, Array, int]:
        """Executes MH accept/reject step for atom-only moves.

        Similar to standard accept() but only updates atomic positions
        while keeping electron positions fixed.

        Args:
            x1: The current state with atoms first.
                Shape: (batch_size, (n_atoms + n_elec) * ndim).
            x2: The proposed state (only atoms different).
                Shape: Same as x1.
            lp_1: The log probability of the current state.
            lp_2: The log probability of the proposed state.
            ratio: The log acceptance ratio.
            key: JAX random number generator key.
            num_accepts: The number of accepted MCMC moves.
            natom: The number of atoms in the system.

        Returns:
            Same as accept() but with only atomic positions potentially updated

        """
        key, subkey = jax.random.split(key)
        rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
        cond = ratio > rnd
        x1_fixed, x1_update = x1[:, natom * 3 :], x1[:, : natom * 3]
        x2_update = x2[:, : natom * 3]
        x_new_update = jnp.where(cond[..., None], x2_update, x1_update)
        x_new = jnp.hstack((x_new_update, x1_fixed))
        lp_new = jnp.where(cond, lp_2, lp_1)
        num_accepts += jnp.sum(cond)
        return x_new, key, lp_new, num_accepts

    @staticmethod
    def accept_elec(
        x1: Array,
        x2: Array,
        lp_1: Array,
        lp_2: Array,
        ratio: Array,
        key: chex.PRNGKey,
        num_accepts: int,
        natom: int,
    ) -> tuple[Array, Any, Array, int]:
        """Executes MH accept/reject step for electron-only moves.

        Similar to standard accept() but only updates electron positions
        while keeping atomic positions fixed.

        Args:
            x1: The current state with atoms first.
                Shape: (batch_size, (n_atoms + n_elec) * ndim).
            x2: The proposed state (only electrons different).
                Shape: Same as x1.
            lp_1: The log probability of the current state.
            lp_2: The log probability of the proposed state.
            ratio: The log acceptance ratio.
            key: JAX random number generator key.
            num_accepts: The number of accepted MCMC moves.
            natom: The number of atoms in the system.

        Returns:
            Same as accept() but with only electron positions potentially updated

        """
        key, subkey = jax.random.split(key)
        rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
        cond = ratio > rnd
        x1_fixed, x1_update = x1[:, : natom * 3], x1[:, natom * 3 :]
        x2_update = x2[:, natom * 3 :]
        x_new_update = jnp.where(cond[..., None], x2_update, x1_update)
        x_new = jnp.hstack((x1_fixed, x_new_update))
        lp_new = jnp.where(cond, lp_2, lp_1)
        num_accepts += jnp.sum(cond)
        return x_new, key, lp_new, num_accepts

    def accept_cell(
        c1: Array,
        c2: Array,
        g_1: Array,
        g_2: Array,
        aux_1: Any,
        aux_2: Any,
        ratio: Array,
        key: chex.PRNGKey,
        num_accepts: int,
    ) -> tuple[Array, Array, Array, int]:
        """Executes MH accept/reject step for cell parameter moves in NPT ensemble."""
        key, subkey = jax.random.split(key)
        rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
        cond = ratio > rnd
        c_new = jnp.where(cond[..., None], c2, c1)
        g_new = jnp.where(cond, g_2, g_1)
        aux_new = jax.tree_util.tree_map(lambda a1, a2: jnp.where(cond, a2, a1), aux_1, aux_2)
        num_accepts += jnp.sum(cond)
        return c_new, key, g_new, aux_new, num_accepts


class MCMCSampler:
    """Main MCMC sampler implementation.

    Provides methods for:
    1. Standard Metropolis-Hastings sampling
    2. Importance sampling with drift
    3. Gibbs sampling for atoms/electrons

    Each method can be configured for different move types and
    can handle both atomic and electronic degrees of freedom.
    """

    def __init__(self, config: MCMCConfig):
        """Initializes the MCMC sampler.

        Args:
            config: The configuration object for the simulation.

        """
        self.config = config

    def importance_sampling(
        self,
        params: Any,
        f: Callable,
        state: MCMCState,
        stddev: float = 0.02,
        update_atom: bool | None = None,
    ) -> MCMCState:
        """Performs an importance sampling update step.

        Implements drift-diffusion sampling using the gradient of log|𝜓|²
        as a guiding function:
        x' = x + τ∇log|𝜓|² + √(2τ)η
        where τ is the time step and η is Gaussian noise.

        Args:
            params: A dictionary of parameters for the neural network and the system.
            f: Function returning (log|𝜓|², ∇log|𝜓|²).
            state: The current MCMC state.
            stddev: The standard deviation of the Gaussian noise used for MCMC proposals.
            update_atom: Whether to update atomic positions.

        Returns:
            Updated MCMC state after importance sampling step

        """
        key, subkey = jax.random.split(state.key)
        x1 = state.data

        if self.config.atoms is None:  # symmetric proposal
            _, grad = f(params, x1)
            grad = limdrift(grad)
            gauss = stddev * jax.random.normal(subkey, shape=x1.shape)
            x2 = x1 + gauss + stddev**2 * grad
            x2, _ = distance.enforce_pbc(self.config.latvec, x2)

            # Compute reverse move
            lpsi_2, new_grad = f(params, x2)
            lp_2 = 2 * lpsi_2
            new_grad = limdrift(new_grad)
            forward = jnp.sum(gauss**2, axis=-1)
            backward = jnp.sum((gauss + stddev**2 * (grad + new_grad)) ** 2, axis=-1)
            lp_2 = lp_2 + 1 / (2 * stddev**2) * (forward - backward)

            ratio = lp_2 - state.logprob

        else:  # asymmetric proposal
            n = x1.shape[0]
            x1_reshaped = jnp.reshape(x1, [n, -1, 1, 3])
            hmean1 = _harmonic_mean(x1_reshaped, self.config.atoms)

            x2_reshaped = x1_reshaped + stddev * hmean1 * jax.random.normal(subkey, shape=x1_reshaped.shape)
            lp_2 = 2.0 * f(params, x2_reshaped)
            hmean2 = _harmonic_mean(x2_reshaped, self.config.atoms)

            lq_1 = _log_prob_gaussian(x1_reshaped, x2_reshaped, stddev * hmean1)
            lq_2 = _log_prob_gaussian(x2_reshaped, x1_reshaped, stddev * hmean2)
            ratio = lp_2 + lq_2 - state.logprob - lq_1

            x1 = jnp.reshape(x1_reshaped, [n, -1])
            x2 = jnp.reshape(x2_reshaped, [n, -1])

        if update_atom is not None:
            if update_atom:
                x_new, key, lp_new, num_accepts = MetropolisHastings.accept_atom(
                    x1,
                    x2,
                    state.logprob,
                    lp_2,
                    ratio,
                    key,
                    state.num_accepts,
                    self.config.natom,
                )
            else:
                x_new, key, lp_new, num_accepts = MetropolisHastings.accept_elec(
                    x1,
                    x2,
                    state.logprob,
                    lp_2,
                    ratio,
                    key,
                    state.num_accepts,
                    self.config.natom,
                )
        else:
            x_new, key, lp_new, num_accepts = MetropolisHastings.accept(
                x1, x2, state.logprob, lp_2, ratio, key, state.num_accepts
            )

        return MCMCState(data=x_new, key=key, logprob=lp_new, num_accepts=num_accepts)

    def metropolis_step(
        self,
        params: Any,
        f: Callable,
        state: MCMCState,
        stddev: float = 0.02,
        update_atom: bool | None = None,
    ) -> MCMCState:
        """Perform Metropolis update step.

        Args:
            params: A dictionary of parameters for the neural network and the system.
            f: Function to compute log probability.
            state: The current MCMC state.
            stddev: The standard deviation of the Gaussian noise used for MCMC proposals.
            update_atom: Whether to update atom positions.

        Returns:
            Updated MCMC state

        """
        del update_atom
        key, subkey = jax.random.split(state.key)
        x1 = state.data
        if self.config.atoms is None:
            # Generate proposal
            x2 = x1 + stddev * jax.random.normal(subkey, shape=x1.shape)
            x2, _ = distance.enforce_pbc(self.config.latvec, x2)

            # Compute acceptance ratio
            lp_2 = 2.0 * f(params, x2)
            ratio = lp_2 - state.logprob
        else:
            n = x1.shape[0]
            x1 = jnp.reshape(x1, [n, -1, 1, 3])
            hmean1 = _harmonic_mean(x1, self.config.atoms)  # harmonic mean of distances to nuclei

            x2 = x1 + stddev * hmean1 * jax.random.normal(subkey, shape=x1.shape)
            x2 = jnp.reshape(x2, [n, -1])
            x2, _ = distance.enforce_pbc(self.config.latvec, x2)
            lp_2 = 2.0 * f(params, x2)

            x2 = jnp.reshape(x2, [n, -1, 1, 3])
            hmean2 = _harmonic_mean(x2, self.config.atoms)  # needed for probability of reverse jump

            lq_1 = _log_prob_gaussian(x1, x2, stddev * hmean1)  # forward probability
            lq_2 = _log_prob_gaussian(x2, x1, stddev * hmean2)  # reverse probability
            ratio = lp_2 + lq_2 - state.logprob - lq_1

            x1 = jnp.reshape(x1, [n, -1])
            x2 = jnp.reshape(x2, [n, -1])

        x_new, key, lp_new, num_accepts = MetropolisHastings.accept(
            x1, x2, state.logprob, lp_2, ratio, key, state.num_accepts
        )

        return MCMCState(data=x_new, key=key, logprob=lp_new, num_accepts=num_accepts)

    def metropolis_step_joint(
        self,
        params: Any,
        f: Callable,
        state: MCMCState,
        stddev: tuple[float, float] = (0.0002, 0.02),
        update_atom: bool | None = None,
    ) -> MCMCState:
        """Perform Metropolis update step for joint moves."""
        del update_atom
        key, subkey = jax.random.split(state.key)
        x1 = state.data
        atom_stddev, elec_stddev = stddev
        if self.config.atoms is None:
            x1_atom, x1_elec = (
                x1[:, : self.config.natom * 3],
                x1[:, self.config.natom * 3 :],
            )
            x2_atom = x1_atom + atom_stddev * jax.random.normal(subkey, shape=x1_atom.shape)  # proposal
            x2_elec = x1_elec + elec_stddev * jax.random.normal(subkey, shape=x1_elec.shape)  # proposal
            x2 = jnp.hstack((x2_atom, x2_elec))
            x2, _ = distance.enforce_pbc(self.config.latvec, x2)
            lp_2 = 2.0 * f(params, x2)  # log prob of proposal
            ratio = lp_2 - state.logprob
        else:
            raise NotImplementedError("Joint mcmc asymmetric moves are not implemented yet")

        x_new, key, lp_new, num_accepts = MetropolisHastings.accept(
            x1, x2, state.logprob, lp_2, ratio, key, state.num_accepts
        )

        return MCMCState(data=x_new, key=key, logprob=lp_new, num_accepts=num_accepts)

    def metropolis_step_gibbs(
        self,
        params: Any,
        f: Callable,
        state: MCMCState,
        stddev: tuple[float] = (0.02),
        update_atom: bool | None = None,
    ) -> MCMCState:
        """Perform Metropolis update step for Gibbs moves."""
        key, subkey = jax.random.split(state.key)
        x1 = state.data
        if self.config.atoms is None:  # symmetric proposal, same stddev everywhere
            x1_atom, x1_elec = (
                x1[:, : self.config.natom * 3],
                x1[:, self.config.natom * 3 :],
            )
            if update_atom:
                x2_atom = x1_atom + stddev * jax.random.normal(
                    subkey, shape=x1_atom.shape, dtype=x1_atom.dtype
                )  # proposal
                x2_atom, _ = distance.enforce_pbc(self.config.latvec, x2_atom)
                # reduce the electrons into the simulation cell.
                x2 = jnp.hstack((x2_atom, x1_elec))
                lp_2 = 2.0 * f(params, x2)  # log prob of proposal
                ratio = lp_2 - state.logprob
            else:
                x2_elec = x1_elec + stddev * jax.random.normal(subkey, shape=x1_elec.shape, dtype=x1_elec.dtype)
                x2_elec, _ = distance.enforce_pbc(self.config.latvec, x2_elec)
                x2 = jnp.hstack((x1_atom, x2_elec))
                lp_2 = 2.0 * f(params, x2)  # log prob of proposal
                ratio = lp_2 - state.logprob
        else:  # asymmetric proposal, stddev propto harmonic mean of nuclear distances
            raise NotImplementedError("Gibbs moves for asymmetric systems are not implemented yet")

        x_new, key, lp_new, num_accepts = MetropolisHastings.accept(
            x1, x2, state.logprob, lp_2, ratio, key, state.num_accepts
        )
        return MCMCState(data=x_new, key=key, logprob=lp_new, num_accepts=num_accepts)

    def metropolis_step_cell(
        self,
        data: Array,
        f: Callable,
        state: AnnealingState,
        cell_annealing_width: tuple[float] = (0.02),
        mcmc_width: tuple[float] = (0.02),
        current_temp: float = 1000.0,
    ) -> AnnealingState:
        """Perform Metropolis update step for cell moves."""
        key, subkey = jax.random.split(state.key)
        params_2 = propose_new_geometry(state.params, subkey, cell_annealing_width)
        key, subkey = jax.random.split(key)
        g_2, aux_data_2 = f(params_2, data, subkey, mcmc_width)
        ratio = -(g_2 - state.enthalpy) / current_temp
        cell_new, key, g_new, aux_new, num_accepts = MetropolisHastings.accept_cell(
            state.params["cell"],
            params_2["cell"],
            state.enthalpy,
            g_2,
            state.aux_data,
            aux_data_2,
            ratio,
            key,
            state.num_accepts,
        )
        params_new = {**state.params, "cell": cell_new}
        return AnnealingState(
            params=params_new,
            key=key,
            enthalpy=g_new,
            aux_data=aux_new,
            num_accepts=num_accepts,
        )


class MCMCStepFactory:
    """Factory class for creating MCMC step functions."""

    @staticmethod
    def create_step(config: MCMCConfig, batch_mcmc_network: Callable, step_type: str = "standard") -> Callable:
        """Create an MCMC step function.

        Args:
            config: The configuration object for the simulation.
            batch_mcmc_network: Network function for computing log probabilities.
            step_type: Type of MCMC step ('electron_only', 'joint', 'gibbs').

        Returns:
            MCMC step function

        """
        sampler = MCMCSampler(config)

        if step_type == "electron_only":
            logger.info("Using electron only sampling")
            return MCMCStepFactory._create_electron_only_step(sampler, batch_mcmc_network, config)
        elif step_type == "joint":
            logger.info("Using joint sampling")
            return MCMCStepFactory._create_joint_step(sampler, batch_mcmc_network, config)
        elif step_type == "gibbs":
            logger.info("Using Gibbs sampling")
            return MCMCStepFactory._create_gibbs_step(sampler, batch_mcmc_network, config)
        elif step_type == "annealing":
            logger.info("Using annealing sampling")
            return MCMCStepFactory._create_annealing_step(sampler, batch_mcmc_network, config)
        else:
            raise ValueError(f"Unknown MCMC step type: {step_type}")

    @staticmethod
    def _create_electron_only_step(sampler: MCMCSampler, batch_mcmc_network: Callable, config: MCMCConfig) -> Callable:
        """Create electron MCMC step function."""
        if config.importance_sampling is not None:
            logger.info("Using importance sampling")
            func = jax.vmap(
                jax.value_and_grad(config.importance_sampling, argnums=1),
                in_axes=(None, 0),
            )
            sampler_func = sampler.importance_sampling
        else:
            func = batch_mcmc_network
            logger.info("Using Metropolis sampling")
            sampler_func = sampler.metropolis_step

        @jax.jit
        def mcmc_step(params: Any, data: Array, key: chex.PRNGKey, width: Any) -> tuple[Array, Array]:
            """Perform standard MCMC step.

            Args:
                params: parameters to pass to the network.
                data: (batched) MCMC configurations to pass to the network.
                key: RNG state.
                width: standard deviation to use in the move proposal.

            Returns:
                (data, pmove), where data is the updated MCMC configurations, key the
                updated RNG state and pmove the average probability a move was accepted.

            """
            state = MCMCState(
                data=data,
                key=key,
                logprob=2.0 * batch_mcmc_network(params, data),
                num_accepts=0,
            )

            def step_fn(i: int, state: MCMCState) -> Callable[[Any, Any, MCMCState, float, bool | None], MCMCState]:
                return sampler_func(params, func, state, width)

            final_state = jax.lax.fori_loop(0, config.steps, step_fn, state)
            pmove = jnp.sum(final_state.num_accepts) / (config.steps * config.batch_per_device)
            pmove = constants.pmean(pmove)
            return final_state.data, [pmove]

        return mcmc_step

    @staticmethod
    def _create_joint_step(sampler: MCMCSampler, batch_mcmc_network: Callable, config: MCMCConfig) -> Callable:
        """Create joint MCMC step function."""
        if config.importance_sampling is not None:
            raise NotImplementedError("Importance sampling for joint moves is not implemented yet")
        else:
            func = batch_mcmc_network
            logger.info("Using Metropolis sampling")
            sampler_func = sampler.metropolis_step_joint

        @jax.jit
        def mcmc_step(params: Any, data: Array, key: chex.PRNGKey, width: tuple[float, float]) -> tuple[Array, Array]:
            """Perform joint MCMC step."""
            state = MCMCState(
                data=data,
                key=key,
                logprob=2.0 * batch_mcmc_network(params, data),
                num_accepts=0,
            )

            def step_fn(i: int, state: MCMCState) -> MCMCState:
                return sampler_func(params, func, state, width)

            final_state = jax.lax.fori_loop(0, config.steps, step_fn, state)
            pmove = jnp.sum(final_state.num_accepts) / (config.steps * config.batch_per_device)
            pmove = constants.pmean(pmove)
            return final_state.data, pmove

        return mcmc_step

    @staticmethod
    def _create_gibbs_step(sampler: MCMCSampler, batch_mcmc_network: Callable, config: MCMCConfig) -> Callable:
        """Create Gibbs sampling step function."""
        if config.importance_sampling is not None:
            raise NotImplementedError("Importance sampling for Gibbs moves is not implemented yet")
        else:
            func = batch_mcmc_network
            logger.info("Using Metropolis sampling")
            sampler_func = sampler.metropolis_step_gibbs

        @jax.jit
        def mcmc_step(params: Any, data: Array, key: chex.PRNGKey, width: tuple[float, float]) -> tuple[Array, Array]:
            atom_width, elec_width = width
            state_atom = MCMCState(
                data=data,
                key=key,
                logprob=2.0 * batch_mcmc_network(params, data),
                num_accepts=0,
            )
            state_elec = MCMCState(
                data=data,
                key=key,
                logprob=2.0 * batch_mcmc_network(params, data),
                num_accepts=0,
            )

            def elec_step_fn(i: int, state: MCMCState) -> MCMCState:
                return sampler_func(params, func, state, elec_width, update_atom=False)

            def atom_step_fn(i: int, state: MCMCState) -> MCMCState:
                return sampler_func(params, func, state, atom_width, update_atom=True)

            def sample_fn(i: int, state_atom: MCMCState, state_elec: MCMCState) -> tuple[MCMCState, MCMCState]:
                state_atom = jax.lax.fori_loop(0, config.steps, atom_step_fn, state_atom)
                state_elec = MCMCState(
                    data=state_atom.data,
                    key=state_atom.key,
                    logprob=state_atom.logprob,
                    num_accepts=state_elec.num_accepts,
                )
                state_elec = jax.lax.fori_loop(0, config.steps, elec_step_fn, state_elec)
                state_atom = MCMCState(
                    data=state_elec.data,
                    key=state_elec.key,
                    logprob=state_elec.logprob,
                    num_accepts=state_atom.num_accepts,
                )
                return state_atom, state_elec

            def sample_fn_fori(i, x):
                return sample_fn(i, *x)

            final_state_atom, final_state_elec = jax.lax.fori_loop(
                0, config.iterations, sample_fn_fori, (state_atom, state_elec)
            )
            pmove_atom = jnp.sum(final_state_atom.num_accepts) / (
                config.steps * config.batch_per_device * config.iterations
            )
            pmove_atom = constants.pmean(pmove_atom)
            pmove_elec = jnp.sum(final_state_elec.num_accepts) / (
                config.steps * config.batch_per_device * config.iterations
            )
            pmove_elec = constants.pmean(pmove_elec)
            pmove = [pmove_atom, pmove_elec]
            return final_state_atom.data, pmove

        return mcmc_step

    @staticmethod
    def _create_annealing_step(sampler: MCMCSampler, calculate_gibbs: Callable, config: MCMCConfig) -> Callable:
        """Create annealing MCMC step function."""
        sampler_func = sampler.metropolis_step_cell

        @jax.jit
        def mcmc_step(
            params: Any,
            data: Array,
            key: chex.PRNGKey,
            mcmc_width: tuple[float],
            cell_annealing_width: tuple[float],
        ) -> tuple[Array, Array]:
            """Perform standard MCMC step.

            Args:
                params: parameters to pass to the network.
                data: (batched) MCMC configurations to pass to the network.
                key: RNG state.
                mcmc_width: standard deviation to use in the move proposal.
                cell_annealing_width: width for cell parameter annealing moves.

            Returns:
                (data, pmove), where data is the updated MCMC configurations, key the
                updated RNG state and pmove the average probability a move was accepted.

            """
            key, subkey = jax.random.split(key)
            enthalpy, aux_data = calculate_gibbs(params, data, subkey, mcmc_width)
            state = AnnealingState(
                params=params,
                key=key,
                enthalpy=enthalpy,
                aux_data=aux_data,
                num_accepts=0,
            )

            def step_fn(
                i: int, state: AnnealingState
            ) -> Callable[
                [Array, Any, AnnealingState, tuple[float], tuple[float], float],
                AnnealingState,
            ]:
                return sampler_func(
                    data,
                    calculate_gibbs,
                    state,
                    cell_annealing_width,
                    mcmc_width,
                    config.current_temp,
                )

            final_state: AnnealingState = jax.lax.fori_loop(0, config.annealing_steps, step_fn, state)
            pmove = jnp.sum(final_state.num_accepts) / config.annealing_steps
            pmove = constants.pmean(pmove)
            return (
                final_state.params,
                [pmove, pmove],
                final_state.enthalpy,
                final_state.aux_data,
            )

        return mcmc_step
