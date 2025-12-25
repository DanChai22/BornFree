# Copyright (c) 2025 Shengdu Chai
#
# Licensed under the Apache License, Version 2.0.

import logging

import jax
import jax.numpy as jnp
import kfac_jax
from pyscf.pbc.gto import Cell as PyscfCell

from BornFree import base_config
from BornFree.mcmc import mcmc

logger = logging.getLogger(__name__)


# Geometric Cooling Strategy (Geometric Cooling / Exponential Cooling)
def geometric_cooling(initial_temp: float, k: int, beta: float) -> float:
    """Geometric (exponential) cooling strategy.

    This is the most commonly used annealing strategy.

    Args:
        initial_temp: Initial temperature T0.
        k: Current iteration step.
        beta: Cooling coefficient.

    Returns:
        New temperature at step k

    """
    return initial_temp * (beta**k)


# Linear Cooling Strategy (Linear Cooling)
def linear_cooling(initial_temp: float, k: int, beta: float) -> float:
    """Linear cooling strategy.

    Temperature decreases linearly from initial value to final value.

    Args:
        initial_temp: Initial temperature T0.
        k: Current iteration step.
        beta: Cooling rate.

    Returns:
        New temperature at step k

    """
    return initial_temp - k * beta


# Cauchy Cooling Strategy (Cauchy Cooling / Fast Annealing)
def cauchy_cooling(initial_temp: float, k: int, beta: float) -> float:
    """Cauchy (fast) cooling strategy.

    Theoretically better than geometric cooling at ensuring convergence to global optimum,
    with slower cooling rate.

    Args:
        initial_temp: Initial temperature T0.
        k: Current iteration step (starting from 1 to avoid division by zero).
        beta: Cooling rate; if beta is 1, it is Cauchy cooling, else Lundy and Mees cooling.

    Returns:
        New temperature at step k

    """
    return initial_temp / (1 + beta * k)


# Constant Temperature Strategy (Constant Temperature)
def constant_temperature(temp: float) -> float:
    """Constant temperature strategy.

    Temperature remains constant throughout the annealing process.

    Args:
        temp: Temperature value to maintain.

    Returns:
        The same temperature value

    """
    return temp


def temperature_schedule(t: int, annealing_config: base_config.AnnealingConfig) -> float:
    """Calculate temperature at given step based on annealing strategy.

    Args:
        t: Current time step or iteration number.
        annealing_config: Configuration object containing annealing parameters.

    Returns:
        Temperature at step t according to the specified annealing strategy

    """
    logger.info("Using %s annealing strategy", annealing_config.annealing_type)
    if annealing_config.annealing_type == "geometric":
        return geometric_cooling(annealing_config.initial_temp, t, annealing_config.beta)
    elif annealing_config.annealing_type == "linear":
        return linear_cooling(annealing_config.initial_temp, t, annealing_config.beta)
    elif annealing_config.annealing_type == "cauchy":
        return cauchy_cooling(annealing_config.initial_temp, t, annealing_config.beta)
    elif annealing_config.annealing_type == "constant":
        return constant_temperature(annealing_config.final_temp)
    else:
        raise ValueError(f"Not supported annealing type: {annealing_config.annealing_type}")


def init_cell_annealing_width(
    cfg: base_config.BornFreeConfig,
    precision: jnp.dtype,
    cell_annealing_width_ckpt: float | None = None,
):
    """Initialize cell annealing width for NPT simulations.

    Args:
        cfg: The configuration object for the simulation.
        precision: The floating-point precision to use ('float32' or 'float64').
        cell_annealing_width_ckpt: Optional checkpoint value for cell annealing width.

    Returns:
        Replicated cell annealing width array.

    """
    if cell_annealing_width_ckpt is not None:
        cell_annealing_width = kfac_jax.utils.replicate_all_local_devices(
            jnp.asarray(cfg.mcmc.annealing.cell_annealing_width, dtype=precision)
        )
    else:
        cell_annealing_width = kfac_jax.utils.replicate_all_local_devices(
            jnp.asarray(cfg.mcmc.annealing.cell_annealing_width, dtype=precision)
        )
    return cell_annealing_width


def setup_local_sampling_step(
    cfg: base_config.BornFreeConfig,
    batch_networks,
    device_batch_size,
    unit_cell: PyscfCell,
    precision,
):
    """Set up local sampling MCMC step function based on configuration.

    Args:
        cfg: The configuration object for the simulation.
        batch_networks: A dictionary of batched neural network functions.
        device_batch_size: Batch size per device.
        unit_cell: The PySCF Cell object defining the simulation cell.
        precision: The floating-point precision to use ('float32' or 'float64').

    Returns:
        MCMC step function for local sampling.

    """
    latvec = jnp.asarray(unit_cell.lattice_vectors(), dtype=precision)

    if cfg.mcmc.mcmc_type == "electron_only":
        raise ValueError(f"MCMC type {cfg.mcmc.mcmc_type} is not supported for NPT")
    else:
        sampling_func = None

    mcmc_config = mcmc.MCMCConfig(
        batch_per_device=device_batch_size,
        latvec=latvec,
        natom=unit_cell.natm,
        steps=cfg.mcmc.annealing.steps,
        iterations=cfg.mcmc.annealing.iter,
        atoms=None,
        importance_sampling=sampling_func,
        current_temp=None,
        annealing_steps=None,
    )

    return mcmc.MCMCStepFactory.create_step(mcmc_config, batch_networks["logabs"], cfg.mcmc.mcmc_type)


def setup_annealing_mcmc_step(
    cfg: base_config.BornFreeConfig,
    total_gibbs,
    current_temp,
    device_batch_size,
    unit_cell: PyscfCell,
    precision,
    mcmc_step,
):
    """Set up annealing MCMC step function based on configuration.

    Args:
        cfg: The configuration object for the simulation.
        total_gibbs: Function to calculate Gibbs free energy.
        current_temp: The current temperature for the annealing process.
        device_batch_size: Batch size per device.
        unit_cell: The PySCF Cell object defining the simulation cell.
        precision: The floating-point precision to use ('float32' or 'float64').
        mcmc_step: MCMC step function to wrap.

    Returns:
        MCMC step function for annealing.

    """
    logger.info("current_temp: %s", current_temp)
    latvec = jnp.asarray(unit_cell.lattice_vectors(), dtype=precision)

    def wrap_total_gibbs(params, data, key, mcmc_width):
        del key, mcmc_width
        return total_gibbs(params, data)

    def calculate_gibbs_func(mcmc_step, total_gibbs):
        logger.info("Using local sampling with %s steps", cfg.mcmc.annealing.local_steps)

        @jax.jit
        def calculate_gibbs_with_local_sampling(params, data, key, mcmc_width):
            enthalpy, aux_data = total_gibbs(params, data)

            def local_sampling_step(i, state):
                """A single step of the local sampling loop."""
                data, key, gibbs, aux_data = state
                mcmc_key, subkey = jax.random.split(key)
                new_data, _ = mcmc_step(params, data, mcmc_key, mcmc_width)

                enthalpy, aux_data = total_gibbs(params, new_data)
                gibbs += enthalpy
                return new_data, subkey, gibbs, aux_data

            data, _, gibbs, aux_data = jax.lax.fori_loop(
                0,
                cfg.mcmc.annealing.local_steps - 1,
                local_sampling_step,
                (data, key, enthalpy, aux_data),
            )
            return gibbs / cfg.mcmc.annealing.local_steps, aux_data

        return calculate_gibbs_with_local_sampling

    calculate_gibbs = calculate_gibbs_func(mcmc_step, total_gibbs)

    gibbs_function = calculate_gibbs if cfg.mcmc.annealing.local_sampling else wrap_total_gibbs

    mcmc_config = mcmc.MCMCConfig(
        batch_per_device=device_batch_size,
        latvec=latvec,
        natom=unit_cell.natm,
        steps=cfg.mcmc.steps,
        iterations=cfg.mcmc.iter,
        atoms=None,
        importance_sampling=None,
        current_temp=current_temp,
        annealing_steps=cfg.mcmc.annealing.annealing_steps,
    )

    return mcmc.MCMCStepFactory.create_step(mcmc_config, gibbs_function, "annealing")
