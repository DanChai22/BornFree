# Copyright (c) 2025 Shengdu Chai
#
# Licensed under the Apache License, Version 2.0.

import logging
from collections.abc import Callable

import jax.numpy as jnp
import kfac_jax
import numpy as np
from jax import Array
from pyscf.pbc.gto import Cell as PyscfCell

from BornFree import base_config
from BornFree.mcmc import mcmc

logger = logging.getLogger(__name__)


def setup_mcmc_step(
    cfg: base_config.BornFreeConfig,
    batch_networks: dict[str, Callable],
    device_batch_size: int,
    simulation_cell: PyscfCell,
    precision: str,
) -> Callable:
    """Creates the MCMC step function.

    Args:
        cfg: The configuration object for the simulation.
        batch_networks: A dictionary of batched neural network functions.
        device_batch_size: Batch size per device.
        simulation_cell: The PySCF Cell object defining the simulation cell.
        precision: The floating-point precision to use ('float32' or 'float64').

    Returns:
        MCMC step function

    """
    logger.info("Setting up MCMC step...")
    latvec = jnp.asarray(simulation_cell.lattice_vectors(), dtype=precision)

    sampling_func = None
    if cfg.mcmc.mcmc_type == "electron_only":
        if cfg.ensemble == "NPT":  # NPT specific restriction
            raise ValueError(f"MCMC type {cfg.mcmc.mcmc_type} is not supported for NPT")
        # NVT specific logic for electron_only
        if cfg.mcmc.importance_sampling:
            sampling_func = batch_networks["logabs"]

    # NPT has a setup_local_sampling_step that uses annealing steps.
    # The current request is only to merge setup_mcmc_step.
    # So, we use cfg.mcmc.steps and cfg.mcmc.iter for the main MCMC step.
    mcmc_config = mcmc.MCMCConfig(
        batch_per_device=device_batch_size,
        latvec=latvec,
        natom=simulation_cell.natm,
        steps=cfg.mcmc.steps,
        iterations=cfg.mcmc.iter,
        atoms=None,
        importance_sampling=sampling_func,
        current_temp=None,
        annealing_steps=None,
    )
    return mcmc.MCMCStepFactory.create_step(
        mcmc_config, batch_networks["logabs"], cfg.mcmc.mcmc_type
    )


def init_mcmc_width(
    cfg: base_config.BornFreeConfig,
    precision: jnp.dtype,
    atom_mcmc_width_ckpt: float | None = None,
    elec_mcmc_width_ckpt: float | None = None,
):
    """Initialize MCMC widths.

    Args:
        cfg: The configuration object for the simulation.
        precision: The floating-point precision to use ('float32' or 'float64').
        atom_mcmc_width_ckpt: Checkpoint value for atom MCMC width.
        elec_mcmc_width_ckpt: Checkpoint value for electron MCMC width.

    Returns:
        Tuple of (atom_mcmc_width, elec_mcmc_width).

    """
    if (elec_mcmc_width_ckpt is not None) and (atom_mcmc_width_ckpt is not None):
        atom_mcmc_width = kfac_jax.utils.broadcast_all_local_devices(
            jnp.asarray(atom_mcmc_width_ckpt, dtype=precision)
        )
        elec_mcmc_width = kfac_jax.utils.broadcast_all_local_devices(
            jnp.asarray(elec_mcmc_width_ckpt, dtype=precision)
        )
    else:
        elec_mcmc_width = kfac_jax.utils.replicate_all_local_devices(
            jnp.asarray(cfg.mcmc.elec_move_width, dtype=precision)
        )
        atom_mcmc_width = kfac_jax.utils.replicate_all_local_devices(
            jnp.asarray(cfg.mcmc.atom_move_width, dtype=precision)
        )

    return atom_mcmc_width, elec_mcmc_width


def get_mcmc_width(
    atom_width: float, elec_width: float, nuclear_treatment: str
) -> tuple[float] | tuple[float, float]:
    """Get MCMC width based on model type.

    Args:
        atom_width: Width for atom moves.
        elec_width: Width for electron moves.
        nuclear_treatment: Model type ('fixed', 'quantum').

    Returns:
        MCMC width(s)

    """
    if nuclear_treatment == "quantum":
        mcmc_width = (atom_width, elec_width)
    elif nuclear_treatment == "fixed":
        mcmc_width = elec_width
    return mcmc_width


def update_mcmc_width(
    atom_mcmc_width: float, elec_mcmc_width: float, pmoves: Array, mcmc_type: str
) -> tuple[float, float]:
    """Update MCMC widths based on acceptance rates.

    Args:
        atom_mcmc_width: The current width for atom moves.
        elec_mcmc_width: The current width for electron moves.
        pmoves: The acceptance probabilities for the MCMC moves.
        mcmc_type: Type of MCMC moves.

    Returns:
        Updated widths for atoms and electrons

    """
    if mcmc_type in ["joint", "electron_only"]:
        if np.mean(pmoves, axis=-1)[0] > 0.55:
            elec_mcmc_width *= 1.1
            atom_mcmc_width *= 1.1
        if np.mean(pmoves, axis=-1)[0] < 0.5:
            elec_mcmc_width /= 1.1
            atom_mcmc_width /= 1.1
    elif mcmc_type == "gibbs":
        if np.mean(pmoves, axis=-1)[1] > 0.55:
            elec_mcmc_width *= 1.1
        if np.mean(pmoves, axis=-1)[1] < 0.5:
            elec_mcmc_width /= 1.1
        if np.mean(pmoves, axis=-1)[0] > 0.66:
            atom_mcmc_width *= 1.4
        if np.mean(pmoves, axis=-1)[0] < 0.50:
            atom_mcmc_width /= 1.4
    return atom_mcmc_width, elec_mcmc_width


def init_pmoves(cfg: base_config.BornFreeConfig, precision: jnp.dtype):
    """Initialize probability of moves array for MCMC sampling.

    Args:
        cfg: The configuration object for the simulation.
        precision: The floating-point precision to use ('float32' or 'float64').

    Returns:
        Array of zeros with shape determined by MCMC type.

    """
    if cfg.mcmc.mcmc_type in ["joint", "electron_only"]:
        pmoves = np.zeros((1, cfg.mcmc.adapt_frequency), dtype=precision)
    elif cfg.mcmc.mcmc_type == "gibbs":
        pmoves = np.zeros((2, cfg.mcmc.adapt_frequency), dtype=precision)
    return pmoves


def update_mcmc_width_if_needed(
    t: int,
    cfg: base_config.BornFreeConfig,
    atom_mcmc_width: Array,
    elec_mcmc_width: Array,
    pmoves: Array,
):
    """Update MCMC width if at adaptation frequency.

    Args:
        t: Current iteration.
        cfg: The configuration object for the simulation.
        atom_mcmc_width: The current width for atom moves.
        elec_mcmc_width: The current width for electron moves.
        pmoves: The acceptance probabilities for the MCMC moves.

    Returns:
        Tuple of (atom_mcmc_width, elec_mcmc_width, mcmc_width, pmoves)

    """
    if t > 0 and t % cfg.mcmc.adapt_frequency == 0:
        atom_mcmc_width, elec_mcmc_width = update_mcmc_width(
            atom_mcmc_width, elec_mcmc_width, pmoves, cfg.mcmc.mcmc_type
        )
        mcmc_width = get_mcmc_width(
            atom_mcmc_width, elec_mcmc_width, cfg.nuclear_treatment
        )
        pmoves[:] = 0
    else:
        mcmc_width = get_mcmc_width(
            atom_mcmc_width, elec_mcmc_width, cfg.nuclear_treatment
        )

    return atom_mcmc_width, elec_mcmc_width, mcmc_width, pmoves
