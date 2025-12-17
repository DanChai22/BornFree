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

"""Initialization utilities for setting up simulations.

This module contains functions for initializing particles, devices, cells,
random seeds, and other basic simulation components.
"""

import logging
import time
from collections.abc import Sequence

import chex
import jax
import jax.numpy as jnp
import kfac_jax
from jax import Array
from jax.experimental import multihost_utils
from pyscf.pbc.gto import Cell as PyscfCell

from BornFree import base_config, hf, init_guess
from BornFree.utils import system


def initialize_data(
    cfg: base_config.BornFreeConfig,
    key: chex.PRNGKey,
    internal_cell: Sequence[system.Atom],
    simulation_cell: PyscfCell,
    host_batch_size: int,
    precision: str,
    data_shape: tuple[int, ...],
) -> Array:
    """Initializes the particle positions for the simulation.

    This function handles the initialization of particle positions based on the
    selected nuclear treatment. For 'quantum', it initializes both electron and
    atom positions. For 'fixed', only electron positions are initialized.

    Args:
        cfg: The configuration object for the simulation.
        key: A JAX random key for reproducible initialization.
        internal_cell: The internal representation of the simulation cell.
        simulation_cell: The PySCF cell object.
        host_batch_size: The batch size per host.
        precision: Floating-point precision ('float32' or 'float64').
        data_shape: The shape of the data to initialize.

    Returns:
        An array containing the initialized particle positions, broadcast to all
        local devices.

    Raises:
        ValueError: If the nuclear treatment is not supported.

    """
    logging.info("Initializing particle positions...")
    if cfg.nuclear_treatment == "quantum":
        data_electron = init_guess.init_electrons(
            key=key,
            cell=internal_cell,
            latvec=jnp.asarray(simulation_cell.lattice_vectors(), dtype=precision),
            electrons=simulation_cell.nelec,
            batch_size=host_batch_size,
            init_width=cfg.mcmc.elec_init_width,
        )
        data_atom = init_guess.init_atom(
            key=key,
            cell=internal_cell,
            latvec=jnp.asarray(simulation_cell.lattice_vectors(), dtype=precision),
            batch_size=host_batch_size,
            init_width=cfg.mcmc.atom_init_width,
        )
        data = jnp.hstack((data_atom, data_electron))
    elif cfg.nuclear_treatment == "fixed":
        if cfg.ensemble == "NPT":
            raise ValueError("Fixed nuclear treatment not supported for NPT ensemble.")

        data = init_guess.init_electrons(
            key=key,
            cell=internal_cell,
            latvec=jnp.asarray(simulation_cell.lattice_vectors(), dtype=precision),
            electrons=simulation_cell.nelec,
            batch_size=host_batch_size,
            init_width=cfg.mcmc.elec_init_width,
        )
    else:
        raise ValueError(f"Unsupported nuclear treatment: {cfg.nuclear_treatment}")

    data = data.astype(precision)
    data = jnp.reshape(data, data_shape + data.shape[1:])
    return kfac_jax.utils.broadcast_all_local_devices(data)


def init_device(cfg: base_config.BornFreeConfig):
    """Initialize device configuration and batch sizes.

    Args:
        cfg: The configuration object for the simulation.

    Returns:
        Tuple of (host_batch_size, device_batch_size, data_shape)

    Raises:
        ValueError: If batch size is not divisible by total number of devices.

    """
    num_devices = jax.local_device_count()
    num_hosts = jax.device_count() // num_devices
    logging.info(
        "Starting QMC with %i XLA devices per host across %i hosts.",
        num_devices,
        num_hosts,
    )

    host_batch_size = cfg.batch_size // num_hosts
    device_batch_size = host_batch_size // num_devices
    data_shape = (num_devices, device_batch_size)
    cfg.host_batch_size = host_batch_size
    if cfg.batch_size % (num_devices * num_hosts) != 0:
        raise ValueError(
            "Batch size must be divisible by number of devices, "
            f"got batch size {host_batch_size} for {num_devices * num_hosts} devices."
        )
    return host_batch_size, device_batch_size, data_shape


def convert_cell_dtype(cell, precision: jnp.dtype):
    """Convert cell attributes to specified precision.

    Args:
        cell: Cell object with attributes a, AV, BV.
        precision: Floating-point precision ('float32' or 'float64').

    Returns:
        Cell object with converted attributes.

    """
    cell.a = jnp.asarray(cell.a, dtype=precision)
    cell.AV = jnp.asarray(cell.AV, dtype=precision)
    cell.BV = jnp.asarray(cell.BV, dtype=precision)
    return cell


def initialize_simulation(cfg: base_config.BornFreeConfig, precision: str) -> tuple:
    """Initializes the simulation cell and Hartree-Fock calculation.

    Args:
        cfg: The configuration object for the simulation.
        precision: Floating-point precision ('float32' or 'float64').

    Returns:
        A tuple containing the simulation cell, the internal cell representation,
        and the Hartree-Fock calculator.

    """
    logging.info("Initializing simulation cell...")
    simulation_cell = cfg.system.pyscf_cell
    simulation_cell = convert_cell_dtype(simulation_cell, precision)
    internal_cell = init_guess.pyscf_to_cell(cell=simulation_cell)
    hartree_fock = hf.SCF(cell=simulation_cell, twist=jnp.array(cfg.network.twist))
    hartree_fock.init_scf()
    return simulation_cell, internal_cell, hartree_fock


def get_precision(cfg: base_config.BornFreeConfig):
    """Get JAX precision type from configuration.

    Args:
        cfg: The configuration object for the simulation.

    Returns:
        JAX dtype (jnp.float64 or jnp.float32)

    Raises:
        ValueError: If precision is not supported.

    """
    if cfg.precision == "float64":
        logging.info("Using float64 precision")
        return jnp.float64
    elif cfg.precision == "float32":
        logging.info("Using float32 precision")
        return jnp.float32
    else:
        raise ValueError(f"Not supported precision: {cfg.precision}")


def setup_random_seed(cfg: base_config.BornFreeConfig):
    """Set up the random seed based on the configuration.

    Args:
        cfg: The configuration object for the simulation.

    Returns:
        JAX PRNGKey initialized with seed.

    """
    logging.info("setup random seed")
    if cfg.debug.deterministic:
        seed = 666
    else:
        seed = jnp.asarray([1e6 * time.time()])
        seed = int(multihost_utils.broadcast_one_to_all(seed)[0])
    return jax.random.PRNGKey(seed)


def setup_basic_components(cfg: base_config.BornFreeConfig):
    """Set up basic components common to both NVT and NPT.

    Args:
        cfg: Configuration object.

    Returns:
        Tuple of (host_batch_size, device_batch_size, data_shape, precision, key)

    Raises:
        ValueError: If system dimension is not 3.

    """
    host_batch_size, device_batch_size, data_shape = init_device(cfg)
    precision = get_precision(cfg)
    key = setup_random_seed(cfg)

    if cfg.system.ndim != 3:
        raise ValueError("Only 3D systems are currently supported.")

    return host_batch_size, device_batch_size, data_shape, precision, key
