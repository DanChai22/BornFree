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

import dataclasses
import logging
from typing import Any

import chex
import jax
import jax.numpy as jnp
import kfac_jax

from BornFree import base_config
from BornFree.network import (
    network_npt_quantum,
    network_nvt_fixed,
    network_nvt_quantum,
)


def setup_network_functions(
    cfg: base_config.BornFreeConfig, system_dict: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Set up network functions based on configuration.

    Args:
        cfg: Configuration object.
        system_dict: Dictionary containing system parameters.

    Returns:
        Tuple of network functions and their batched versions.

    """
    if cfg.network.detnet.is_rezero:
        logging.info("Using ReZero in atom network")
    else:
        logging.info("Not using ReZero in atom network")

    networks = {}
    batched_networks = {}

    if cfg.nuclear_treatment == "quantum":
        if cfg.ensemble == "NPT":
            network_module = network_npt_quantum
        elif cfg.ensemble == "NVT":
            network_module = network_nvt_quantum
        else:
            raise ValueError(
                f"Unsupported ensemble for quantum treatment: {cfg.ensemble}"
            )

        log_network = network_module.make_solid_fermi_net(
            **system_dict, method_name="eval_log_network"
        )
        logabs_network = network_module.make_solid_fermi_net(
            **system_dict, method_name="eval_logabs_network"
        )
        electron_network = network_module.make_solid_fermi_net(
            **system_dict, method_name="eval_log_network", mcmc="electron"
        )
        atom_network = network_module.make_solid_fermi_net(
            **system_dict, method_name="eval_log_network", mcmc="atom"
        )

        networks = {
            "log": log_network,
            "logabs": logabs_network,
            "electron": electron_network,
            "atom": atom_network,
        }

        batched_networks = {
            "log": jax.vmap(log_network.apply, in_axes=(None, 0), out_axes=0),
            "logabs": jax.vmap(logabs_network.apply, in_axes=(None, 0), out_axes=0),
        }

    elif cfg.nuclear_treatment == "fixed":
        if cfg.ensemble == "NPT":
            raise ValueError("Fixed nuclear treatment not yet supported for NPT.")

        log_network = network_nvt_fixed.make_solid_fermi_net(
            **system_dict, method_name="eval_log_network"
        )
        logabs_network = network_nvt_fixed.make_solid_fermi_net(
            **system_dict, method_name="eval_logabs_network"
        )

        networks = {
            "log": log_network,
            "logabs": logabs_network,
        }

        batched_networks = {
            "log": jax.vmap(log_network.apply, in_axes=(None, 0), out_axes=0),
            "logabs": jax.vmap(logabs_network.apply, in_axes=(None, 0), out_axes=0),
        }

    else:
        raise ValueError(f"Unsupported nuclear treatment: {cfg.nuclear_treatment}")

    return networks, batched_networks


def init_params(log_network, key, precision):
    """Initialize parameters for the network.

    Args:
        log_network: Network object to initialize.
        key: JAX random key for initialization.
        precision: Floating-point precision ('float32' or 'float64').

    Returns:
        Initialized network parameters replicated across all devices.

    """
    params = log_network.init(key=key, data=None)
    params = jax.tree_map(lambda x: x.astype(precision), params)
    return kfac_jax.utils.replicate_all_local_devices(params)


def setup_networks_and_params(
    cfg: base_config.BornFreeConfig,
    system_dict: dict[str, Any],
    key: chex.PRNGKey,
    precision: jnp.dtype,
):
    """Set up neural networks and initialize parameters.

    Args:
        cfg: Configuration object.
        system_dict: Dictionary containing system information.
        key: JAX random key for initialization.
        precision: Floating-point precision ('float32' or 'float64').

    Returns:
        Tuple of (networks, batched_networks, params, key).

    """
    system_dict.update(dataclasses.asdict(cfg.network.detnet))

    networks, batched_networks = setup_network_functions(cfg, system_dict)

    key, subkey = jax.random.split(key)
    params = init_params(networks["log"], subkey, precision)

    return networks, batched_networks, params, key
