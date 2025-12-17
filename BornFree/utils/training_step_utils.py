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
"""Training step utilities for creating and managing training loops.

This module contains functions for creating training step functions for various
ensemble types (NVT/NPT), optimizers, and training strategies including annealing.
"""

import functools
import logging

import chex
import jax
import jax.numpy as jnp
import kfac_jax

from BornFree import base_config, constants
from BornFree.mcmc import annealing_utils


def create_training_step(mcmc_step, optimizer_step, reset_if_nan=False):
    """Create a training step function based on the MCMC type and optimizer.

    Args:
        mcmc_step: MCMC step function(s).
        optimizer_step: Optimizer step function.
        reset_if_nan: Whether to reset params if NaN loss is encountered.

    Returns:
        A pmapped training step function.

    """

    @functools.partial(constants.pmap, donate_argnums=(0, 1, 2))
    def step(data, params, state, key: chex.PRNGKey, mcmc_width: tuple):
        # MCMC loop
        mcmc_key, _ = jax.random.split(key, num=2)
        data, pmove = mcmc_step(params, data, mcmc_key, mcmc_width)

        # Optimization step
        new_params, new_state, loss, aux_data = optimizer_step(params, data, state)

        if reset_if_nan:
            new_params = jax.lax.cond(
                jnp.isnan(loss), lambda: params, lambda: new_params
            )
            new_state = jax.lax.cond(jnp.isnan(loss), lambda: state, lambda: new_state)
        pmove = [jnp.mean(p) for p in pmove]

        return data, new_params, new_state, loss, aux_data, pmove

    return step


def create_annealing_step(mcmc_step):
    """Create an annealing step function for geometry optimization.

    Args:
        mcmc_step: MCMC step function(s).

    Returns:
        A pmapped annealing step function.

    """

    @functools.partial(constants.pmap, donate_argnums=(0, 1, 2))
    def step(
        data,
        params,
        state,
        key: chex.PRNGKey,
        mcmc_width: tuple,
        cell_annealing_width: float,
    ):
        # MCMC loop
        mcmc_key, _ = jax.random.split(key, num=2)
        new_params, pmove, loss, aux_data = mcmc_step(
            params, data, mcmc_key, mcmc_width, cell_annealing_width
        )
        pmove = [constants.pmean(jnp.mean(p)) for p in pmove]

        return data, new_params, state, loss, aux_data, pmove

    return step


def create_kfac_training_step(
    mcmc_step, damping: float, optimizer: kfac_jax.Optimizer, reset_if_nan: bool = False
):
    """Factory to create training step for KFAC optimizers.

    Args:
        mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
            for creating the callable.
        damping: Value of damping to use for each KFAC update step.
        optimizer: KFAC optimizer instance.
        reset_if_nan: If true, reset the params and opt state to the state at the
            previous step when the loss is NaN.

    Returns:
        step, a callable which performs a set of MCMC steps and then an optimization
        update. See the Step protocol for details.

    """
    mcmc_step = constants.pmap(mcmc_step)
    shared_mom = kfac_jax.utils.replicate_all_local_devices(
        jnp.zeros([], dtype=jnp.float32)
    )
    shared_damping = kfac_jax.utils.replicate_all_local_devices(
        jnp.asarray(damping, dtype=jnp.float32)
    )
    # Due to some KFAC cleverness related to donated buffers, need to do this
    # to make state resettable
    copy_tree = constants.pmap(
        functools.partial(jax.tree_util.tree_map, lambda x: (1.0 * x).astype(x.dtype))
    )

    def step(
        data,
        params,
        state: kfac_jax.Optimizer.State,
        key: chex.PRNGKey,
        mcmc_width: tuple,
    ):
        # MCMC loop
        mcmc_keys, loss_keys = kfac_jax.utils.p_split(key)
        data, pmove = mcmc_step(params, data, mcmc_keys, mcmc_width)

        # Handle NaN reset if needed
        if reset_if_nan:
            old_params = copy_tree(params)
            old_state = copy_tree(state)
        # Optimization step
        new_params, new_state, stats = optimizer.step(
            params=params,
            state=state,
            rng=loss_keys,
            batch=data,
            momentum=shared_mom,
            damping=shared_damping,
        )

        if reset_if_nan and jnp.isnan(stats["loss"]):
            new_params = old_params
            new_state = old_state
        pmove = [jnp.mean(p).item() for p in pmove]

        return data, new_params, new_state, stats["loss"], stats["aux"], pmove

    return step


def create_training_step_npt(mcmc_step, optimizer_step, reset_if_nan=False):
    """Create a training step function for NPT ensemble.

    Args:
        mcmc_step: MCMC step function(s).
        optimizer_step: Optimizer step function.
        reset_if_nan: Whether to reset params if NaN loss is encountered.

    Returns:
        A pmapped training step function.

    """

    @functools.partial(constants.pmap, donate_argnums=(0, 1, 2))
    def step(
        data,
        params,
        state,
        key: chex.PRNGKey,
        mcmc_width: tuple,
        cell_annealing_width: float,
    ):
        # MCMC loop
        del cell_annealing_width
        mcmc_key, _ = jax.random.split(key, num=2)
        data, pmove = mcmc_step(params, data, mcmc_key, mcmc_width)

        # Optimization step
        new_params, new_state, loss, aux_data = optimizer_step(params, data, state)

        if reset_if_nan:
            new_params = jax.lax.cond(
                jnp.isnan(loss), lambda: params, lambda: new_params
            )
            new_state = jax.lax.cond(jnp.isnan(loss), lambda: state, lambda: new_state)
        pmove = [jnp.mean(p) for p in pmove]

        return data, new_params, new_state, loss, aux_data, pmove

    return step


def create_kfac_training_step_npt(
    mcmc_step, damping: float, optimizer: kfac_jax.Optimizer, reset_if_nan: bool = False
):
    """Factory to create training step for KFAC optimizers in NPT ensemble.

    Args:
        mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
            for creating the callable.
        damping: Value of damping to use for each KFAC update step.
        optimizer: KFAC optimizer instance.
        reset_if_nan: If true, reset the params and opt state to the state at the
            previous step when the loss is NaN.

    Returns:
        step, a callable which performs a set of MCMC steps and then an optimization
        update. See the Step protocol for details.

    """
    mcmc_step = constants.pmap(mcmc_step)
    shared_mom = kfac_jax.utils.replicate_all_local_devices(
        jnp.zeros([], dtype=jnp.float32)
    )
    shared_damping = kfac_jax.utils.replicate_all_local_devices(
        jnp.asarray(damping, dtype=jnp.float32)
    )
    # Due to some KFAC cleverness related to donated buffers, need to do this
    # to make state resettable
    copy_tree = constants.pmap(
        functools.partial(jax.tree_util.tree_map, lambda x: (1.0 * x).astype(x.dtype))
    )

    def step(
        data,
        params,
        state: kfac_jax.Optimizer.State,
        key: chex.PRNGKey,
        mcmc_width: tuple,
        cell_annealing_width: float,
    ):
        # MCMC loop
        del cell_annealing_width
        mcmc_keys, loss_keys = kfac_jax.utils.p_split(key)
        data, pmove = mcmc_step(params, data, mcmc_keys, mcmc_width)

        # Handle NaN reset if needed
        if reset_if_nan:
            old_params = copy_tree(params)
            old_state = copy_tree(state)
        # Optimization step
        new_params, new_state, stats = optimizer.step(
            params=params,
            state=state,
            rng=loss_keys,
            batch=data,
            momentum=shared_mom,
            damping=shared_damping,
        )

        if reset_if_nan and jnp.isnan(stats["loss"]):
            new_params = old_params
            new_state = old_state
        pmove = [jnp.mean(p).item() for p in pmove]

        return data, new_params, new_state, stats["loss"], stats["aux"], pmove

    return step


def determine_training_phase_and_step(
    t: int,
    cfg: base_config.BornFreeConfig,
    last_phase: str,
    optimizer_step_fn,
    wavefunction_step_fn,
    evaluate_loss,
    device_batch_size: int,
    unit_cell,
    precision,
    local_sampling_step,
    burn_in_step,
    data,
    params,
    sharded_key,
    mcmc_width,
    cell_annealing_width,
    current_temp: float,
):
    """Determine the current training phase and set up the appropriate optimizer step function.

    This function manages the training strategy by determining which phase the training is in
    (Inference, Warmup, Geo Opt, or Standard Opt) and setting up the appropriate optimizer
    step function for that phase.

    Args:
        t: Current training iteration.
        cfg: Configuration object.
        last_phase: The phase from the previous iteration.
        optimizer_step_fn: Current optimizer step function (will be updated if phase changes).
        wavefunction_step_fn: Standard wavefunction optimization step function.
        evaluate_loss: Loss evaluation function.
        device_batch_size: Batch size per device.
        unit_cell: Unit cell object.
        precision: Floating-point precision ('float32' or 'float64').
        local_sampling_step: Local sampling step function for annealing.
        burn_in_step: Burn-in MCMC step function.
        data: Current particle positions.
        params: Current parameters.
        sharded_key: Sharded random key.
        mcmc_width: MCMC step width.
        cell_annealing_width: Cell annealing width.
        current_temp: Current temperature for annealing.

    Returns:
        A tuple containing:
        - phase: Current training phase.
        - optimizer_step_fn: Optimizer step function for current phase.
        - last_phase: Updated last phase.
        - data: Updated particle positions (may be modified during burn-in).
        - params: Updated parameters (may be modified during burn-in).
        - sharded_key: Updated sharded key (may be modified during burn-in).
        - current_temp: Updated current temperature.

    """
    if cfg.optim.optimizer == "none":
        phase = "Inference"
        optimizer_step_fn = wavefunction_step_fn
    else:
        is_in_warmup = t < cfg.strategy.warmup_steps
        if is_in_warmup:
            phase = "Warmup"
        else:
            steps_after_warmup = t - cfg.strategy.warmup_steps
            cycle_length = cfg.strategy.opt_steps + cfg.strategy.geo_opt_steps
            step_in_cycle = steps_after_warmup % cycle_length
            if step_in_cycle < cfg.strategy.geo_opt_steps:
                phase = "Geo Opt"
            else:
                phase = "Standard Opt"

        if phase != last_phase:
            if phase == "Warmup":
                optimizer_step_fn = wavefunction_step_fn
            elif phase == "Geo Opt":
                current_temp = annealing_utils.temperature_schedule(
                    t - cfg.strategy.warmup_steps, cfg.mcmc.annealing
                )
                annealing_mcmc_step = annealing_utils.setup_annealing_mcmc_step(
                    cfg,
                    evaluate_loss,
                    current_temp,
                    device_batch_size,
                    unit_cell,
                    precision,
                    local_sampling_step,
                )
                optimizer_step_fn = create_annealing_step(annealing_mcmc_step)
            elif phase == "Standard Opt":
                optimizer_step_fn = wavefunction_step_fn
                for _ in range(cfg.mcmc.burn_in):
                    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
                    data, params, *_ = burn_in_step(
                        data,
                        params,
                        state=None,
                        key=subkeys,
                        mcmc_width=mcmc_width,
                        cell_annealing_width=cell_annealing_width,
                    )
                logging.info("Completed burn-in MCMC steps before standard opt")
            last_phase = phase

    return phase, optimizer_step_fn, last_phase, data, params, sharded_key, current_temp
