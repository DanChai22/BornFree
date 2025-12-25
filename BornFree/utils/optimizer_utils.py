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

"""Optimizer utilities for training neural network wave functions.

This module contains functions for creating optimizer steps, handling parameter
updates, and managing optimizer states for various optimizers including Adam,
Muon, and KFAC.
"""

import dataclasses
import logging

import jax
import jax.numpy as jnp
import kfac_jax
import optax

from BornFree import base_config, constants, curvature_tags_and_blocks

logger = logging.getLogger(__name__)


def null_update(params, data, opt_state):
    """Performs an identity operation with an OptUpdate interface.

    Args:
        params: Model parameters.
        data: Batch data.
        opt_state: Optimizer state.

    Returns:
        Tuple of (params, opt_state, zeros, None)

    """
    del data
    return params, opt_state, jnp.zeros(1), None


def make_opt_update_step(evaluate_loss, optimizer: optax.GradientTransformation):
    """Returns an OptUpdate function for performing a parameter update.

    Args:
        evaluate_loss: Loss evaluation function.
        optimizer: Optax gradient transformation optimizer.

    Returns:
        Function that evaluates loss and updates parameters.

    """
    # Differentiate wrt parameters (argument 0)
    loss_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)

    def opt_update(params, data, opt_state):
        """Evaluates the loss and gradients and updates the parameters using optax."""
        (loss, aux_data), grad = loss_and_grad(params, data)
        grad = constants.pmean(grad)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux_data

    return opt_update


def make_loss_step(evaluate_loss):
    """Returns an OptUpdate function for evaluating the loss.

    Args:
        evaluate_loss: Loss evaluation function.

    Returns:
        Function that evaluates loss with OptUpdate interface.

    """

    def loss_eval(
        params,
        data,
        opt_state,
    ):
        """Evaluates just the loss and gradients with an OptUpdate interface."""
        loss, aux_data = evaluate_loss(params, data)
        return params, opt_state, loss, aux_data

    return loss_eval


def init_opt_state_and_step(
    cfg: base_config.BornFreeConfig,
    params,
    data,
    sharded_key,
    evaluate_loss,
    make_training_step,
    make_kfac_training_step,
    learning_rate_schedule,
    mcmc_step,
    opt_state_ckpt,
):
    """Initialize optimizer and opt state.

    Args:
        cfg: Configuration object.
        params: Model parameters.
        data: Batch data.
        sharded_key: Sharded random key.
        evaluate_loss: Loss function.
        make_training_step: Function to create training step for optax optimizers.
        make_kfac_training_step: Function to create training step for KFAC optimizer.
        learning_rate_schedule: Learning rate schedule function.
        mcmc_step: MCMC step function.
        opt_state_ckpt: Checkpoint for optimizer state (optional).

    Returns:
        Tuple of (opt_state, step_function)

    Raises:
        ValueError: If optimizer is not recognized.

    """
    optimizer_name = cfg.optim.optimizer
    logger.info("Initializing optimizer and opt state for %s", optimizer_name)

    if optimizer_name == "none":
        optimizer = None
    elif optimizer_name == "adam":
        optimizer = optax.chain(
            optax.scale_by_adam(**dataclasses.asdict(cfg.optim.adam)),
            optax.scale_by_schedule(learning_rate_schedule),
            optax.scale(-1.0),
        )
    elif optimizer_name == "muon":
        optimizer = optax.contrib.muon(learning_rate_schedule, **dataclasses.asdict(cfg.optim.muon))
    elif optimizer_name == "kfac":
        # Differentiate wrt parameters (argument 0)
        val_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)
        optimizer = kfac_jax.Optimizer(
            val_and_grad,
            l2_reg=cfg.optim.kfac.l2_reg,
            norm_constraint=cfg.optim.kfac.norm_constraint,
            value_func_has_aux=True,
            learning_rate_schedule=learning_rate_schedule,
            curvature_ema=cfg.optim.kfac.cov_ema_decay,
            inverse_update_period=cfg.optim.kfac.invert_every,
            min_damping=cfg.optim.kfac.min_damping,
            num_burnin_steps=0,
            register_only_generic=cfg.optim.kfac.register_only_generic,
            estimation_mode="fisher_exact",
            multi_device=True,
            pmap_axis_name=constants.PMAP_AXIS_NAME,
            auto_register_kwargs=dict(
                graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS,
            ),
        )
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        opt_state = optimizer.init(params, subkeys, data)
        opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state
    else:
        raise ValueError(f"Not a recognized optimizer: {cfg.optim.optimizer}")

    if not optimizer:
        opt_state = None
        step = make_training_step(mcmc_step=mcmc_step, optimizer_step=make_loss_step(evaluate_loss))
    elif isinstance(optimizer, optax.GradientTransformation):
        # optax/optax-compatible optimizer (ADAM, LAMB, MUON, ...)
        optimizer = optax.MultiSteps(optimizer, every_k_schedule=cfg.optim.ministeps)
        opt_state = jax.pmap(optimizer.init)(params)
        opt_state = opt_state if opt_state_ckpt is None else optax._src.wrappers.MultiStepsState(*opt_state)
        step = make_training_step(
            mcmc_step=mcmc_step,
            optimizer_step=make_opt_update_step(evaluate_loss, optimizer),
            reset_if_nan=cfg.optim.reset_if_nan,
        )
    elif isinstance(optimizer, kfac_jax.Optimizer):
        step = make_kfac_training_step(
            mcmc_step=mcmc_step,
            damping=cfg.optim.kfac.damping,
            optimizer=optimizer,
            reset_if_nan=cfg.optim.reset_if_nan,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    return opt_state, step
