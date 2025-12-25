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
import time

import jax
import jax.numpy as jnp
import kfac_jax
import wandb

from BornFree import (
    base_config,
    checkpoint,
)
from BornFree.loss import setup_evaluate_loss
from BornFree.mcmc import mcmc_utils
from BornFree.network import network_utils
from BornFree.utils import (
    initialization_utils,
    logging_utils,
    optimizer_utils,
    training_step_utils,
    visualize,
    writers,
)

logger = logging.getLogger(__name__)


def process(cfg: base_config.BornFreeConfig) -> None:
    """Runs the main NVT simulation process.

    This function orchestrates the entire simulation, including device setup,
    initialization, pre-training, the main optimization loop, and logging.

    Args:
        cfg: The configuration object for the simulation.

    """
    # Basic setup
    host_batch_size, device_batch_size, data_shape, precision, key = (
        initialization_utils.setup_basic_components(cfg)
    )

    # Initialization
    simulation_cell, internal_cell, hartree_fock = (
        initialization_utils.initialize_simulation(cfg, precision)
    )

    system_dict = {
        "klist": hartree_fock.klist,
        "simulation_cell": simulation_cell,
    }
    networks, batched_networks, params, key = network_utils.setup_networks_and_params(
        cfg, system_dict, key, precision
    )

    # Checkpointing setup
    ckpt_save_path, ckpt_restore_filename = checkpoint.setup_checkpoint_and_config(cfg)

    # Restore from checkpoint or initialize new state
    run_id_to_resume = None
    if ckpt_restore_filename:
        (
            t_init,
            data,
            params,
            opt_state_ckpt,
            atom_mcmc_width_ckpt,
            elec_mcmc_width_ckpt,
            run_id_to_resume,
        ) = checkpoint.restore(ckpt_restore_filename, host_batch_size)
        if run_id_to_resume:
            logger.info("Found Wandb Run ID %s in checkpoint.", run_id_to_resume)
        else:
            logger.warning(
                "No Wandb Run ID found in checkpoint. A new run will be created."
            )
    else:
        logger.info("No checkpoint found. Training new model.")
        key, subkey = jax.random.split(key)
        # make sure data on each host is initialized differently
        subkey = jax.random.fold_in(subkey, jax.process_index())
        data = initialization_utils.initialize_data(
            cfg,
            subkey,
            internal_cell,
            simulation_cell,
            host_batch_size,
            precision,
            data_shape,
        )
        t_init = 0
        opt_state_ckpt = None
        atom_mcmc_width_ckpt = None
        elec_mcmc_width_ckpt = None

    # Set up Weights & Biases for logging
    run_id_to_resume = logging_utils.initialize_wandb(cfg, run_id_to_resume)

    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)

    # Set up MCMC, loss, and optimization
    logger.info("create mcmc functions")
    mcmc_step = mcmc_utils.setup_mcmc_step(
        cfg, batched_networks, device_batch_size, simulation_cell, precision
    )
    evaluate_loss = setup_evaluate_loss(
        cfg, networks, batched_networks, simulation_cell
    )

    def learning_rate_schedule(t):
        return cfg.optim.lr.rate * jnp.power(
            (1.0 / (1.0 + (t / cfg.optim.lr.delay))), cfg.optim.lr.decay
        )

    opt_state, step = optimizer_utils.init_opt_state_and_step(
        cfg,
        params,
        data,
        sharded_key,
        evaluate_loss,
        training_step_utils.create_training_step,
        training_step_utils.create_kfac_training_step,
        learning_rate_schedule,
        mcmc_step,
        opt_state_ckpt,
    )
    atom_mcmc_width, elec_mcmc_width = mcmc_utils.init_mcmc_width(
        cfg, precision, atom_mcmc_width_ckpt, elec_mcmc_width_ckpt
    )
    pmoves = mcmc_utils.init_pmoves(cfg, precision)
    mcmc_width = mcmc_utils.get_mcmc_width(
        atom_mcmc_width, elec_mcmc_width, cfg.nuclear_treatment
    )

    # MCMC burn-in
    if t_init == 0:
        logger.info("Burning in MCMC chain for %d steps...", cfg.mcmc.burn_in)
        burn_in_step = training_step_utils.create_training_step(
            mcmc_step=mcmc_step, optimizer_step=optimizer_utils.null_update
        )
        for _ in range(cfg.mcmc.burn_in):
            sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
            data, params, *_ = burn_in_step(
                data, params, state=None, key=subkeys, mcmc_width=mcmc_width
            )
        logger.info("Completed MCMC burn-in.")

    time_of_last_ckpt = time.time()

    if cfg.optim.optimizer == "none" and opt_state_ckpt is not None:
        # If opt_state_ckpt is None, then we're restarting from a previous inference
        # run (most likely due to preemption) and so should continue from the last
        # iteration in the checkpoint. Otherwise, starting an inference run from a
        # training run.
        logger.info("No optimizer provided. Assuming inference run.")
        logger.info("Setting initial iteration to 0.")
        t_init = 0

    # Main training loop
    logger.info("Starting optimization...")
    train_schema = [
        "step",
        "energy",
        "variance",
        "pmove",
        "imaginary",
        "kinetic",
        "ewald",
        "ee",
        "ei",
        "ii",
        "atom_mcmc_width",
        "elec_mcmc_width",
    ]
    with writers.Writer(
        name=cfg.log.stats_file_name,
        schema=train_schema,
        directory=ckpt_save_path,
        iteration_key=None,
        log=False,
    ) as writer:
        for t in range(t_init, cfg.optim.iterations):
            sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
            data, params, opt_state, loss, aux_data, pmove = step(
                data, params, opt_state, subkeys, mcmc_width
            )

            # Adapt MCMC move width
            atom_mcmc_width, elec_mcmc_width, mcmc_width, pmoves = (
                mcmc_utils.update_mcmc_width_if_needed(
                    t, cfg, atom_mcmc_width, elec_mcmc_width, pmoves
                )
            )

            if cfg.optim.optimizer in ["none", "adam", "muon"]:
                pmove = [p[0] for p in pmove]
            pmoves[:, t % cfg.mcmc.adapt_frequency] = pmove

            # Extract and scale metrics
            metrics = logging_utils.extract_and_scale_metrics(
                loss, aux_data, simulation_cell.scale
            )

            # Log and save checkpoint
            if jax.process_index() == 0:
                if t % cfg.log.stats_frequency == 0 and metrics["loss"] is not None:
                    result_dict = logging_utils.log_training_step(
                        t,
                        metrics,
                        pmove,
                        atom_mcmc_width,
                        elec_mcmc_width,
                        simulation_cell,
                        cfg,
                    )
                    writer.write(t, **result_dict)
                    writer.flush()

                if checkpoint.should_save_checkpoint(t, cfg, time_of_last_ckpt):
                    checkpoint.save(
                        ckpt_save_path,
                        t,
                        data,
                        params,
                        opt_state,
                        atom_mcmc_width,
                        elec_mcmc_width,
                        run_id_to_resume,
                    )
                    if cfg.nuclear_treatment == "quantum":
                        visualize.plot_nvt_quantum_result(
                            data, t, cfg, metrics["kinetic"], metrics["ewald"]
                        )
                    time_of_last_ckpt = time.time()

        writers.clean_csv(
            ckpt_save_path, cfg.log.stats_file_name, train_schema
        ) if jax.process_index() == 0 else None

        # Final calculations
        if cfg.nuclear_treatment == "quantum":
            logging_utils.log_bo_energy(
                cfg, networks, simulation_cell, ckpt_save_path, params, data
            )

        wandb.finish()
