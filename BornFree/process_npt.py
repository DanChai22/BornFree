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
import numpy as np
import wandb
from pyscf.pbc.gto import Cell as PyscfCell

from BornFree import (
    base_config,
    checkpoint,
    init_guess,
    supercell,
)
from BornFree.loss import setup_evaluate_loss
from BornFree.mcmc import annealing_utils, mcmc_utils
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


def process(cfg: base_config.BornFreeConfig):
    """Process NPT (constant pressure-temperature) quantum Monte Carlo simulation.

    Args:
        cfg: Configuration object containing all simulation parameters
    """
    host_batch_size, device_batch_size, data_shape, precision, key = (
        initialization_utils.setup_basic_components(cfg)
    )
    simulation_cell: PyscfCell = cfg.system.pyscf_cell
    unit_cell = supercell.get_supercell(
        supercell.convert_simulation_cell_to_unit_cell(simulation_cell),
        np.diag([1, 1, 1]),
    )
    unit_cell = initialization_utils.convert_cell_dtype(unit_cell, precision)
    simulation_cell = initialization_utils.convert_cell_dtype(
        simulation_cell, precision
    )
    internal_cell = init_guess.pyscf_to_cell(cell=unit_cell)
    klist = supercell.get_klist(cell=unit_cell, twist=jnp.array(cfg.network.twist))

    system_dict = {
        "klist": klist,
        "simulation_cell": simulation_cell,
        "unit_cell": unit_cell,
        "lattice_mode": cfg.crystal.lattice.mode,
    }

    networks, batched_networks, params, key = network_utils.setup_networks_and_params(
        cfg, system_dict, key, precision
    )

    # Checkpointing setup
    ckpt_save_path, ckpt_restore_filename = checkpoint.setup_checkpoint_and_config(cfg)

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
        cell_annealing_width_ckpt = jnp.asarray(cfg.mcmc.annealing.cell_annealing_width)
        cfg.strategy.warmup_steps = 0
        if run_id_to_resume and cfg.optim.optimizer != "none":
            logger.info("Found Wandb Run ID %s in checkpoint.", run_id_to_resume)
        else:
            run_id_to_resume = None
            logger.warning(
                "No Wandb Run ID found in checkpoint. A new run will be created."
            )
    else:
        logger.info("No checkpoint found. Training new model.")
        key, subkey = jax.random.split(key)
        # make sure data on each host is initialized differently
        subkey = jax.random.fold_in(subkey, jax.process_index())
        # initialize the configuration in a unit cell [0,1]^3
        data = initialization_utils.initialize_data(
            cfg,
            subkey,
            internal_cell,
            unit_cell,
            host_batch_size,
            precision,
            data_shape,
        )
        t_init = 0
        opt_state_ckpt = None
        atom_mcmc_width_ckpt = None
        elec_mcmc_width_ckpt = None
        cell_annealing_width_ckpt = None

    # Initialize wandb only on the main process
    run_id_to_resume = logging_utils.initialize_wandb(cfg, run_id_to_resume)

    sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)

    logger.info("create mcmc functions")
    mcmc_step = mcmc_utils.setup_mcmc_step(
        cfg, batched_networks, device_batch_size, unit_cell, precision
    )
    local_sampling_step = annealing_utils.setup_local_sampling_step(
        cfg, batched_networks, device_batch_size, unit_cell, precision
    )
    evaluate_loss = setup_evaluate_loss(
        cfg,
        networks,
        batched_networks,
        simulation_cell,
    )

    def learning_rate_schedule(t):
        return cfg.optim.lr.rate * jnp.power(
            (1.0 / (1.0 + (t / cfg.optim.lr.delay))), cfg.optim.lr.decay
        )

    atom_mcmc_width, elec_mcmc_width = mcmc_utils.init_mcmc_width(
        cfg, precision, atom_mcmc_width_ckpt, elec_mcmc_width_ckpt
    )
    cell_annealing_width = annealing_utils.init_cell_annealing_width(
        cfg, precision, cell_annealing_width_ckpt
    )
    pmoves = mcmc_utils.init_pmoves(cfg, precision)

    mcmc_width = mcmc_utils.get_mcmc_width(
        atom_mcmc_width, elec_mcmc_width, cfg.nuclear_treatment
    )

    burn_in_step = training_step_utils.create_training_step_npt(
        mcmc_step=mcmc_step, optimizer_step=optimizer_utils.null_update
    )
    # Initialize optimizer state and step function
    opt_state, wavefunction_step_fn = optimizer_utils.init_opt_state_and_step(
        cfg,
        params,
        data,
        sharded_key,
        evaluate_loss,
        training_step_utils.create_training_step_npt,
        training_step_utils.create_kfac_training_step_npt,
        learning_rate_schedule,
        mcmc_step,
        opt_state_ckpt,
    )

    if t_init == 0:
        logger.info("Burning in MCMC chain for %d steps", cfg.mcmc.burn_in)

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
        logger.info("Completed burn-in MCMC steps")
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)

    time_of_last_ckpt = time.time()

    if cfg.optim.optimizer == "none" and opt_state_ckpt is not None:
        # If opt_state_ckpt is None, then we're restarting from a previous inference
        # run (most likely due to preemption) and so should continue from the last
        # iteration in the checkpoint. Otherwise, starting an inference run from a
        # training run.
        logger.info("No optimizer provided. Assuming inference run.")
        logger.info("Setting initial iteration to 0.")
        t_init = 0

    train_schema = [
        "step",
        "phase",
        "energy",
        "variance",
        "pmove",
        "imaginary",
        "kinetic",
        "ewald",
        "ee",
        "ei",
        "ii",
        "pv",
        "atom_mcmc_width",
        "elec_mcmc_width",
    ]
    last_phase = None
    current_temp = cfg.mcmc.annealing.initial_temp
    optimizer_step_fn = wavefunction_step_fn
    with writers.Writer(
        name=cfg.log.stats_file_name,
        schema=train_schema,
        directory=ckpt_save_path,
        iteration_key=None,
        log=False,
    ) as writer:
        logger.info("start optimize")
        for t in range(t_init, cfg.optim.iterations):
            (
                phase,
                optimizer_step_fn,
                last_phase,
                data,
                params,
                sharded_key,
                current_temp,
            ) = training_step_utils.determine_training_phase_and_step(
                t=t,
                cfg=cfg,
                last_phase=last_phase,
                optimizer_step_fn=optimizer_step_fn,
                wavefunction_step_fn=wavefunction_step_fn,
                evaluate_loss=evaluate_loss,
                device_batch_size=device_batch_size,
                unit_cell=unit_cell,
                precision=precision,
                local_sampling_step=local_sampling_step,
                burn_in_step=burn_in_step,
                data=data,
                params=params,
                sharded_key=sharded_key,
                mcmc_width=mcmc_width,
                cell_annealing_width=cell_annealing_width,
                current_temp=current_temp,
            )
            sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
            data, params, opt_state, loss, aux_data, pmove = optimizer_step_fn(
                data,
                params,
                opt_state,
                subkeys,
                mcmc_width,
                cell_annealing_width,
            )

            if phase == "Geo Opt":
                pmove = [p[0] for p in pmove]
            else:
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
                        params,
                        phase,
                        current_temp,
                    )
                    writer.write(t, **result_dict)
                    writer.flush()

                    if phase == "Geo Opt":
                        cell_params = params["cell"][0]  # Shape (6,)
                        param_names = (
                            ["a", "b", "c", "alpha", "beta", "gamma"]
                            if cfg.crystal.lattice.mode == "angle"
                            else ["a", "b", "c"]
                        )
                        cell_metrics = {
                            f"cell/{param_names[i]}": cell_params[i]
                            for i in range(len(param_names))
                        }
                        wandb.log(cell_metrics, step=t)

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

                    visualize.plot_npt_result(
                        data,
                        params,
                        t,
                        cfg,
                        metrics["kinetic"],
                        metrics["ewald"] - metrics["pv"],
                    )

                    time_of_last_ckpt = time.time()

        writers.clean_csv(
            ckpt_save_path, cfg.log.stats_file_name, train_schema
        ) if jax.process_index() == 0 else None
        if cfg.nuclear_treatment == "quantum":
            logging_utils.log_bo_energy(
                cfg, networks, simulation_cell, ckpt_save_path, params, data
            )

        # Close wandb at the end of training
        wandb.finish()
