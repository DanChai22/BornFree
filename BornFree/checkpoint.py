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

import datetime
import os
import time
import zipfile

import jax
import numpy as np
from absl import logging

from BornFree import base_config
from BornFree.utils import writers


def get_restore_path(restore_path: str | None = None) -> str | None:
    """Gets the path containing checkpoints from a previous calculation.

    This function validates and returns the checkpoint restore path. It serves
    as a simple wrapper to ensure consistent handling of restore paths throughout
    the codebase.

    Args:
        restore_path: Path to the directory containing checkpoint files from a
            previous run. If None or empty, no restoration will be performed.

    Returns:
        The validated restore path, or None if restore_path is not provided
        (falsy value).

    """
    ckpt_restore_path = restore_path or None
    return ckpt_restore_path


def find_last_checkpoint(ckpt_path: str | None = None) -> str | None:
    """Finds the most recent valid checkpoint in a directory.

    Searches for checkpoint files matching the pattern 'qmcjax_ckpt_*' in the
    specified directory and returns the most recent one that can be successfully
    loaded. Checkpoints are sorted in reverse order (most recent first) by filename.

    The function validates each checkpoint by attempting to load it with np.load.
    If a checkpoint is corrupt or cannot be read (OSError, EOFError, or
    BadZipFile), it logs a warning and tries the next most recent checkpoint.

    Args:
        ckpt_path: Directory path containing checkpoint files. If None or the
            directory doesn't exist, no checkpoint will be found.

    Returns:
        Full path to the most recent valid checkpoint file, or None if:
        - ckpt_path is None or empty
        - The directory doesn't exist
        - No checkpoint files are found
        - All checkpoint files are corrupt or unreadable

    """
    if ckpt_path and os.path.exists(ckpt_path):
        files = [f for f in os.listdir(ckpt_path) if "qmcjax_ckpt_" in f]
        # Handle case where last checkpoint is corrupt/empty.
        for file in sorted(files, reverse=True):
            fname = os.path.join(ckpt_path, file)
            with open(fname, "rb") as f:
                try:
                    np.load(f, allow_pickle=True)
                    return fname
                except (OSError, EOFError, zipfile.BadZipFile):
                    logging.info("Error loading checkpoint %s. Trying next checkpoint...", fname)
    return None


def create_save_path(
    save_path: str | None,
) -> str:
    """Creates the directory for saving checkpoints, if it doesn't exist.

    If a save path is not provided, automatically generates a timestamped
    directory name in the current working directory using the pattern
    'BornFree_YYYY_MM_DD_HH_MM_SS'.

    Args:
        save_path: Directory path to use for saving checkpoints. If None or
            empty (falsy), a new directory will be created in the current
            working directory with a timestamp-based name.

    Returns:
        Absolute path to the checkpoint directory. The directory is guaranteed
        to exist after this function returns (created if necessary).

    """
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    default_save_path = os.path.join(os.getcwd(), f"BornFree_{timestamp}")
    ckpt_save_path = save_path or default_save_path
    if ckpt_save_path and not os.path.isdir(ckpt_save_path):
        os.makedirs(ckpt_save_path)
    return ckpt_save_path


def save(
    save_path: str,
    t: int,
    data,
    params,
    opt_state,
    atom_mcmc_width,
    elec_mcmc_width,
    run_id,
) -> str:
    """Saves checkpoint information to a numpy npz file.

    Creates a checkpoint file containing all necessary information to resume
    a simulation, including walker configurations, network parameters, optimizer
    state, MCMC parameters, and Weights & Biases run ID.

    The checkpoint file is saved as 'qmcjax_ckpt_{t:06d}.npz' where t is the
    iteration number, zero-padded to 6 digits.

    Args:
        save_path: Path to the directory where the checkpoint will be saved.
            The directory should already exist (created by create_save_path).
        t: Current iteration number (number of completed training steps). This
            is used in the checkpoint filename.
        data: MCMC walker configurations as a JAX array. Shape is typically
            (n_devices, batch_per_device, n_particles, spatial_dim).
        params: PyTree (nested dictionary/tuple structure) containing all
            trainable network parameters.
        opt_state: Optimizer state (e.g., Adam moments, KFAC statistics).
            Structure depends on the optimizer being used.
        atom_mcmc_width: Array containing the MCMC proposal width for atomic
            moves. Adapted during training to maintain target acceptance rate.
        elec_mcmc_width: Array containing the MCMC proposal width for electronic
            moves. Adapted during training to maintain target acceptance rate.
        run_id: Weights & Biases run ID for tracking and resuming logging.
            Can be None if W&B is not being used.

    Returns:
        Full path to the saved checkpoint file, including filename.

    """
    ckpt_filename = os.path.join(save_path, f"qmcjax_ckpt_{t:06d}.npz")
    logging.info("Saving checkpoint %s", ckpt_filename)
    with open(ckpt_filename, "wb") as f:
        np.savez(
            f,
            t=t,
            data=data,
            params=params,
            opt_state=opt_state,
            atom_mcmc_width=atom_mcmc_width,
            elec_mcmc_width=elec_mcmc_width,
            wandb_run_id=run_id,
        )

    return ckpt_filename


def restore(restore_filename: str, batch_size: int | None = None, shape_check=True):
    """Restores simulation state from a checkpoint file.

    Loads all checkpoint data including walker configurations, network parameters,
    optimizer state, MCMC parameters, and Weights & Biases run ID. Performs
    validation checks to ensure the checkpoint is compatible with the current
    hardware configuration and requested batch size.

    The iteration counter is automatically incremented by 1 to reflect that
    training will resume from the next iteration.

    Args:
        restore_filename: Full path to the checkpoint file (*.npz) to load.
        batch_size: Expected total batch size (across all devices). If provided,
            validates that the checkpoint's batch size matches. Set to None to
            skip batch size validation.
        shape_check: If True, validates that the checkpoint's data shape is
            compatible with the current device configuration and batch size.
            Set to False to skip validation (use with caution).

    Returns:
        A tuple containing (in order):
        - t (int): Next iteration number to resume from (checkpoint iteration + 1).
        - data (ndarray): MCMC walker configurations with shape
            (n_devices, batch_per_device, n_particles, spatial_dim).
        - params (dict): PyTree of network parameters.
        - opt_state (dict/tuple): Optimizer state, structure depends on optimizer.
        - atom_mcmc_width (ndarray): MCMC proposal width for atomic moves.
        - elec_mcmc_width (ndarray): MCMC proposal width for electronic moves.
        - wandb_run_id (str or None): Weights & Biases run ID for resuming logging,
            or None if not present in the checkpoint.

    Raises:
        ValueError: If shape_check is True and either:
            - The number of devices in the checkpoint doesn't match the current
              number of local devices (data.shape[0] != jax.local_device_count()).
            - The total batch size doesn't match the requested batch_size
              (data.shape[0] * data.shape[1] != batch_size).

    """
    logging.info("Loading checkpoint %s", restore_filename)
    with open(restore_filename, "rb") as f:
        ckpt_data = np.load(f, allow_pickle=True)
        # Retrieve data from npz file. Non-array variables need to be converted back
        # to natives types using .tolist().
        t = ckpt_data["t"].tolist() + 1  # Return the iterations completed.
        data = ckpt_data["data"]
        params = ckpt_data["params"].tolist()
        opt_state = ckpt_data["opt_state"].tolist()
        atom_mcmc_width = ckpt_data["atom_mcmc_width"].tolist()
        elec_mcmc_width = ckpt_data["elec_mcmc_width"].tolist()
        wandb_run_id = ckpt_data["wandb_run_id"].item() if "wandb_run_id" in ckpt_data else None
        if shape_check:
            if data.shape[0] != jax.local_device_count():
                raise ValueError(
                    f"Incorrect number of devices found. Expected {data.shape[0]}, found {jax.local_device_count()}."
                )
            if batch_size and data.shape[0] * data.shape[1] != batch_size:
                raise ValueError(
                    f"Wrong batch size in loaded data. Expected {batch_size}, found {data.shape[0] * data.shape[1]}."
                )
    return t, data, params, opt_state, atom_mcmc_width, elec_mcmc_width, wandb_run_id


def setup_checkpoint_and_config(cfg: base_config.BornFreeConfig):
    """Set up checkpoint paths and save configuration.

    Args:
        cfg: Configuration object.

    Returns:
        Tuple of (ckpt_save_path, ckpt_restore_filename)

    """
    ckpt_save_path = create_save_path(cfg.log.save_path)
    ckpt_restore_path = get_restore_path(cfg.log.restore_path)
    ckpt_restore_filename = find_last_checkpoint(ckpt_save_path) or find_last_checkpoint(ckpt_restore_path)

    config_save_path = os.path.join(ckpt_save_path, "config.yaml")
    writers.save_config_to_yaml(cfg, config_save_path)

    return ckpt_save_path, ckpt_restore_filename


def should_save_checkpoint(
    t: int,
    cfg: base_config.BornFreeConfig,
    time_of_last_ckpt: float,
) -> bool:
    """Check if a checkpoint should be saved at this iteration.

    Args:
        t: Current iteration.
        cfg: Configuration object.
        time_of_last_ckpt: Timestamp of last checkpoint.

    Returns:
        True if checkpoint should be saved.

    """
    # Check if warmup_steps exists and is greater than 0 (NPT specific)
    save_at_warmup_end = False
    if (
        hasattr(cfg, "strategy")
        and hasattr(cfg.strategy, "warmup_steps")
        and cfg.strategy.warmup_steps > 0
        and t == cfg.strategy.warmup_steps - 1
    ):
        save_at_warmup_end = True

    return (
        time.time() - time_of_last_ckpt > cfg.log.save_frequency * 60
        or t >= cfg.optim.iterations - 1
        or (cfg.log.save_frequency_in_step > 0 and t % cfg.log.save_frequency_in_step == 0)
        or save_at_warmup_end
    )
