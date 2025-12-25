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
import logging
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from jax import Array
from pyscf.pbc.gto import Cell as PyscfCell

from BornFree import base_config, hamiltonian
from BornFree.network import network_block
from BornFree.utils import units, writers

logger = logging.getLogger(__name__)


def extract_and_scale_metrics(loss, aux_data, scale):
    """Extract and scale training metrics from loss and auxiliary data.

    This function extracts various energy components from the auxiliary data
    and scales them appropriately using the simulation cell scale factor.

    Args:
        loss: The loss value (typically a JAX array), or None.
        aux_data: Auxiliary data object containing energy components
                  (variance, imaginary, kinetic, ewald, ewald_ee, ewald_ei,
                  ewald_ii, and optionally pv), or None.
        scale: The simulation cell scale factor for normalization.

    Returns:
        A dictionary containing the scaled metrics:
        - loss: Scaled total energy (loss / scale)
        - variance: Scaled variance (variance / scale^2)
        - imaginary: Scaled imaginary component (imaginary / scale)
        - kinetic: Mean scaled kinetic energy (mean(kinetic) / scale)
        - ewald: Mean scaled Ewald sum (mean(ewald) / scale)
        - ee: Mean scaled electron-electron interaction (mean(ewald_ee) / scale)
        - ei: Mean scaled electron-ion interaction (mean(ewald_ei) / scale)
        - ii: Mean scaled ion-ion interaction (mean(ewald_ii) / scale)
        - pv: Mean scaled pressure-volume term (mean(pv) / scale), if available
        All values are None if the corresponding input is None.

    """
    metrics = {}

    metrics["loss"] = loss[0] / scale if loss is not None else None
    metrics["variance"] = (
        aux_data.variance[0] / scale**2 if aux_data is not None else None
    )
    metrics["imaginary"] = (
        aux_data.imaginary[0] / scale if aux_data is not None else None
    )
    metrics["kinetic"] = (
        jnp.mean(aux_data.kinetic) / scale if aux_data is not None else None
    )
    metrics["ewald"] = (
        jnp.mean(aux_data.ewald) / scale if aux_data is not None else None
    )
    metrics["ee"] = (
        jnp.mean(aux_data.ewald_ee) / scale if aux_data is not None else None
    )
    metrics["ei"] = (
        jnp.mean(aux_data.ewald_ei) / scale if aux_data is not None else None
    )
    metrics["ii"] = (
        jnp.mean(aux_data.ewald_ii) / scale if aux_data is not None else None
    )

    # NPT-specific metric
    if aux_data is not None and hasattr(aux_data, "pv"):
        metrics["pv"] = jnp.mean(aux_data.pv) / scale
    else:
        metrics["pv"] = None

    return metrics


def initialize_wandb(
    cfg: base_config.BornFreeConfig, run_id_to_resume: str | None
) -> str | None:
    """Initialize Weights & Biases on the main process.

    Args:
        cfg: Configuration object.
        run_id_to_resume: W&B run ID to resume from, if any.

    Returns:
        The W&B run ID (either resumed or newly created).

    """
    if jax.process_index() == 0:
        run = wandb.init(
            project=f"bornfree_{cfg.ensemble}_{cfg.nuclear_treatment}",
            config=writers.mlcollections_to_dict(cfg),
            name=writers.get_wandb_save_name(cfg),
            resume="must" if run_id_to_resume else None,
            id=run_id_to_resume,
            dir=cfg.log.save_path,
        )
        return run.id
    return run_id_to_resume


def log_training_step(
    t: int,
    metrics: dict[str, Array | None],
    pmove: Array,
    atom_mcmc_width: Array,
    elec_mcmc_width: Array,
    simulation_cell: PyscfCell,
    cfg: base_config.BornFreeConfig,
    # Optional arguments for NPT or specific phases
    params: dict | None = None,
    phase: str | None = None,
    current_temp: float | None = None,
) -> dict[str, Any]:
    """Logs a summary of the training step.

    Handles both NVT and NPT ensembles based on the provided arguments and configuration.

    Args:
        t: The current training step.
        metrics: Dictionary containing scaled energy metrics with keys:
            - loss: The total energy (or enthalpy for NPT).
            - variance: The variance of the energy/enthalpy.
            - imaginary: The imaginary part of the energy.
            - kinetic: The kinetic energy.
            - ewald: The Ewald energy.
            - ee: The electron-electron Ewald energy.
            - ei: The electron-ion Ewald energy.
            - ii: The ion-ion Ewald energy.
            - pv: (NPT only) The PV term.
        pmove: The acceptance ratios for the MCMC moves.
        atom_mcmc_width: The MCMC move width for the atoms.
        elec_mcmc_width: The MCMC move width for the electrons.
        simulation_cell: The PySCF cell object.
        cfg: The configuration object for the simulation.
        params: (NPT only) Network parameters, used for volume calculation.
        phase: (NPT only) The current simulation phase (e.g., 'Warmup', 'Geo Opt').
        current_temp: (NPT only) Current temperature during annealing.

    Returns:
        A dictionary containing the training step results for CSV logging.

    """
    # Extract individual metrics from the dictionary
    loss = metrics["loss"]
    variance = metrics["variance"]
    imaginary = metrics["imaginary"]
    kinetic = metrics["kinetic"]
    ewald = metrics["ewald"]
    ee = metrics["ee"]
    ei = metrics["ei"]
    ii = metrics["ii"]
    pv = metrics.get("pv")

    log_metrics = {}

    # Common helper to format keys
    def format_key(key):
        return f"{phase}/{key}" if phase else key

    # --- 1. Basic Metrics (Common) ---
    log_metrics[format_key("loss")] = np.asarray(loss)
    log_metrics[format_key("variance")] = np.asarray(variance)

    # For NVT, it's energy per atom. For NPT, it's enthalpy per atom.
    label_per_atom = "enthalpy per atom" if cfg.ensemble == "NPT" else "energy per atom"
    log_metrics[format_key(label_per_atom)] = np.asarray(loss / simulation_cell.natm)
    # NPT specifically logs "enthalpy per atom" without phase prefix as well?
    # The original NPT code did: "enthalpy per atom": ..., and f"{phase}/enthalpy per atom": ...
    # We will replicate the specific behavior if NPT.
    if cfg.ensemble == "NPT":
        log_metrics["enthalpy per atom"] = np.asarray(loss / simulation_cell.natm)

    std_per_atom = np.sqrt(variance) / simulation_cell.natm / np.sqrt(cfg.batch_size)
    log_metrics[format_key("std per atom")] = np.asarray(std_per_atom)
    if cfg.ensemble == "NPT":
        log_metrics["std per atom"] = np.asarray(std_per_atom)

    log_metrics[format_key("imaginary_part")] = np.asarray(imaginary)
    log_metrics[format_key("kinetic_energy")] = np.asarray(kinetic.real)
    log_metrics[format_key("ewald_energy")] = np.asarray(ewald)
    log_metrics[format_key("ewald_ee")] = np.asarray(ee)
    log_metrics[format_key("ewald_ei")] = np.asarray(ei)
    log_metrics[format_key("ewald_ii")] = np.asarray(ii)
    log_metrics[format_key("atom_mcmc_width")] = np.asarray(jnp.mean(atom_mcmc_width))
    log_metrics[format_key("elec_mcmc_width")] = np.asarray(jnp.mean(elec_mcmc_width))

    if cfg.ensemble == "NPT":
        log_metrics[format_key("pv")] = np.asarray(pv)

    if cfg.ensemble == "NPT":
        if params is None or pv is None:
            raise ValueError("params and pv are required for NPT logging")

        volume = jnp.linalg.det(
            network_block.get_jacobian(params["cell"][0], cfg.crystal.lattice)
        )
        _, p = units.pressure_estimator(kinetic.real, ewald - pv, volume)
        log_metrics[format_key("estimated_pressure")] = np.asarray(p)

        if phase == "Geo Opt":
            log_metrics[format_key("current_temp")] = np.asarray(current_temp)

    log_msg_parts = [
        f"{datetime.datetime.now()}",
    ]
    if phase:
        log_msg_parts.append(f"[{phase}]")

    log_msg_parts.extend([
        f"Step {t:05d}: {loss:03.4f} E_h",
        f"variance={variance:03.4f} E_h^2",
    ])

    if cfg.mcmc.mcmc_type == "gibbs":
        log_metrics[format_key("atom_pmove")] = pmove[0]
        log_metrics[format_key("elec_pmove")] = pmove[1]
        log_msg_parts.extend([
            f"atom_pmove={pmove[0]:0.2f}",
            f"elec_pmove={pmove[1]:0.2f}",
        ])
    elif cfg.mcmc.mcmc_type in ["joint", "electron_only"] and cfg.ensemble == "NVT":
        log_metrics[format_key("pmove")] = pmove[0]
        log_msg_parts.append(f"pmove={pmove[0]:0.2f}")
    else:
        raise ValueError(f"Not supported mcmc type: {cfg.mcmc.mcmc_type}")

    log_msg_parts.extend([
        f"imaginary part={imaginary:03.4f}",
        f"kinetic={kinetic.real:03.4f} E_h",
        f"ewald={ewald:03.4f} E_h",
        f"ee={ee:03.4f} E_h",
        f"ei={ei:03.4f} E_h",
        f"ii={ii:03.4f} E_h",
    ])

    if cfg.ensemble == "NPT":
        log_msg_parts.append(f"pv={pv:03.4f} E_h")

    logger.info(", ".join(log_msg_parts))

    wandb.log(log_metrics, step=t)

    # Prepare result_dict for CSV logging
    result_dict = {
        "step": t,
        "energy": np.asarray(loss),
        "variance": np.asarray(variance),
        "pmove": np.asarray(pmove),
        "imaginary": np.asarray(imaginary),
        "kinetic": np.asarray(kinetic),
        "ewald": np.asarray(ewald),
        "ee": np.asarray(ee),
        "ei": np.asarray(ei),
        "ii": np.asarray(ii),
        "atom_mcmc_width": np.asarray(jnp.mean(atom_mcmc_width)),
        "elec_mcmc_width": np.asarray(jnp.mean(elec_mcmc_width)),
    }

    if cfg.ensemble == "NPT":
        result_dict["phase"] = phase
        result_dict["pv"] = np.asarray(pv)

    return result_dict


def log_bo_energy(
    cfg: base_config.BornFreeConfig,
    networks: dict[str, Callable],
    simulation_cell: PyscfCell,
    ckpt_save_path: str,
    params: Any,
    data: Array,
) -> None:
    """Logs the Born-Oppenheimer energy components.

    Calculates and logs the kinetic energy components (total, atom-atom, atom-electron, cross-term)
    to WandB and a CSV file. This is only performed if nuclear treatment is "quantum".

    Args:
        cfg: Configuration object.
        networks: Dictionary of network functions.
        simulation_cell: Simulation cell object.
        ckpt_save_path: Path to save the CSV log file.
        params: Network parameters.
        data: Input data (electron and atom positions).

    """
    if cfg.nuclear_treatment != "quantum":
        return

    BO_schema = [
        "step",
        "first line real",
        "first term real",
        "second term real",
        "third term real",
        "first line imag",
        "first term imag",
        "second term imag",
        "third term imag",
    ]

    with writers.Writer(
        name="BO_energy",
        schema=BO_schema,
        directory=ckpt_save_path,
        iteration_key=None,
        log=False,
    ) as writer:
        # Validate ensemble and set lattice parameter accordingly
        if cfg.ensemble == "NPT":
            lattice = cfg.crystal.lattice
        elif cfg.ensemble == "NVT":
            lattice = None
        else:
            raise ValueError(
                f"Unsupported ensemble: {cfg.ensemble}. Use 'nvt' or 'npt'."
            )

        # Create Born-Oppenheimer kinetic energy function
        kindif = hamiltonian.make_BO_kin(
            networks["log"].apply,
            networks["atom"].apply,
            networks["electron"].apply,
            simulation_cell,
            "for",
            lattice,
            cfg.crystal.is_deuterium,
            3,
            cfg.ensemble,
        )

        kin_atom_total, kin_atom_atom, kin_atom_elec, kin_atom_cross = kindif(
            params, data
        )

        # Handle NPT specific unpacking if necessary (based on original code behavior)
        kin_atom_total = kin_atom_total[0]
        kin_atom_atom = kin_atom_atom[0]
        kin_atom_elec = kin_atom_elec[0]
        kin_atom_cross = kin_atom_cross[0]

        if jax.process_index() == 0:
            logger.info(
                "BO energy: first line %s, second line, first term %s, "
                "second term %s, third term %s",
                kin_atom_total,
                kin_atom_atom,
                kin_atom_elec,
                kin_atom_cross,
            )
            metrics_BO = {
                "first line real": np.asarray(kin_atom_total.real),
                "first term real": np.asarray(kin_atom_atom.real),
                "second term real": np.asarray(kin_atom_elec.real),
                "third term real": np.asarray(kin_atom_cross.real),
                "first line imag": np.asarray(kin_atom_total.imag),
                "first term imag": np.asarray(kin_atom_atom.imag),
                "second term imag": np.asarray(kin_atom_elec.imag),
                "third term imag": np.asarray(kin_atom_cross.imag),
            }
            wandb.log(metrics_BO)
            result_dict = {"step": 1, **metrics_BO}
            writer.write(1, **result_dict)
            writer.flush()
            writers.clean_csv(
                ckpt_save_path,
                "BO_energy",
                BO_schema,
            )
