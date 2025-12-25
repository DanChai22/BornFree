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
from typing import Any

import ml_collections
from pyscf.pbc.gto import Cell as PyscfCell


@dataclasses.dataclass
class AnnealingConfig:
    """Configuration for simulated annealing.

    Attributes:
        annealing_type: The type of annealing schedule to use (e.g., 'constant',
            'geometric', 'linear', 'cauchy').
        initial_temp: The initial temperature for the annealing schedule.
        final_temp: The final temperature for the annealing schedule.
        beta: The cooling factor for the annealing schedule.
        annealing_steps: The number of annealing steps.
        local_sampling: Whether to use local sampling during annealing.
        local_steps: The number of local sampling steps.
        steps: The number of MCMC steps per annealing iteration.
        iter: The number of MCMC iterations per annealing step.
        cell_annealing_width: The width of the cell annealing moves.

    """

    annealing_type: str = "cauchy"
    initial_temp: float = 0.1
    final_temp: float = 0.001
    beta: float = 0.002
    annealing_steps: int = 10
    local_sampling: bool = False
    local_steps: int = 10
    steps: int = 5
    iter: int = 5
    cell_annealing_width: float = 0.002


@dataclasses.dataclass
class Strategy:
    """Defines the training strategy, including warmup and optimization steps.

    Attributes:
        warmup_steps: The number of warmup steps(same as NVT) before the main optimization.
        opt_steps: The number of standard optimization(optimize the wavefunction) steps in a cycle.
        geo_opt_steps: The number of geometric optimization(optimize the cell parameters) steps in a cycle.

    """

    warmup_steps: int = 10
    opt_steps: int = 10
    geo_opt_steps: int = 1


@dataclasses.dataclass
class OptimLrConfig:
    """Configuration for the learning rate schedule.

    Attributes:
        rate: The base learning rate.
        decay: The learning rate decay exponent.
        delay: The delay for the learning rate decay.

    """

    rate: float
    decay: float
    delay: float


@dataclasses.dataclass
class OptimAdamConfig:
    """Configuration for the ADAM optimizer."""

    b1: float
    b2: float
    eps: float
    eps_root: float


@dataclasses.dataclass
class OptimMuonConfig:
    """Configuration for the MUON optimizer."""

    ns_coeffs: tuple[float, float, float]
    ns_steps: int
    beta: float
    eps: float
    adam_b1: float
    adam_b2: float
    adam_eps_root: float


@dataclasses.dataclass
class OptimKfacConfig:
    """Configuration for the K-FAC optimizer."""

    invert_every: int
    cov_update_every: int
    damping: float
    cov_ema_decay: float
    momentum: float
    momentum_type: str
    min_damping: float
    norm_constraint: float
    mean_center: bool
    l2_reg: float
    register_only_generic: bool


@dataclasses.dataclass
class OptimConfig:
    """Configuration for the optimization process.

    Attributes:
        iterations: The total number of optimization iterations.
        optimizer: The optimizer to use ('kfac', 'adam', 'muon', 'none').
        local_energy_outlier_width: The width for clipping local energy outliers.
        lr: The learning rate configuration.
        clip_el: The value at which to clip the local energy.
        clip_type: The type of clipping to use.
        reset_if_nan: Whether to reset the optimization if NaNs are encountered.
        adam: The ADAM optimizer configuration.
        muon: The MUON optimizer configuration.
        kfac: The K-FAC optimizer configuration.
        ministeps: The number of mini-steps per optimization step.
        laplacian_mode: The mode for calculating the Laplacian ('folx', 'for', 'partition', 'hessian').
        partition_number: The number of partitions for the Laplacian calculation.

    """

    iterations: int
    optimizer: str
    local_energy_outlier_width: float
    lr: OptimLrConfig  # Nested dataclass
    clip_el: float
    clip_type: str
    reset_if_nan: bool
    adam: OptimAdamConfig  # Nested dataclass
    muon: OptimMuonConfig  # Nested dataclass
    kfac: OptimKfacConfig  # Nested dataclass
    ministeps: int
    laplacian_mode: str
    partition_number: int


@dataclasses.dataclass
class LogConfig:
    """Configuration for logging and saving results.

    Attributes:
        stats_frequency: The frequency at which to log statistics.
        save_frequency: The frequency (in minutes) at which to save checkpoints.
        save_frequency_in_step: The frequency (in steps) at which to save checkpoints.
        save_path: The path to save checkpoints and other results.
        restore_path: The path to restore checkpoints from.
        stats_file_name: The name of the statistics file.

    """

    stats_frequency: int
    save_frequency: float
    save_frequency_in_step: int
    save_path: str
    restore_path: str
    stats_file_name: str


@dataclasses.dataclass
class SystemConfig:
    """Configuration for the physical system.

    Attributes:
        pyscf_cell: The PySCF cell object for the system.
        ndim: The number of dimensions of the system.
        internal_cell: The internal representation of the cell.

    """

    pyscf_cell: PyscfCell | None
    ndim: int
    internal_cell: Any | None


@dataclasses.dataclass
class McmcConfig:
    """Configuration for the Markov Chain Monte Carlo (MCMC) sampling.

    Attributes:
        mcmc_type: The type of MCMC move to use ('gibbs', 'joint', 'electron_only').
        burn_in: The number of burn-in steps for the MCMC chain.
        steps: The number of MCMC steps between optimization updates.
        iter: The number of MCMC iterations per step.
        elec_init_width: The initial width for electron positions.
        atom_init_width: The initial width for atom positions.
        elec_move_width: The width of electron MCMC moves.
        atom_move_width: The width of atom MCMC moves.
        adapt_frequency: The frequency at which to adapt the MCMC move widths.
        importance_sampling: Whether to use importance sampling.
        one_electron: Whether to use one-electron moves.
        annealing: The configuration for simulated annealing.

    """

    mcmc_type: str
    burn_in: int
    steps: int
    iter: int
    elec_init_width: float
    atom_init_width: float
    elec_move_width: float
    atom_move_width: float
    adapt_frequency: int
    importance_sampling: bool
    one_electron: bool
    annealing: AnnealingConfig  # Nested dataclass


@dataclasses.dataclass
class NetworkDetnetConfig:
    """Configuration for the DetNet network architecture.

    Attributes:
        envelope_type: The type of envelope function to use.
        atom_center_dynamic: Whether to use dynamic atomic centers.
        is_rezero: Whether to use ReZero.
        bias_orbitals: Whether to use bias in the orbital layer.
        use_last_layer: Whether to use the last layer of the network.
        full_det: Whether to use the full determinant.
        hidden_dims: The dimensions of the hidden layers.
        determinants: The number of determinants.
        distance_type: The type of distance to use.

    """

    envelope_type: str
    atom_center_dynamic: bool
    is_rezero: bool
    bias_orbitals: bool
    use_last_layer: bool
    full_det: bool
    hidden_dims: tuple[tuple[int, ...], ...]
    determinants: int
    distance_type: str


@dataclasses.dataclass
class NetworkConfig:
    """Configuration for the neural network.

    Attributes:
        detnet: The configuration for the DetNet architecture.
        twist: The twist angle for the simulation.

    """

    detnet: NetworkDetnetConfig  # Nested dataclass
    twist: tuple[float, float, float]


@dataclasses.dataclass
class DebugConfig:
    """Configuration for debugging."""

    deterministic: bool


@dataclasses.dataclass
class CrystalKptsConfig:
    """Configuration for the k-points."""

    number: list[int]
    twist_index: int
    length: int
    weights: float


@dataclasses.dataclass
class CrystalLatticeConfig:
    """Configuration for the crystal lattice."""

    mode: str = "angle"
    a: float = 1.0
    b: float = 1.0
    c: float = 1.0
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0


@dataclasses.dataclass
class CrystalConfig:
    """Configuration for the crystal structure.

    Attributes:
        structure: The type of crystal structure.
        rs: The Wigner-Seitz radius.
        ncopy: The size of the supercell.
        basis: The basis set to use.
        kpts: The k-points configuration.
        cif_path: The path to the CIF file.
        is_deuterium: Whether to average over isotopes.
        dis: The displacement of the atoms.
        natm: The number of atoms.
        lattice: The lattice configuration.

    """

    structure: str
    rs: float
    ncopy: list[int]
    basis: str
    kpts: CrystalKptsConfig  # Nested dataclass
    cif_path: str | None
    is_deuterium: bool
    natm: int | None
    lattice: CrystalLatticeConfig  # Nested dataclass


# --- Main Configuration Dataclass ---
@dataclasses.dataclass
class BornFreeConfig:
    """The main configuration object for BornFree simulations."""

    strategy: Strategy
    ensemble: str
    nuclear_treatment: str
    precision: str
    target_pressure: float
    batch_size: int
    host_batch_size: int
    xrd_exp: list | None
    infer: bool
    config_module: str
    use_x64: bool
    optim: OptimConfig
    log: LogConfig
    system: SystemConfig
    mcmc: McmcConfig
    network: NetworkConfig
    debug: DebugConfig
    crystal: CrystalConfig


def default() -> ml_collections.ConfigDict:
    """Creates a set of default parameters for running a simulation.

    Note that some parameters, such as the system definition, must be set by the
    user in a separate configuration file.

    Returns:
        An `ml_collections.ConfigDict` containing the default settings.

    """
    # wavefunction output.
    cfg = ml_collections.ConfigDict(
        {
            "config_module": __name__,
            "ensemble": "NVT",  # NVT or NPT
            "nuclear_treatment": "fixed",  # 'quantum','fixed'
            "precision": "float32",  # float32 or float64
            "target_pressure": 100.0,  # target pressure in GPa
            "batch_size": 1024,  # batch size
            "host_batch_size": 1024,  # host batch size
            "xrd_exp": None,  # list of xrd experimental data
            "infer": False,  # whether to run in inference mode
            # Config module used. Should be set in get_config function as either the
            # absolute module or relative to the configs subdirectory. Relative
            # imports must start with a '.' (e.g. .atom). Do *not* override on
            # command-line. Do *not* set using __name__ from inside a get_config
            # function, as config_flags overrides this when importing the module using
            # importlib.import_module.
            "use_x64": True,  # use float64 or 32
            "strategy": {
                "warmup_steps": 10000,
                "opt_steps": 1000,
                "geo_opt_steps": 100,
            },
            "optim": {
                "iterations": 10000,  # number of iterations
                "optimizer": "kfac",
                "local_energy_outlier_width": 5.0,
                "lr": {
                    "rate": 3.0e-2,  # learning rate, different from the reported lr in FermiNet
                    "decay": 1.0,  # exponent of learning rate decay
                    "delay": 10000.0,  # term that sets the scale of the rate decay
                },
                "clip_el": 5.0,  # If not none, scale at which to clip local energy
                "clip_type": "real",  # Clip real and imag part of gradient.
                "reset_if_nan": False,
                # ADAM hyperparameters. See optax documentation for details.
                "adam": {
                    "b1": 0.9,
                    "b2": 0.999,
                    "eps": 1.0e-8,
                    "eps_root": 0.0,
                },
                "muon": {
                    "ns_coeffs": (3.4445, -4.7750, 2.0315),
                    "ns_steps": 5,
                    "beta": 0.95,
                    "eps": 1.0e-8,
                    "adam_b1": 0.9,
                    "adam_b2": 0.999,
                    "adam_eps_root": 0.0,
                },
                "kfac": {
                    "invert_every": 1,
                    "cov_update_every": 1,
                    "damping": 0.001,
                    "cov_ema_decay": 0.95,
                    "momentum": 0.0,
                    "momentum_type": "regular",
                    # Warning: adaptive damping is not currently available.
                    "min_damping": 1.0e-4,
                    "norm_constraint": 0.001,
                    "mean_center": True,
                    "l2_reg": 0.0,
                    "register_only_generic": False,
                },
                "ministeps": 1,
                "laplacian_mode": "folx",  # 'folx', 'for', 'partition', 'hessian'
                "partition_number": 3,
                # Only used for 'partition' mode.
                # partition_number must be divisivle by (dim * number of electrons).
                # The smaller the faster, but requires more memory.
            },
            "log": {
                "stats_frequency": 10,  # iterations between logging of stats
                "save_frequency": 60.0,  # minutes between saving network params
                "save_frequency_in_step": -1,
                "save_path": "",
                # specify the local save path
                "restore_path": "",
                # specify the restore path which contained saved Model parameters.
                "stats_file_name": "train_stats",
            },
            "system": {
                "pyscf_cell": None,  # simulation cell object
                "ndim": 3,  # dimension of the system
                "internal_cell": None,
            },
            "mcmc": {
                "annealing": {
                    "annealing_steps": 10,
                    "annealing_type": "cauchy",
                    "initial_temp": 0.1,
                    "final_temp": 0.001,
                    "beta": 0.002,
                    "local_sampling": False,
                    "local_steps": 10,
                    "steps": 5,
                    "iter": 5,
                    "cell_annealing_width": 0.02,
                },
                "mcmc_type": "gibbs",  # 'joint' or 'gibbs' or 'electron_only'
                "burn_in": 100,
                "steps": 5,  # Number of MCMC steps to make between network updates.
                "iter": 10,
                "elec_init_width": 0.8,
                "atom_init_width": 0.08,
                "elec_move_width": 0.02,
                "atom_move_width": 0.0002,
                "adapt_frequency": 100,  # Number of steps after which to update the adaptive MCMC step size
                # If true, scale the proposal width for each electron by the harmonic
                # mean of the distance to the nuclei.
                "importance_sampling": False,  # not tested
                # whether to use importance sampling in MCMC step, untested yet
                # Metropolis sampling will be used if false
                "one_electron": False,
                # If true, use one-electron moves, untested yet
            },
            "network": {
                "detnet": {
                    "envelope_type": "isotropic",
                    "atom_center_dynamic": True,
                    # only isotropic mode has been tested
                    "is_rezero": False,
                    "bias_orbitals": False,
                    "use_last_layer": False,
                    "full_det": False,
                    "hidden_dims": ((64, 32, 32), (64, 32, 32), (64, 32, 32)),
                    "determinants": 8,
                    "distance_type": "tri",  # 'nu' or 'tri'
                },
                "twist": (
                    0.25,
                     0.25,
                     0.25,
                 ),  # Define the twist of wavefunction, twists are given in terms
                # of fractions of supercell reciprocal vectors
            },
            "debug": {
                "deterministic": False,  # Use a deterministic seed.
            },
            "crystal": {
                "structure": "bcc_H",  # crystal structure type: 'bcc', etc.
                "rs": 1.31,  # Wigner-Seitz radius
                "ncopy": [1, 1, 1],  # supercell size
                "basis": "sto-3g",  # basis
                "kpts": {
                    "number": [1, 1, 1],  # k-points grid
                    "twist_index": 0,  # twist index in the k-points list
                    "length": 1,
                    "weights": 1.0,
                },
                "lattice": {
                    # 'angle' or 'partial_angle' or 'diag', 'partial_angle' means
                    # we do not optimize angle
                    "mode": "angle",
                    "a": 1.0,
                    "b": 1.0,
                    "c": 1.0,
                    "alpha": 90.0,
                    "beta": 90.0,
                    "gamma": 90.0,
                },
                "cif_path": None,  # path to the cif file
                "is_deuterium": False,
                "natm": None,
            },
        }
    )

    return cfg


def _convert_dict_to_dataclass(dc, data_dict):
    """Recursively converts a dictionary to a nested dataclass structure.

    Args:
        dc: Target dataclass type to convert to.
        data_dict: Dictionary or ConfigDict containing the data.

    Returns:
        Instance of the dataclass with nested dataclasses populated.

    """
    field_types = {f.name: f.type for f in dataclasses.fields(dc)}
    return dc(
        **{
            f: (
                _convert_dict_to_dataclass(field_types[f], data_dict[f])
                if dataclasses.is_dataclass(field_types[f])
                and isinstance(data_dict.get(f), dict | ml_collections.ConfigDict)
                else data_dict.get(f)
            )  # Use .get for robustness if needed
            for f in field_types
            # Add filtering or error handling if dict keys might not match dataclass fields
            if f in data_dict
        }
    )


def config_dict_to_dataclass(cfg: ml_collections.ConfigDict) -> BornFreeConfig:
    """Converts ml_collections.ConfigDict to BornFreeConfig dataclass.

    Args:
        cfg: Configuration dictionary to convert.

    Returns:
        BornFreeConfig dataclass instance with all nested structures populated.

    """
    # Convert ConfigDict to a regular dict first if necessary,
    # although direct iteration might work.
    # Using to_dict() ensures compatibility.
    cfg_dict = cfg.to_dict()
    return _convert_dict_to_dataclass(BornFreeConfig, cfg_dict)


def resolve(cfg):
    """Resolve configuration references.

    Args:
        cfg: ml_collections.ConfigDict with potential references

    Returns:
        Resolved configuration with all references expanded
    """
    cfg = cfg.copy_and_resolve_references()
    return cfg
