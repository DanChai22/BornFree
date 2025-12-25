# Copyright (c) 2025 Shengdu Chai
#
# Licensed under the Apache License, Version 2.0.

import itertools
import logging

import numpy as np
from pyscf.pbc import gto
from pyscf.pbc.gto import Cell as PyscfCell

from BornFree import base_config, supercell
from BornFree.config import utils

logger = logging.getLogger(__name__)


def _lattice_const(rs: float = 1.31, nh: int = 2) -> np.ndarray:
    volume = (4 / 3) * np.pi * (rs**3) * nh
    length = volume ** (1 / 3)
    return length


def make_atoms(
    rs: float = 1.31,
    ncopy: np.ndarray = np.array([2, 2, 2]),
):
    """Make atom pyscf style coords."""
    lattice = _lattice_const(rs)
    atom_strs = []
    for ii, jj, kk in itertools.product(
        range(ncopy[0]), range(ncopy[1]), range(ncopy[2])
    ):
        xx = ii * lattice
        yy = jj * lattice
        zz = kk * lattice
        atom_strs += [f"H {xx} {yy} {zz}"]
        atom_strs += [
            f"H {xx + 0.5 * lattice} {yy + 0.5 * lattice} {zz + 0.5 * lattice}"
        ]
    return ";".join(atom_strs)


def make_lattice(
    rs: float = 1.31,
    ncopy: np.ndarray = np.array([2, 2, 2]),
):
    """Create lattice vectors for BCC structure.

    Args:
        rs: Wigner-Seitz radius in atomic units
        ncopy: Array specifying supercell dimensions

    Returns:
        Lattice vectors as diagonal matrix
    """
    lattice = _lattice_const(rs)
    return lattice * np.diag(ncopy)


def make_cell(
    rs: float = 1.31,
    ncopy: np.ndarray = np.array([2, 2, 2]),
    basis: str = "sto-3g",
    gen_k: bool = True,
) -> PyscfCell:
    """Create PySCF cell for BCC hydrogen structure.

    Args:
        rs: Wigner-Seitz radius in atomic units
        ncopy: Array specifying supercell dimensions
        basis: Basis set specification string
        gen_k: Whether to enable space group symmetry for k-point generation

    Returns:
        Configured PySCF cell object
    """
    nh = np.prod(ncopy) * 2
    spins = [int(nh / 2), int(nh / 2)]
    cell = gto.Cell()
    cell.atom = make_atoms(rs, ncopy)
    cell.basis = basis
    cell.a = make_lattice(rs, ncopy)
    cell.unit = "B"
    cell.spin = spins[0] - spins[1]
    cell.verbose = 0
    cell.exp_to_discard = 0.1
    if gen_k:
        cell.space_group_symmetry = True
    cell.build()
    return cell


def get_config(input_str):
    """Parse input string and create a complete configuration object.

    This function creates a configuration object with all necessary parameters
    for crystal structure calculations, including network architecture, optimization
    settings, and MCMC parameters.

    Args:
        input_str: Comma-separated string containing configuration parameters:
            - rs: Wigner-Seitz radius
            - Sx,Sy,Sz: Supercell dimensions
            - basis: Basis set specification
            - batch_size: Batch size for training
            - nuclear_treatment: Nuclear treatment type ('fixed' or 'quantum')
            - infer: Whether in inference mode
            - is_rezero: Whether to use rezero initialization
            - local_sampling: Whether to use local sampling
            - atom_center_dynamic: Whether to use dynamic atom center

    Returns:
        Config: Configuration object with all parameters set

    """
    (
        rs,
        Sx,
        Sy,
        Sz,
        batch_size,
        nuclear_treatment,
        infer,
        ensemble,
        atom_center_dynamic,
        is_rezero,
    ) = input_str.split(",")

    # Create base configuration
    cfg = base_config.default()

    # Set crystal structure parameters
    cfg.crystal.structure = "bcc_D" if cfg.crystal.is_deuterium else "bcc_H"
    cfg.crystal.rs = float(rs)
    cfg.crystal.ncopy = [int(Sx), int(Sy), int(Sz)]
    ncopy_array = np.array(cfg.crystal.ncopy)
    cell = make_cell(cfg.crystal.rs, ncopy_array)
    kpts_class = cell.make_kpts(
        cfg.crystal.kpts.number, with_gamma_point=False, space_group_symmetry=True
    )
    twist_list = kpts_class.kpts_scaled_ibz
    cfg.crystal.kpts.length = len(twist_list)
    cfg.crystal.kpts.weights = kpts_class.weights_ibz[cfg.crystal.kpts.twist_index]
    assert len(twist_list) > cfg.crystal.kpts.twist_index
    cfg.network.twist = tuple(twist_list[cfg.crystal.kpts.twist_index])

    cfg.crystal.natm = cell.natm

    # Set up simulation cell
    simulation_cell = supercell.get_supercell(cell, np.diag([1, 1, 1]))
    cfg.system.pyscf_cell = simulation_cell

    # Set model and simulation parameters
    cfg.infer = bool(int(infer))
    cfg.ensemble = ensemble
    cfg.nuclear_treatment = nuclear_treatment
    cfg.batch_size = int(batch_size)
    logger.info(
        "Ensemble: %s, Nuclear treatment: %s",
        cfg.ensemble,
        cfg.nuclear_treatment,
    )

    # Set precision
    cfg.use_x64 = False
    cfg.precision = "float32" if not cfg.use_x64 else "float64"

    # Configure network architecture
    cfg.network.detnet.determinants = 16
    cfg.network.detnet.is_rezero = bool(int(is_rezero))
    cfg.network.detnet.atom_center_dynamic = bool(int(atom_center_dynamic))

    # Configure MCMC type based on nuclear_treatment type
    cfg.mcmc.mcmc_type = (
        "electron_only" if cfg.nuclear_treatment == "fixed" else "gibbs"
    )

    # Configure optimization settings based on inference mode
    if cfg.infer:
        # Inference mode: no optimization, load from checkpoint
        cfg.optim.optimizer = "none"
        cfg.optim.iterations = 10000
        save_name = utils.get_save_name(cfg)
        cfg.log.restore_path = save_name
        cfg.log.save_path = f"{save_name}/infer"
    else:
        # Training mode: use KFAC optimizer
        cfg.optim.optimizer = "kfac"
        cfg.optim.iterations = 600000
        cfg.log.restore_path = ""
        cfg.log.save_path = utils.get_save_name(cfg)

    # Set network hidden dimensions based on nuclear_treatment type
    if cfg.nuclear_treatment == "quantum":
        cfg.network.detnet.hidden_dims = ((256, 32, 32), (256, 32, 32), (256, 32, 32))
    elif cfg.nuclear_treatment == "fixed":
        cfg.network.detnet.hidden_dims = ((64, 32), (64, 32), (64, 32))

    return cfg
