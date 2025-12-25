# Copyright (c) 2025 Shengdu Chai
#
# Licensed under the Apache License, Version 2.0.

import logging

import ase.io
import numpy as np
from pyscf.pbc import gto
from pyscf.pbc.gto import Cell as PyscfCell

from BornFree import base_config, supercell
from BornFree.config import utils

logger = logging.getLogger(__name__)


def eval_rs(ase_cell, nh):
    """Calculate Wigner-Seitz radius from cell volume and number of atoms.

    Args:
        ase_cell: ASE cell object containing crystal structure information
        nh: Number of hydrogen atoms in the cell

    Returns:
        Wigner-Seitz radius in atomic units

    """
    volume = ase_cell.get_volume()
    return (volume / (nh * 4 * np.pi / 3)) ** (1 / 3)


def make_supercell_atoms(original_lattice, original_atom, ncopy):
    """Create atomic coordinates for a supercell from the original cell.

    Args:
        original_lattice: Lattice vectors as ndarray of shape (3,3)
        original_atom: List of tuples (atom_name, coordinates) for each atom
        ncopy: Number of copies along each direction, array-like of shape (3,)

    Returns:
        List of (atom_name, coordinates) tuples for the supercell

    """
    assert len(ncopy) == 3
    S = np.diag(ncopy)
    Rpts = supercell.get_supercell_copies(original_lattice, S)
    atom = []
    for name, xyz in original_atom:
        atom.extend([(name, xyz + R) for R in Rpts])
    return atom


def make_supercell_lattice(original_lattice, ncopy):
    """Generate lattice vectors for a supercell.

    Args:
        original_lattice: Original lattice vectors as ndarray of shape (3,3)
        ncopy: Number of copies along each direction, array-like of shape (3,)

    Returns:
        New lattice vectors for the supercell

    """
    assert len(ncopy) == 3
    S = np.diag(ncopy)
    return np.dot(S, original_lattice)


def make_scaled_ase_cell(ase_cell, rs):
    """Scale an ASE cell to match a target Wigner-Seitz radius.

    Args:
        ase_cell: ASE cell object containing crystal structure
        rs: Target Wigner-Seitz radius in atomic units (float)

    Returns:
        Tuple of (number_of_atoms, scaled_lattice_vectors, scaled_atomic_positions)

    """
    nh = ase_cell.get_global_number_of_atoms()
    rs_ref = eval_rs(ase_cell, nh)
    lattice_vectors = ase_cell.cell * (rs / rs_ref)
    positions = ase_cell.get_positions()
    symbols = ase_cell.get_chemical_symbols()
    atom = [(symbols[i], positions[i] * (rs / rs_ref)) for i in range(len(symbols))]
    return nh, lattice_vectors, atom


def make_identity_cell(ase_cell):
    """Create a PySCF cell object from an ASE cell with specified parameters.

    Args:
        ase_cell: ASE cell object containing crystal structure

    """
    nh = ase_cell.get_global_number_of_atoms()
    lattice_vectors = ase_cell.cell
    positions = ase_cell.get_positions()
    symbols = ase_cell.get_chemical_symbols()
    atom = [(symbols[i], positions[i]) for i in range(len(symbols))]
    return nh, lattice_vectors, atom


def make_cell(
    ase_cell,
    rs: float,
    ncopy: np.ndarray,
    basis: str = "sto-3g",
    gen_k: bool = True,
    identity: bool = False,
) -> PyscfCell:
    """Create a PySCF cell object from an ASE cell with specified parameters.

    Args:
        ase_cell: ASE cell object containing crystal structure
        rs: Target Wigner-Seitz radius in atomic units
        ncopy: Array specifying supercell dimensions
        basis: Basis set specification string
        gen_k: Whether to enable space group symmetry for k-point generation
        identity: Whether to use identity scaling for lattice vectors

    Returns:
        Configured PySCF cell object

    """
    if identity:
        nh, lattice_vectors, atom = make_identity_cell(ase_cell)
    else:
        nh, lattice_vectors, atom = make_scaled_ase_cell(ase_cell, rs)
    supercell_lattice = make_supercell_lattice(lattice_vectors, ncopy)
    supercell_atom = make_supercell_atoms(lattice_vectors, atom, ncopy)
    supercell_nh = nh * np.prod(ncopy)
    cell = gto.Cell()
    cell.atom = supercell_atom
    cell.a = np.asarray(supercell_lattice)
    cell.unit = "B"
    cell.basis = basis
    spins = [int(supercell_nh / 2), int(supercell_nh / 2)]
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
            - cif_path: Path to the CIF file
            - rs: Wigner-Seitz radius
            - Sx,Sy,Sz: Supercell dimensions
            - basis: Basis set specification
            - batch_size: Batch size for training
            - nuclear_treatment: Nuclear treatment type ('fixed' or 'quantum')
            - infer: Whether in inference mode
            - lattice_mode: Mode for lattice operations
            - pressure: Target pressure in GPa
            - warmup_steps: Number of warmup steps
            - opt_steps: Number of optimization steps
            - geo_opt_steps: Number of geometry optimization steps
            - is_rezero: Whether to use rezero initialization
            - local_sampling: Whether to use local sampling
            - atom_center_dynamic: Whether to use dynamic atom center

    Returns:
        Config: Configuration object with all parameters set

    """
    (
        cif_path,
        rs,
        Sx,
        Sy,
        Sz,
        batch_size,
        nuclear_treatment,
        infer,
        ensemble,
        lattice_mode,
        pressure,
        warmup_steps,
        opt_steps,
        geo_opt_steps,
        is_rezero,
        local_sampling,
        atom_center_dynamic,
    ) = input_str.split(",")

    # Create base configuration with default values
    cfg = base_config.default()

    # Configure crystal structure parameters
    cfg.crystal.cif_path = cif_path
    structure = cfg.crystal.cif_path.split("/")[-1].split(".")[0].split("_")[0]
    cfg.crystal.structure = f"{structure}_D" if cfg.crystal.is_deuterium else f"{structure}_H"
    cfg.crystal.rs = float(rs)
    cfg.crystal.ncopy = [int(Sx), int(Sy), int(Sz)]
    ncopy_array = np.array(cfg.crystal.ncopy)
    ase_cell = ase.io.read(cfg.crystal.cif_path)
    cell = make_cell(ase_cell, cfg.crystal.rs, ncopy_array)
    kpts_class = cell.make_kpts(cfg.crystal.kpts.number, with_gamma_point=False, space_group_symmetry=True)
    twist_list = kpts_class.kpts_scaled_ibz
    cfg.crystal.kpts.length = len(twist_list)
    cfg.crystal.kpts.weights = kpts_class.weights_ibz[cfg.crystal.kpts.twist_index]
    assert len(twist_list) > cfg.crystal.kpts.twist_index
    cfg.network.twist = tuple(twist_list[cfg.crystal.kpts.twist_index])

    # Set up simulation cell
    simulation_cell = supercell.get_supercell(cell, np.diag([1, 1, 1]))
    cfg.system.pyscf_cell = simulation_cell
    latvec = simulation_cell.lattice_vectors()  # assumming each row is a lattice vector
    cfg.crystal.lattice.a = float(np.linalg.norm(latvec[0]))
    cfg.crystal.lattice.b = float(np.linalg.norm(latvec[1]))
    cfg.crystal.lattice.c = float(np.linalg.norm(latvec[2]))
    cfg.crystal.lattice.alpha = float(
        np.arccos(np.dot(latvec[1], latvec[2]) / (cfg.crystal.lattice.b * cfg.crystal.lattice.c))
    )
    cfg.crystal.lattice.beta = float(
        np.arccos(np.dot(latvec[0], latvec[2]) / (cfg.crystal.lattice.a * cfg.crystal.lattice.c))
    )
    cfg.crystal.lattice.gamma = float(
        np.arccos(np.dot(latvec[0], latvec[1]) / (cfg.crystal.lattice.a * cfg.crystal.lattice.b))
    )
    cfg.crystal.lattice.mode = lattice_mode
    cfg.crystal.natm = cell.natm

    # Configure simulation parameters
    cfg.infer = bool(int(infer))
    cfg.ensemble = ensemble
    logger.info("ensemble is %s", cfg.ensemble)
    cfg.target_pressure = float(pressure)
    cfg.xrd_exp = [1.46, 1.54, 1.66] if np.isclose(cfg.target_pressure, 95.0) else None
    cfg.use_x64 = False
    cfg.precision = "float32" if not cfg.use_x64 else "float64"

    # Set network architecture parameters
    cfg.network.detnet.determinants = 8
    cfg.debug.deterministic = False
    cfg.nuclear_treatment = nuclear_treatment
    cfg.batch_size = int(batch_size)

    cfg.network.detnet.is_rezero = bool(int(is_rezero))
    cfg.network.detnet.atom_center_dynamic = bool(int(atom_center_dynamic))

    # Configure optimization parameters
    cfg.optim.lr.rate = 3e-3

    cfg.mcmc.annealing.local_sampling = bool(int(local_sampling))

    # Set MCMC parameters
    cfg.mcmc.mcmc_type = "electron_only" if cfg.nuclear_treatment == "fixed" else "gibbs"

    cfg.mcmc.steps = 10
    cfg.mcmc.iter = 10
    cfg.mcmc.elec_init_width = 0.8 / 25
    cfg.mcmc.atom_init_width = 0.0008

    cfg.strategy.warmup_steps = int(warmup_steps)
    cfg.strategy.opt_steps = int(opt_steps)
    cfg.strategy.geo_opt_steps = int(geo_opt_steps)

    # Configure paths and optimization settings based on mode
    if cfg.infer:
        cfg.optim.optimizer = "none"
        cfg.optim.iterations = 10000
        cfg.log.restore_path = utils.get_save_name(cfg)
        cfg.log.save_path = utils.get_save_name(cfg) + "/infer"
    else:
        cfg.optim.optimizer = "kfac"
        cfg.optim.iterations = 500000
        cfg.log.save_path = utils.get_save_name(cfg)

    # Configure network architecture dimensions based on nuclear_treatment type
    if cfg.nuclear_treatment == "quantum":
        cfg.network.detnet.hidden_dims = ((256, 32, 32), (256, 32, 32), (256, 32, 32))
    elif cfg.nuclear_treatment == "fixed":
        cfg.network.detnet.hidden_dims = ((64, 32), (64, 32), (64, 32))

    return cfg
