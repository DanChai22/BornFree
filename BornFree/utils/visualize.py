# Copyright (c) 2025 Shengdu Chai
#
# Licensed under the Apache License, Version 2.0.

import operator
import os
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb
from ase import Atoms
from jax import Array
from pyscf.pbc.gto import Cell as PyscfCell

from BornFree import base_config, constants, estimator
from BornFree.network import network_block
from BornFree.utils.get_cell import SimulationCellGenerator
from BornFree.utils.units import pressure_estimator


# Data structures for better organization
class AnalysisResult(NamedTuple):
    """Container for analysis results."""

    metrics: dict[str, Any]
    ase_cell: Atoms
    batched_coords: Array | None
    mean_coords: Array
    lattice: Array
    reference_coords: Array
    reference_lattice: Array
    batch_size: int = 1


class PlotConfig(NamedTuple):
    """Configuration for plotting parameters."""

    save_path: str
    label_list: list[str]
    step: int
    crystal_name: str


def shift_atom_first(atoms_coords: Array, atoms_central_coords: Array, i: int) -> Array:
    """Shift the atoms_coords to the central atom wrt the first atom.

    Args:
        atoms_coords: Array of atomic coordinates with shape (num_atoms, 3)
        atoms_central_coords: Reference atomic coordinates with shape (num_atoms, 3)
        i: Index of the atom to use as the reference point

    Returns:
        Array: Shifted coordinates with shape (num_atoms, 3)

    """
    shifted_coords = atoms_coords[i] - atoms_central_coords[i]
    atoms_shifted_coords = atoms_coords - shifted_coords
    return atoms_shifted_coords


# Vectorized functions
vmap_enforce_pbc = jax.vmap(network_block.enforce_pbc, in_axes=(None, 0), out_axes=0)
vconvert = jax.vmap(
    network_block.convert_to_simulation_cell,
    in_axes=(None, 0, None),
    out_axes=(0, None),
)
batched_shift_atom_first = jax.vmap(shift_atom_first, in_axes=(0, None, None), out_axes=0)


def get_rs(lattice, cfg: base_config.BornFreeConfig) -> tuple[float, float]:
    """Calculate rs parameter and volume."""
    volume = jnp.linalg.det(lattice)
    rs = (volume / cfg.crystal.natm * 3.0 / (4.0 * jnp.pi)) ** (1.0 / 3.0)
    return float(rs), float(volume)


def _get_simulation_cell(rs: float, cfg: base_config.BornFreeConfig) -> PyscfCell:
    """Get simulation cell based on configuration.

    Args:
        rs: rs value
        cfg: ConfigDict

    Returns:
        Simulation cell object

    """
    generator = SimulationCellGenerator(np.array(cfg.crystal.ncopy))
    crystal_type = cfg.crystal.structure.split("_")[0]
    return (
        generator.get_simulation_cell(rs, crystal_type, cfg.crystal.cif_path)
        if cfg.crystal.cif_path is not None
        else generator.get_simulation_cell(rs, crystal_type)
    )


def _create_label_list(cfg: base_config.BornFreeConfig) -> list[str]:
    """Create standardized label list for file naming."""
    labels = [
        cfg.crystal.structure.split("_")[0],
        cfg.crystal.lattice.mode,
        str(cfg.target_pressure),
        str(cfg.crystal.natm),
        "".join(map(str, cfg.crystal.ncopy)),
    ]
    return labels


def _calculate_basic_metrics(
    cellpar: Array | None,
    cfg: base_config.BornFreeConfig,
    rs_init: float | None = None,
    kinetic: float | None = None,
    potential: float | None = None,
) -> tuple[dict[str, Any], float, Array]:
    """Calculate basic metrics and lattice from parameters or rs."""
    metrics = {}

    if cellpar is not None:
        lattice_current = network_block.get_jacobian(cellpar, cfg.crystal.lattice)
        rs_current, volume = get_rs(lattice_current, cfg)

        # Log lattice parameters
        for i, param in enumerate(
            ["a", "b", "c", "alpha", "beta", "gamma"] if cfg.crystal.lattice.mode == "angle" else ["a", "b", "c"]
        ):
            metrics[f"params_cell/{param}"] = float(cellpar[i])
    else:
        # Use provided rs for NVT case
        rs_current = rs_init or cfg.crystal.rs
        simulation_cell = _get_simulation_cell(rs_current, cfg)
        lattice_current = jnp.array(simulation_cell.lattice_vectors())
        volume = jnp.linalg.det(lattice_current)
    if kinetic is not None and potential is not None:
        _, p = pressure_estimator(kinetic.real, potential, volume)
        metrics["estimated_pressure"] = p

    metrics["rs"] = rs_current
    metrics["volume per atom[Å^3]"] = volume * constants.BOHR_TO_ANGSTROM**3 / cfg.crystal.natm

    return metrics, rs_current, lattice_current


def _get_reference_coordinates_and_lattice(rs: float, cfg: base_config.BornFreeConfig) -> Array:
    """Get reference atomic coordinates."""
    simulation_cell = _get_simulation_cell(rs, cfg)
    reference_lattice = jnp.array(simulation_cell.lattice_vectors())
    coords = jnp.array(simulation_cell.atom_coords())
    return network_block.enforce_pbc(reference_lattice, coords)[0], reference_lattice


def _process_coordinate_data(
    data: Array,
    cellpar: Array | None,
    cfg: base_config.BornFreeConfig,
    lattice: Array,
) -> tuple[Array, int]:
    """Process coordinate data from simulation."""
    if cellpar is not None:
        # NPT case - convert from unit cell
        batched_unit_coords = data[0][:, : cfg.crystal.natm * 3]
        batched_coords, _ = vconvert(cellpar, batched_unit_coords, cfg.crystal.lattice)
    else:
        # NVT case - already in simulation cell
        batched_coords = data[0][:, : cfg.crystal.natm * 3]

    batch_size = batched_coords.shape[0]
    coords_3d = batched_coords.reshape([batch_size, -1, 3])
    return vmap_enforce_pbc(lattice, coords_3d)[0], batch_size


def find_mean_atom_coords(positions: Array, Ls: Array) -> Array:
    """Calculate mean atomic coordinates under periodic boundary conditions."""
    _, natom, dim = positions.shape
    mean_positions = jnp.zeros((natom, dim))
    inv_Ls = jnp.linalg.inv(Ls)
    for atom in range(natom):
        ref = positions[0, atom, :].copy()
        unwrapped = positions[:, atom, :].copy()
        delta = unwrapped - ref
        unwrapped = ref + delta - jnp.dot(delta, inv_Ls).round() @ Ls
        avg_unwrapped = jnp.mean(unwrapped, axis=0)
        mean_positions = mean_positions.at[atom].set(avg_unwrapped)
    mean_positions = network_block.enforce_pbc(Ls, mean_positions)[0]
    return mean_positions


def coords_to_ase(coords: Array, Ls: Array) -> Atoms:
    """Convert coordinates to ASE Atoms object."""
    natom = coords.shape[0]
    symbols = ["H" for _ in range(natom)]
    return Atoms(symbols=symbols, positions=coords, cell=Ls, pbc=True)


def _get_analysis_result(
    data: Array | None,
    cellpar: Array | None = None,
    cfg: base_config.BornFreeConfig = None,
    rs: float | None = None,
    kinetic: float | None = None,
    potential: float | None = None,
) -> AnalysisResult:
    """Unified function to get analysis results for different simulation types."""
    # Calculate basic metrics and lattice
    metrics, rs_current, lattice = _calculate_basic_metrics(cellpar, cfg, rs, kinetic, potential)

    # Get reference coordinates
    reference_coords, reference_lattice = _get_reference_coordinates_and_lattice(rs_current, cfg)

    # Handle coordinate processing based on simulation type
    batched_coords, batch_size = _process_coordinate_data(data, cellpar, cfg, lattice)

    # Shift coordinates if needed
    if batched_coords.shape[0] > 1:  # Multiple configurations
        shifted_coords = batched_shift_atom_first(batched_coords, reference_coords, 0)
        batched_coords = vmap_enforce_pbc(lattice, shifted_coords)[0]
        mean_coords = find_mean_atom_coords(batched_coords, lattice)
    else:
        mean_coords = batched_coords[0]

    ase_cell = coords_to_ase(mean_coords, lattice)

    return AnalysisResult(
        metrics=metrics,
        ase_cell=ase_cell,
        batched_coords=batched_coords,
        mean_coords=mean_coords,
        lattice=lattice,
        reference_coords=reference_coords,
        reference_lattice=reference_lattice,
        batch_size=batch_size,
    )


def _create_projection_subplot(
    ax: plt.Axes,
    x: Array | np.ndarray,
    y: Array | np.ndarray,
    z: Array | np.ndarray,
    x_c: Array | np.ndarray | None = None,
    y_c: Array | np.ndarray | None = None,
    z_c: Array | np.ndarray | None = None,
    t: int | None = None,
    projection: str | None = None,
):
    """Helper function to create a projection subplot.

    Args:
        ax: matplotlib axis
        x: x-coordinates to plot
        y: y-coordinates to plot
        z: z-coordinates to plot
        x_c: reference x-coordinates (optional)
        y_c: reference y-coordinates (optional)
        z_c: reference z-coordinates (optional)
        t: step number
        projection: projection type ('xy', 'yz', 'xz', '3d')

    Returns:
        matplotlib scatter plot object

    """
    if projection == "3d":
        if x_c is not None:
            ax.scatter(x_c, y_c, z_c, c="r", label=f"step={t} Mean Configuration")
        scatter = ax.scatter(
            x,
            y,
            z,
            c=z,
            cmap="viridis",
            s=1,
            alpha=0.6,
            label=f"step={t} Configuration",
        )
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
    else:
        coords = {"xy": (x, y, z), "yz": (y, z, x), "xz": (x, z, y)}
        labels = {"xy": ("X", "Y"), "yz": ("Y", "Z"), "xz": ("X", "Z")}
        plot_x, plot_y, color = coords[projection]
        xlabel, ylabel = labels[projection]

        if x_c is not None:
            ref_x, ref_y = (x_c, y_c) if projection == "xy" else ((y_c, z_c) if projection == "yz" else (x_c, z_c))
            ax.scatter(ref_x, ref_y, c="r", label=f"step={t} Mean Configuration")

        scatter = ax.scatter(
            plot_x,
            plot_y,
            c=color,
            cmap="viridis",
            s=1,
            alpha=0.6,
            label=f"step={t} Configuration",
        )
        ax.set_xlabel(f"{xlabel}-axis")
        ax.set_ylabel(f"{ylabel}-axis")

    ax.legend(loc="upper right")
    return scatter


def _plot_projections_solid(
    atoms_central_coords: Array,
    atoms_coords: Array,
    cfg: base_config.BornFreeConfig,
    t: int,
    metrics: dict[str, Any],
) -> None:
    """Plot all projections of atomic coordinates.

    Args:
        atoms_central_coords: Reference atomic coordinates with shape (natm, 3)
        atoms_coords: Atomic coordinates with shape (natm, 3)
        cfg: Configuration object
        t: Current step number
        metrics: Dictionary for storing metrics

    """
    fig = plt.figure(figsize=(12, 10))
    x_c, y_c, z_c = atoms_central_coords.T
    x, y, z = atoms_coords.T

    projections = [("xy", 221), ("yz", 222), ("xz", 223), ("3d", 224)]
    scatters = []

    for proj, pos in projections:
        ax = fig.add_subplot(pos, projection="3d" if proj == "3d" else None)
        scatter = _create_projection_subplot(ax, x, y, z, x_c, y_c, z_c, t, proj)
        ax.set_title(f"{proj.upper()} Projection" if proj != "3d" else "3D View")
        scatters.append(scatter)

    plt.suptitle(f"All Projections of {cfg.crystal.structure.split('_')[0]}", y=0.95)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(scatters[0], cax=cbar_ax, label="Position")

    save_name_list = [
        "projections",
        cfg.crystal.structure,
        cfg.ensemble,
        str(cfg.target_pressure),
        cfg.crystal.lattice.mode,
    ]
    metrics["projections"] = wandb.Image(fig)
    fig.savefig(os.path.join(cfg.log.save_path, "_".join(save_name_list) + ".png"))
    plt.show()


def identify_molecule(
    atoms_coords: Array,
    Ls: Array,
    threshold_max: float,
    threshold_min: float,
) -> Array:
    """Identify diatomic molecules in a cell based on atomic coordinates.

    Args:
        atoms_coords: Coordinates of the atoms with shape (natm, 3)
        Ls: Lattice vectors with shape (3, 3)
        threshold_max: Distance threshold for bonding (default: DEFAULT_BOND_THRESHOLD_MAX)
        threshold_min: Distance threshold for bonding (default: DEFAULT_BOND_THRESHOLD_MIN)

    Returns:
        Array: Array of molecule indices that are bonded as molecules,
        e.g., [[0, 1], [2, 3]]. Atoms that are not paired (i.e. do not meet the
        bonding criterion) are ignored.with shape (n_molecules, 2)

    """
    natm = atoms_coords.shape[0]
    diff = atoms_coords[:, None, :] - atoms_coords[None, :, :]
    diff_nearest = diff - jnp.dot(diff, jnp.linalg.inv(Ls)).round() @ Ls
    pair_distance = jnp.linalg.norm(diff_nearest, axis=-1)

    candidate_bonds = []
    for i in range(natm):
        for j in range(i + 1, natm):
            d = float(pair_distance[i, j])
            if d < threshold_max and d > threshold_min:
                candidate_bonds.append((i, j, d))

    candidate_bonds.sort(key=operator.itemgetter(2))

    paired = set()
    molecules = []
    for i, j, _ in candidate_bonds:
        if i not in paired and j not in paired:
            molecules.append([i, j])
            paired.add(i)
            paired.add(j)

    return jnp.asarray(molecules)


def calculate_bond(molecules_index: Array, batched_atoms_coords: Array, Ls: Array) -> tuple[Array, Array]:
    """Calculate bond vectors and lengths for molecules.

    Args:
        molecules_index: Array of shape (n_molecule, 2) containing atom indices
        batched_atoms_coords: Array of shape (nbatch, natom, 3) with coordinates
        Ls: Lattice vectors for periodic boundary conditions

    Returns:
        Tuple of (vectors, norm) where vectors are bond vectors and norm are bond lengths
    """
    idx1 = molecules_index[:, 0]  # shape: (n_molecule,)
    idx2 = molecules_index[:, 1]  # shape: (n_molecule,)

    coords1 = batched_atoms_coords[:, idx1, :]
    coords2 = batched_atoms_coords[:, idx2, :]

    # Compute the orientation vector for each molecule: v = atom_j - atom_i.
    vectors = coords2 - coords1  # shape: (nbatch, n_molecule, 3)
    vectors = vectors - jnp.dot(vectors, jnp.linalg.inv(Ls)).round() @ Ls

    norm = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors, norm


def calculate_direction_vector(molecules_index: Array, batched_atoms_coords: Array, Ls: Array) -> Array:
    """Calculate the direction vector of the molecules."""
    vectors, norm = calculate_bond(molecules_index, batched_atoms_coords, Ls)
    # Avoid division by zero by replacing zeros with ones.
    norm = jnp.where(norm == 0, 1.0, norm)
    directions = vectors / norm  # shape: (nbatch, n_molecule, 3)

    return directions


def calculate_phi_distribution(
    molecules_index: Array,
    batched_atoms_coords: Array,
    Ls: Array,
    n_phi: int = constants.DEFAULT_ORIENTATION_BINS,
) -> tuple[Array, Array]:
    """Calculate the azimuthal angle (φ) distribution function P(φ) of the molecules.

    Args:
        molecules_index: Array of indices for the two atoms forming each molecule.
            Each row (i, j) indicates that atoms with indices i and j form a molecule with shape (n_molecule, 2).
        batched_atoms_coords: Atomic coordinates with shape (nbatch, natm, 3)
        Ls: Lattice vectors with shape (3, 3)
        n_phi: Number of phi bins (default: DEFAULT_ORIENTATION_BINS)

    Returns:
        Tuple containing:
            - phi_dist: jax.numpy.ndarray, shape (n_phi,)
                The normalized distribution function P(φ)
            - phi_centers: jax.numpy.ndarray, shape (n_phi,)
                The centers of phi bins, spanning from 0 to 2*pi

    """
    # Calculate the direction vector of the molecules.
    directions = calculate_direction_vector(molecules_index, batched_atoms_coords, Ls)

    x = directions[..., 0]
    y = directions[..., 1]

    # Calculate the azimuthal angle phi: phi = arctan2(y, x), initially in [-pi, pi].
    phi = jnp.arctan2(y, x)
    # Shift phi to be in [0, 2*pi].
    phi = jnp.where(phi < 0, phi + 2 * jnp.pi, phi)

    phi_flat = phi.reshape(-1)

    # Phi bins from 0 to 2*pi.
    phi_edges = jnp.linspace(0, 2 * jnp.pi, n_phi + 1)
    phi_centers = (phi_edges[:-1] + phi_edges[1:]) / 2

    # Convert to numpy for histogram calculation
    phi_np = np.array(phi_flat)
    phi_edges_np = np.array(phi_edges)

    # Calculate histogram
    hist, _ = np.histogram(phi_np, bins=phi_edges_np, density=True)

    # Convert the histogram back to a JAX array.
    phi_dist = jnp.array(hist)

    return phi_dist, phi_centers


def calculate_cos_theta_distribution(
    molecules_index: Array,
    batched_atoms_coords: Array,
    Ls: Array,
    n_cos_theta: int = constants.DEFAULT_ORIENTATION_BINS,
) -> tuple[Array, Array]:
    """Calculate the cos(θ) distribution function P(cos θ) of the molecules.

    Args:
        molecules_index: Array of indices for the two atoms forming each molecule.
            Each row (i, j) indicates that atoms with indices i and j form a molecule with shape (n_molecule, 2).
        batched_atoms_coords: Atomic coordinates with shape (nbatch, natm, 3)
        Ls: Lattice vectors with shape (3, 3)
        n_cos_theta: Number of cos(theta) bins (default: DEFAULT_ORIENTATION_BINS)

    Returns:
        Tuple containing:
            - cos_theta_dist: jax.numpy.ndarray, shape (n_cos_theta,)
                The normalized distribution function P(cos θ)
            - cos_theta_centers: jax.numpy.ndarray, shape (n_cos_theta,)
                The centers of cos(theta) bins, spanning from -1 to 1

    """
    directions = calculate_direction_vector(molecules_index, batched_atoms_coords, Ls)

    # z component is cos(theta)
    cos_theta = directions[..., 2]
    cos_theta_flat = cos_theta.reshape(-1)

    # Cos(theta) bins from -1 to 1
    cos_theta_edges = jnp.linspace(-1, 1, n_cos_theta + 1)
    cos_theta_centers = (cos_theta_edges[:-1] + cos_theta_edges[1:]) / 2

    # Convert to numpy for histogram calculation
    cos_theta_np = np.array(cos_theta_flat)
    cos_theta_edges_np = np.array(cos_theta_edges)

    # Calculate histogram
    hist, _ = np.histogram(cos_theta_np, bins=cos_theta_edges_np, density=True)

    # Convert the histogram back to a JAX array.
    cos_theta_dist = jnp.array(hist)

    return cos_theta_dist, cos_theta_centers


def calculate_distribution_function(
    molecules_index: Array,
    batched_atoms_coords: Array,
    Ls: Array,
    n_theta: int = 50,
    n_phi: int = 50,
) -> tuple[Array, Array, Array]:
    """Calculate the orientation distribution function g(theta, phi) of the molecules.

    Args:
        molecules_index: Array of indices for the two atoms forming each molecule.
        Each row (i, j) indicates that atoms with indices i and j form a molecule with shape (n_molecule, 2).
        batched_atoms_coords: Atomic coordinates with shape (nbatch, natm, 3)
        Ls: Lattice vectors with shape (3, 3)
        n_theta: Number of theta bins (default: 50)
        n_phi: Number of phi bins (default: 50)

    Returns:
        Tuple containing:
            - g: jax.numpy.ndarray, shape (n_theta, n_phi)
        The normalized orientation distribution function (i.e., a probability density
        function) of the molecular orientations.
            - theta_edges: jax.numpy.ndarray, shape (n_theta+1,)
        Bin edges for the polar angle theta, spanning from 0 to pi.
            - phi_edges: jax.numpy.ndarray, shape (n_phi+1,)
        Bin edges for the azimuthal angle phi, spanning from 0 to 2*pi.

    """
    directions = calculate_direction_vector(molecules_index, batched_atoms_coords, Ls)
    x = directions[..., 0]
    y = directions[..., 1]
    z = directions[..., 2]

    # Calculate the polar angle theta: theta = arccos(z), with z in [-1,1] so theta is in [0, pi].
    theta = jnp.arccos(z)

    # Calculate the azimuthal angle phi: phi = arctan2(y, x), initially in [-pi, pi].
    phi = jnp.arctan2(y, x)
    # Shift phi to be in [0, 2*pi].
    phi = jnp.where(phi < 0, phi + 2 * jnp.pi, phi)

    theta_flat = theta.reshape(-1)
    phi_flat = phi.reshape(-1)

    # Theta bins from 0 to pi and phi bins from 0 to 2*pi.
    theta_edges = jnp.linspace(0, jnp.pi, n_theta + 1)
    phi_edges = jnp.linspace(0, 2 * jnp.pi, n_phi + 1)

    theta_np = np.array(theta_flat)
    phi_np = np.array(phi_flat)
    theta_edges_np = np.array(theta_edges)
    phi_edges_np = np.array(phi_edges)

    H, theta_edges_hist, phi_edges_hist = np.histogram2d(
        theta_np, phi_np, bins=[theta_edges_np, phi_edges_np], density=True
    )

    # Convert the histogram back to a JAX array.
    g = jnp.array(H)
    return g, jnp.array(theta_edges_hist), jnp.array(phi_edges_hist)


def _plot_orientation_distribution(
    g: Array,
    theta_edges: Array,
    phi_edges: Array,
    title_suffix: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """Helper function to plot orientation distribution.

    Args:
        g: Orientation distribution function with shape (n_theta, n_phi)
        theta_edges: Bin edges for theta
        phi_edges: Bin edges for phi
        title_suffix: Additional text for plot title
        save_path: Path to save the plot

    Returns:
        matplotlib Figure object

    """
    plt.figure(figsize=(8, 6))
    theta_edges_np = np.array(theta_edges)
    phi_edges_np = np.array(phi_edges)

    mesh = plt.pcolormesh(
        phi_edges_np / np.pi * 180,
        theta_edges_np / np.pi * 180,
        np.array(g),
        shading="auto",
        cmap="viridis",
    )
    plt.xlabel(r"Azimuthal angle, $\phi$ (degrees)")
    plt.ylabel(r"Polar angle, $\theta$ (degrees)")
    plt.title(f"Orientation Distribution Function $g(\\theta,\\phi)$ {title_suffix}")
    plt.colorbar(mesh, label="Density")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    return plt.gcf()


def _plot_phi_distribution(
    phi_dist: Array,
    phi_centers: Array,
    title_suffix: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot the azimuthal angle (φ) distribution function P(φ) of the molecules.

    Args:
        phi_dist: Orientation distribution function with shape (n_phi,)
        phi_centers: Bin centers for phi
        title_suffix: Additional text for plot title
        save_path: Path to save the plot

    Returns:
        matplotlib.pyplot.Figure: The figure object

    """
    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(phi_centers, phi_dist, "b-", linewidth=2)
    ax.set_xlabel(r"$\phi$ (rad)", fontsize=14)
    ax.set_ylabel(r"$P(\phi)$", fontsize=14)
    ax.set_title(f"Azimuthal Angle ($\\phi$) Distribution {title_suffix}", fontsize=16)
    ax.grid(True, alpha=0.3)

    # Convert to degrees for x-ticks
    phi_degrees = np.linspace(0, 360, 7)
    phi_radians = np.deg2rad(phi_degrees)
    ax.set_xticks(phi_radians)
    ax.set_xticklabels([f"{int(deg)}°" for deg in phi_degrees])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return plt.gcf()


def _plot_cos_theta_distribution(
    cos_theta_dist: Array,
    cos_theta_centers: Array,
    title_suffix: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot the cos(θ) distribution function P(cos θ) of the molecules.

    Args:
        cos_theta_dist: Cosine distribution function with shape (n_cos_theta,)
        cos_theta_centers: Bin centers for cos(theta)
        title_suffix: Additional text for plot title
        save_path: Path to save the plot

    Returns:
        matplotlib.pyplot.Figure: The figure object

    """
    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cos_theta_centers, cos_theta_dist, "r-", linewidth=2)
    ax.set_xlabel(r"$\cos(\theta)$", fontsize=14)
    ax.set_ylabel(r"$P(\cos(\theta))$", fontsize=14)
    ax.set_title(f"Polar Angle ($\\cos(\\theta)$) Distribution {title_suffix}", fontsize=16)
    ax.grid(True, alpha=0.3)

    # Add secondary x-axis with theta in degrees
    ax2 = ax.twiny()
    cos_theta_ticks = np.array([-1, -0.866, -0.5, 0, 0.5, 0.866, 1])
    theta_degrees = np.rad2deg(np.arccos(cos_theta_ticks))
    ax2.set_xticks(cos_theta_ticks)
    ax2.set_xticklabels([f"{int(deg)}°" for deg in theta_degrees])
    ax2.set_xlabel(r"$\theta$ (degrees)", fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return plt.gcf()


def plot_orientation_distributions(
    mean_atoms_coords: Array,
    batched_atoms_coords: Array,
    Ls: Array,
    cfg: base_config.BornFreeConfig,
    metrics: dict[str, Any],
) -> None:
    """Plot orientation distribution functions for both mean and all configurations.

    Args:
        mean_atoms_coords: Mean atomic coordinates with shape (natm, 3)
        batched_atoms_coords: Coordinates for all configurations with shape (nbatch, natm, 3)
        Ls: Lattice vectors with shape (3, 3)
        cfg: Configuration object
        metrics: Dictionary for storing metrics

    """
    molecules_index = identify_molecule(
        mean_atoms_coords,
        Ls,
        threshold_max=constants.DEFAULT_MOLECULE_THRESHOLD_MAX,
        threshold_min=constants.DEFAULT_MOLECULE_THRESHOLD_MIN,
    )
    if molecules_index.shape[0] == 0:
        metrics["n_molecules_ratio"] = 0.0
        return None
    metrics["n_molecules_ratio"] = molecules_index.shape[0] / cfg.crystal.natm * 2

    g, theta_edges, phi_edges = calculate_distribution_function(molecules_index, batched_atoms_coords, Ls)

    phi_dist, phi_centers = calculate_phi_distribution(molecules_index, batched_atoms_coords, Ls)
    cos_theta_dist, cos_theta_centers = calculate_cos_theta_distribution(molecules_index, batched_atoms_coords, Ls)

    save_name_list = [
        "orientation_distribution_function",
        cfg.crystal.structure.split("_")[0],
        cfg.crystal.lattice.mode,
        str(cfg.target_pressure),
        "".join(map(str, cfg.crystal.ncopy)),
    ]
    base_path = os.path.join(cfg.log.save_path, "_".join(save_name_list))

    # Plot full distribution
    fig = _plot_orientation_distribution(g, theta_edges, phi_edges, save_path=base_path + ".png")
    metrics["orientation_distribution_function"] = wandb.Image(fig)
    plt.show()

    fig = _plot_phi_distribution(phi_dist, phi_centers, title_suffix="(1D)", save_path=base_path + "_phi.png")
    metrics["phi_distribution"] = wandb.Image(fig)
    plt.show()

    fig = _plot_cos_theta_distribution(
        cos_theta_dist,
        cos_theta_centers,
        title_suffix="(1D)",
        save_path=base_path + "_cos_theta.png",
    )
    metrics["cos_theta_distribution"] = wandb.Image(fig)
    plt.show()


def _save_and_log_plot(
    fig: plt.Figure,
    save_path: str,
    metrics_key: str,
    metrics: dict[str, Any],
    show_plot: bool = True,
) -> None:
    """Helper to save plot, log to wandb, and optionally show."""
    metrics[metrics_key] = wandb.Image(fig)
    fig.savefig(save_path)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _unified_plot_analysis(
    result: AnalysisResult,
    cfg: base_config.BornFreeConfig,
    step: int,
    plot_xrd: bool = True,
    plot_projections: bool = True,
    plot_orientations: bool = True,
) -> None:
    """Unified plotting function for all analysis types."""
    plot_config = PlotConfig(
        save_path=cfg.log.save_path,
        label_list=_create_label_list(cfg),
        step=step,
        crystal_name=cfg.crystal.structure.split("_")[0],
    )

    if plot_xrd:
        _plot_analysis_xrd(result, cfg, plot_config)

    if plot_projections:
        _plot_projections_solid(
            result.mean_coords,
            result.batched_coords.reshape([-1, 3]),
            cfg,
            plot_config.step,
            result.metrics,
        )

    if plot_orientations:
        plot_orientation_distributions(
            result.mean_coords,
            result.batched_coords,
            result.lattice,
            cfg,
            result.metrics,
        )


def _plot_analysis_xrd(result: AnalysisResult, cfg: base_config.BornFreeConfig, plot_config: PlotConfig) -> None:
    """Plot X-ray diffraction pattern analysis."""
    xrd = estimator.get_xrd(
        wavelength=constants.DEFAULT_XRD_WAVELENGTH,
        two_theta_range=constants.DEFAULT_XRD_RANGE,
        step=constants.DEFAULT_XRD_STEP,
        gamma=constants.DEFAULT_XRD_GAMMA,
    )

    # Calculate XRD for current data
    d_spacing, averaged_intensity = xrd.compute_xrd(result.batched_coords, result.lattice, n_samples=result.batch_size)
    # Create plot
    plt.figure(figsize=(8, 4))
    if d_spacing is not None:
        plt.plot(d_spacing, averaged_intensity, label=f"step={plot_config.step}")
    if hasattr(cfg, "xrd_exp") and cfg.xrd_exp is not None:
        for d in cfg.xrd_exp:
            plt.axvline(x=d, color="k", linestyle="--")
    plt.xlabel("d-spacing (Å)")
    plt.ylabel("Intensity (a.u.)")
    plt.legend()

    save_path = os.path.join(
        plot_config.save_path,
        f"XRD_{'_'.join(plot_config.label_list)}.png",
    )
    _save_and_log_plot(plt.gcf(), save_path, "XRD", result.metrics, show_plot=False)


def plot_npt_result(
    data: Array,
    params: dict[str, Any],
    t: int,
    cfg: base_config.BornFreeConfig,
    kinetic: float | None,
    potential: float | None,
) -> Atoms:
    """Plot various analysis results for NPT simulation."""
    params = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0, keepdims=False), params)
    cellpar = params.get("cell", None)
    data = data.reshape((-1, cfg.batch_size, data.shape[-1]))

    # Get unified analysis result
    result = _get_analysis_result(data, cellpar, cfg, rs=None, kinetic=kinetic, potential=potential)

    # Save CIF file
    label_list = _create_label_list(cfg)
    result.ase_cell.write(os.path.join(cfg.log.save_path, "_".join(label_list) + ".cif"))

    # Unified plotting
    _unified_plot_analysis(result, cfg, t)

    # Log metrics
    wandb.log(result.metrics, step=t)
    return result.ase_cell


def plot_nvt_quantum_result(
    data: Array,
    t: int,
    cfg: base_config.BornFreeConfig,
    kinetic: float | None,
    potential: float | None,
) -> Atoms:
    """Plot various analysis results for NVT quantum simulation."""
    # Get unified analysis result
    result = _get_analysis_result(data, None, cfg, cfg.crystal.rs, kinetic, potential)

    # Save CIF file
    label_list = _create_label_list(cfg)
    result.ase_cell.write(os.path.join(cfg.log.save_path, "_".join(label_list) + ".cif"))

    # Unified plotting (skip mean projections and orientations for NVT quantum)
    _unified_plot_analysis(result, cfg, t, plot_orientations=False)
    wandb.log(result.metrics, step=t)
    return result.ase_cell
