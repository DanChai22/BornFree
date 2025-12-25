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
"""Implementation of Fermionic Neural Network blocks and utilities in JAX."""

import functools
import operator
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import chex
import jax
import jax.numpy as jnp
from jax import Array
from pyscf.pbc.gto import Cell as PyscfCell

from BornFree import curvature_tags_and_blocks
from BornFree.base_config import CrystalLatticeConfig

FermiLayers = tuple[tuple[int, int], ...]
# Recursive types are not yet supported in pytype - b/109648354.
# pytype: disable=not-supported-yet  # noqa: ERA001
ParamTree = Array | Iterable["ParamTree"] | Mapping[Any, "ParamTree"]
# pytype: enable=not-supported-yet  # noqa: ERA001
# init(key) -> params
FermiNetInit = Callable[[Array], ParamTree]
# network(params, x) -> sign_out, log_out
FermiNetApply = Callable[[ParamTree, Array], tuple[Array, Array]]


def eval_rs(simulation_cell: PyscfCell) -> float:
    """Calculates the Wigner-Seitz radius (rs).

    Args:
        simulation_cell: PySCF Cell object containing lattice vectors and number of atoms.

    Returns:
        The Wigner-Seitz radius (rs) of the simulation cell.

    """
    volume = jnp.linalg.det(simulation_cell.lattice_vectors())
    rs = (volume / simulation_cell.natm * 3.0 / (4.0 * jnp.pi)) ** (1.0 / 3.0)
    return float(rs)


def enforce_pbc(latvec: Array, epos: Array) -> tuple[Array, Array]:
    """Enforces periodic boundary conditions on a set of coordinates.

    Args:
        latvec: Orthogonal lattice vectors defining the 3D torus. Shape (3, 3).
        epos: Coordinates to be wrapped. Shape (N, 3).

    Returns:
        Tuple of (final_epos, wrap), where:
          final_epos: Coordinates with PBCs imposed. Shape (N, 3).
          wrap: Integer array indicating the number of lattice vector shifts. Shape (N, 3).

    """
    # Writes epos in terms of (lattice vecs) fractional coordinates
    recpvecs = jnp.linalg.inv(latvec)
    epos_lvecs_coord = jnp.matmul(epos, recpvecs)
    wrap = jax.lax.floor(epos_lvecs_coord)
    final_epos = jnp.matmul(epos_lvecs_coord - wrap, latvec)

    return final_epos, wrap


def scaled_f(w: Array) -> Array:
    """Computes the scaling function f(w) for periodic distances.

    See Phys. Rev. B 94, 035157.

    Args:
        w: Projection of position vectors on reciprocal vectors.

    Returns:
        The value of the scaling function f.

    """
    return jnp.abs(w) * (1 - jnp.abs(w / jnp.pi) ** 3 / 4.0)


def scaled_g(w: Array) -> Array:
    """Computes the scaling function g(w) for periodic distances.

    See Phys. Rev. B 94, 035157.

    Args:
        w: Projection of position vectors on reciprocal vectors.

    Returns:
        The value of the scaling function g.

    """
    return w * (
        1 - 3.0 / 2.0 * jnp.abs(w / jnp.pi) + 1.0 / 2.0 * jnp.abs(w / jnp.pi) ** 2
    )


def nu_distance(xea: Array, a: Array, b: Array) -> tuple[Array, Array]:
    """Computes periodic generalized relative and absolute distance using the 'nu' method.

    See Phys. Rev. B 94, 035157.

    Args:
        xea: Relative distance between electrons and atoms (or particles).
        a: Lattice vectors of simulation cell divided by 2 pi.
        b: Reciprocal vectors of simulation cell.

    Returns:
        Tuple of (sd, rel), where:
          sd: Periodic generalized absolute distance.
          rel: Periodic generalized relative distance vector.

    """
    w = jnp.matmul(xea, jnp.transpose(b))
    mod = jax.lax.floor((w + jnp.pi) / (2 * jnp.pi))
    w = w - mod * 2 * jnp.pi
    r1 = (jnp.linalg.norm(a, axis=-1) * scaled_f(w)) ** 2
    sg = scaled_g(w)
    rel = jnp.matmul(sg, a)
    r2 = jnp.matmul(a, jnp.transpose(a)) * (sg[..., :, None] * sg[..., None, :])
    result = jnp.sum(r1, axis=-1) + jnp.sum(
        r2 * (jnp.ones(r2.shape[-2:]) - jnp.eye(r2.shape[-1])), axis=[-1, -2]
    )
    sd = result**0.5
    return sd, rel


def tri_distance(xea: Array, a: Array, b: Array) -> tuple[Array, Array]:
    """Computes periodic generalized relative and absolute distance using the 'tri' method.

    See Phys. Rev. Lett. 130, 036401 (2023).

    Args:
        xea: Relative distance between electrons and atoms (or particles).
        a: Lattice vectors of simulation cell divided by 2 pi.
        b: Reciprocal vectors of simulation cell.

    Returns:
        Tuple of (sd, rel), where:
          sd: Periodic generalized absolute distance.
          rel: Periodic generalized relative distance vector (concatenated sin and cos components).

    """
    w = jnp.matmul(xea, jnp.transpose(b))
    sg = jnp.sin(w)
    cg = jnp.cos(w)
    rel_sin = jnp.matmul(sg, a)
    rel_cos = jnp.matmul(cg, a)
    rel = jnp.concatenate([rel_sin, rel_cos], axis=-1)
    metric = jnp.matmul(a, jnp.transpose(a))
    vector_sin = sg[..., :, None] * sg[..., None, :]
    vector_cos = (1 - cg[..., :, None]) * (1 - cg[..., None, :])
    vector = vector_cos + vector_sin
    sd = jnp.sum(vector * metric, axis=(-2, -1)) ** 0.5
    return sd, rel


def get_distance_function(distance_type: str) -> Callable:
    """Returns the appropriate distance function based on the specified type.

    Args:
        distance_type: The type of distance function ('nu' or 'tri').

    Returns:
        The corresponding distance function.

    Raises:
        ValueError: If an unrecognized distance_type is provided.

    """
    if distance_type == "nu":
        return nu_distance
    elif distance_type == "tri":
        return tri_distance
    else:
        raise ValueError(f"Unrecognized distance type: {distance_type}")


def construct_symmetric_features(
    h_one: Array, h_two: Array, spins: tuple[int, int]
) -> Array:
    """Combines intermediate features from rank-one and rank-two streams.

    Args:
        h_one: Set of one-electron features. Shape: (nelectrons, n1), where n1 is
            the output size of the previous layer.
        h_two: Set of two-electron features. Shape: (nelectrons, nelectrons, n2),
            where n2 is the output size of the previous layer.
        spins: Tuple containing the number of spin-up and spin-down electrons.

    Returns:
        Array containing the permutation-equivariant features: the input set of
        one-electron features, the mean of the one-electron features over each
        (occupied) spin channel, and the mean of the two-electron features over each
        (occupied) spin channel. Output shape (nelectrons, 3*n1 + 2*n2) if there are
        both spin-up and spin-down electrons and (nelectrons, 2*n1+n2) otherwise.

    """
    # Split features into spin up and spin down electrons
    h_ones = jnp.split(
        h_one, spins[0:1], axis=0
    )  # List[array(spinup electrons, n1), array(spindown electrons, n1)]
    h_twos = jnp.split(
        h_two, spins[0:1], axis=0
    )  # List[array(spinup electrons, nelectrons, n2), array(spindown electrons, nelectrons, n2)]

    # Construct inputs to next layer
    # h.size == 0 corresponds to unoccupied spin channels.
    g_one = [
        jnp.mean(h, axis=0, keepdims=True) for h in h_ones if h.size > 0
    ]  # shape (1, n1)
    g_two = [
        jnp.mean(h, axis=0) for h in h_twos if h.size > 0
    ]  # shape (nelectrons, n2)

    g_one = [jnp.tile(g, [h_one.shape[0], 1]) for g in g_one]  # shape (nelectrons, n1)

    return jnp.concatenate(
        [h_one, *g_one, *g_two], axis=1
    )  # shape (nelectrons, 2*n1 + n2)


def _compute_periodic_features_core(
    sim_elec: Array,
    sim_atom: Array,
    sim_AV: Array,
    sim_BV: Array,
    distance_func: Callable,
    mask_elec: Array | None = None,
    mask_atom: Array | None = None,
    compute_aa: bool = True,
) -> tuple[Array, Array, Array | None, Array, Array, Array | None]:
    """Core logic to compute periodic features."""
    # ea (Electron-Atom)  # noqa: ERA001
    sim_xea = sim_elec[..., None, :] - sim_atom
    sim_periodic_sea, sim_periodic_xea = distance_func(sim_xea, sim_AV, sim_BV)
    sim_periodic_sea = sim_periodic_sea[..., None]

    # ee (Electron-Electron)  # noqa: ERA001
    nelec = sim_elec.shape[0]
    if mask_elec is None:
        mask_elec = 1.0 - jnp.eye(nelec, dtype=sim_elec.dtype)

    sim_xee = sim_elec[:, None, :] - sim_elec[None, :, :]
    sim_periodic_see, sim_periodic_xee = distance_func(
        sim_xee + jnp.eye(nelec, dtype=sim_elec.dtype)[..., None], sim_AV, sim_BV
    )
    sim_periodic_see = sim_periodic_see * mask_elec
    sim_periodic_see = sim_periodic_see[..., None]
    sim_periodic_xee = sim_periodic_xee * mask_elec[..., None]

    # aa (Atom-Atom)  # noqa: ERA001
    sim_periodic_xaa = None
    sim_periodic_saa = None

    if compute_aa:
        natom = sim_atom.shape[0]
        if mask_atom is None:
            mask_atom = 1.0 - jnp.eye(natom, dtype=sim_atom.dtype)

        sim_xaa = sim_atom[:, None, :] - sim_atom[None, :, :]
        sim_periodic_saa, sim_periodic_xaa = distance_func(
            sim_xaa + jnp.eye(natom, dtype=sim_atom.dtype)[..., None], sim_AV, sim_BV
        )
        sim_periodic_saa = sim_periodic_saa * mask_atom
        sim_periodic_saa = sim_periodic_saa[..., None]
        sim_periodic_xaa = sim_periodic_xaa * mask_atom[..., None]

    return (
        sim_periodic_xea,
        sim_periodic_xee,
        sim_periodic_xaa,
        sim_periodic_sea,
        sim_periodic_see,
        sim_periodic_saa,
    )


def construct_periodic_input_features(
    x: Array,
    simulation_cell: PyscfCell = None,
    ndim: int = 3,
    distance_type: str = "nu",
    sim_AV: Array | None = None,
    sim_BV: Array | None = None,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Constructs periodic generalized inputs to Fermi Net from raw electron and atomic positions.

    Args:
        x: Particle positions. Shape ((nelectrons+natom)*ndim,).
        simulation_cell: Simulation cell object containing scale and other properties.
        ndim: Dimension of system. Change only with caution.
        distance_type: Type of distance function to use ('nu' or 'tri').
        sim_AV: Optional lattice vectors (divided by 2pi). If None, uses simulation_cell.AV.
        sim_BV: Optional reciprocal vectors. If None, uses simulation_cell.BV.

    Returns:
        Tuple of (ae, ee, aa, r_ae, r_ee, r_aa).

    """
    distance_func = get_distance_function(distance_type)
    if sim_AV is None:
        sim_AV = simulation_cell.AV
    if sim_BV is None:
        sim_BV = simulation_cell.BV

    natom = simulation_cell.natm
    elec_coord = x[natom * 3 :].reshape(-1, ndim)
    atom_coord = x[: natom * 3].reshape(-1, ndim)

    sim_elec, _ = enforce_pbc(simulation_cell.a, elec_coord)
    sim_atom, _ = enforce_pbc(simulation_cell.a, atom_coord)

    return _compute_periodic_features_core(
        sim_elec, sim_atom, sim_AV, sim_BV, distance_func
    )


def construct_periodic_input_features_nvt_fixed(
    x: Array,
    simulation_cell: PyscfCell = None,
    ndim: int = 3,
    distance_type: str = "nu",
) -> tuple[Array, Array, Array, Array]:
    """Constructs periodic input features for the Fermi Net (NVT Fixed)."""
    distance_func = get_distance_function(distance_type)

    x = x.reshape(-1, ndim)
    sim_elec, _ = enforce_pbc(simulation_cell.a, x)

    atoms = jnp.asarray(simulation_cell.atom_coords(), dtype=simulation_cell.a.dtype)
    sim_atom, _ = enforce_pbc(simulation_cell.a, atoms)

    xea, xee, _, sea, see, _ = _compute_periodic_features_core(
        sim_elec,
        sim_atom,
        simulation_cell.AV,
        simulation_cell.BV,
        distance_func,
        compute_aa=False,
    )

    return xea, xee, sea, see


def construct_periodic_input_features_npt_quantum(
    x: Array,
    simulation_cell: PyscfCell = None,
    ndim: int = 3,
    distance_type: str = "nu",
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Constructs periodic generalized inputs to Fermi Net from raw electron and atomic positions (NPT Quantum)."""
    sim_AV = jnp.eye(3) / jnp.pi / 2
    sim_BV = jnp.eye(3) * jnp.pi * 2

    return construct_periodic_input_features(
        x, simulation_cell, ndim, distance_type, sim_AV=sim_AV, sim_BV=sim_BV
    )


def construct_periodic_input_features_fixed_npt_quantum(
    x: Array,
    simulation_cell: PyscfCell = None,
    ndim: int = 3,
    distance_type: str = "nu",
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Constructs periodic generalized inputs to Fermi Net from raw electron and atomic positions (Fixed Atoms, NPT)."""
    distance_func = get_distance_function(distance_type)
    sim_AV = jnp.eye(3) / jnp.pi / 2
    sim_BV = jnp.eye(3) * jnp.pi * 2

    natom = simulation_cell.natm
    # x contains something in first natom slots but we ignore it for atoms
    elec_coord = x[natom * 3 :].reshape(-1, ndim)

    sim_elec, _ = enforce_pbc(simulation_cell.a, elec_coord)
    atoms = jnp.asarray(simulation_cell.atom_coords(), dtype=simulation_cell.a.dtype)
    sim_atom, _ = enforce_pbc(simulation_cell.a, atoms)

    return _compute_periodic_features_core(
        sim_elec, sim_atom, sim_AV, sim_BV, distance_func, compute_aa=True
    )


def construct_periodic_2body_init_features(
    simulation_cell: PyscfCell = None,
    distance_type: str = "nu",
) -> tuple[Array]:
    """Constructs periodic generalized inputs to Fermi Net from raw electron and atomic positions.

    Args:
        simulation_cell: Simulation cell object containing scale and other properties.
        distance_type: Type of distance function to use ('nu' or 'tri').

    Returns:
        r_aa: Atom-atom distance. Shape (natom, natom, 1).

        The diagonal terms in r_aa are masked out such that the gradients of these
        terms are also zero.

    """
    distance_func = get_distance_function(distance_type)
    natom = simulation_cell.natm
    atoms = jnp.asarray(simulation_cell.atom_coords(), dtype=simulation_cell.a.dtype)
    sim_atom, _ = enforce_pbc(simulation_cell.a, atoms)
    sim_xaa = sim_atom[:, None, :] - sim_atom[None, :, :]
    sim_periodic_saa, _ = distance_func(
        sim_xaa + jnp.eye(natom, dtype=atoms.dtype)[..., None],
        simulation_cell.AV,
        simulation_cell.BV,
    )
    sim_periodic_saa = sim_periodic_saa * (1.0 - jnp.eye(natom, dtype=atoms.dtype))
    sim_periodic_saa = sim_periodic_saa[..., None]
    return sim_periodic_saa


def _get_input_dimensions(natom: int, distance_type: str) -> tuple[int, int, int]:
    """Determines input dimensions based on the number of atoms and distance type.

    Args:
        natom: Number of atoms.
        distance_type: Type of distance function ('nu' or 'tri').

    Returns:
        Tuple of (dim_ae, dim_ee, dim_aa):
          dim_ae: Dimension of atom-electron features.
          dim_ee: Dimension of electron-electron features.
          dim_aa: Dimension of atom-atom features.

    Raises:
        ValueError: If an unrecognized distance_type is provided.

    """
    if distance_type == "nu":
        return (natom * 4, 4, 4)
    elif distance_type == "tri":
        return (natom * 7, 7, 7)
    else:
        raise ValueError(f"Unrecognized distance function: {distance_type}")


def _init_atom_params(
    key: chex.PRNGKey,
    simulation_cell: PyscfCell,
    distance_type: str,
    hidden_dims: tuple[int, ...],
    rs: float | None = None,
    ensemble: str = "NVT",
):
    """Initializes atom parameters for the network.

    Args:
        key: JAX random key for initialization.
        simulation_cell: Simulation cell object containing scale and other properties.
        distance_type: Type of distance function to use.
        hidden_dims: List of hidden layer dimensions.
        rs: Wigner-Seitz radius.
        ensemble: Statistical ensemble type ("NVT" or "NPT").

    Returns:
        Tuple of (params, key):
          params: Dictionary containing initialized atom parameters.
          key: Updated JAX random key.

    """
    natom = simulation_cell.natm
    in_dims = _get_input_dimensions(natom, distance_type)
    dims_atom_double = [in_dims[2]] + [hdim[2] for hdim in hidden_dims]

    # Initialize base parameter structure
    params = {
        "atom_double": [{} for _ in range(len(hidden_dims))],
        "atom_envelope": {},
        "atom_distance": {},
        "alpha": jnp.zeros(1),
    }
    # Initialize atom_double layers
    params, key = _init_atom_double_layers(
        params, key, dims_atom_double, simulation_cell
    )

    params, key = _init_atom_distance_layer(
        params, key, dims_atom_double, simulation_cell
    )

    # Initialize atom_envelope based on new parameters
    params, key = _init_atom_envelope(
        params,
        key,
        simulation_cell,
        distance_type,
        natom,
        rs,
        ensemble,
    )

    return params, key


def _init_atom_double_layers(
    params: dict,
    key: chex.PRNGKey,
    dims_atom_double: Sequence[int],
    simulation_cell: PyscfCell,
):
    """Initializes the atom_double layers with weights and biases.

    Args:
        params: Parameter dictionary to update.
        key: JAX random key.
        dims_atom_double: Dimensions for the atom double layers.
        simulation_cell: Simulation cell object.

    Returns:
        Tuple of (params, key).

    """
    for i in range(len(dims_atom_double) - 1):
        # Initialize weights
        key, subkey = jax.random.split(key)
        params["atom_double"][i]["w"] = jax.random.normal(
            subkey, shape=(dims_atom_double[i], dims_atom_double[i + 1])
        ) / jnp.sqrt(float(dims_atom_double[i]))

        # Initialize biases
        key, subkey = jax.random.split(key)
        params["atom_double"][i]["b"] = (
            jax.random.uniform(subkey, shape=(dims_atom_double[i + 1],))
            / simulation_cell.scale
            / jnp.sqrt(float(dims_atom_double[i]))
        )

    return params, key


def _init_atom_distance_layer(
    params: dict,
    key: chex.PRNGKey,
    dims_atom_double: Sequence[int],
    simulation_cell: PyscfCell,
):
    """Initializes the atom_distance layer with weights and biases.

    Args:
        params: Parameter dictionary to update.
        key: JAX random key.
        dims_atom_double: Dimensions for the atom double layers.
        simulation_cell: Simulation cell object.

    Returns:
        Tuple of (params, key).

    """
    # Initialize weights for the non-hybrid case (output dimension 1)
    key, subkey = jax.random.split(key)
    params["atom_distance"]["w"] = (
        jax.random.normal(  # Renamed to avoid conflict if base 'w' is used elsewhere
            subkey, shape=(dims_atom_double[-1], 1)
        )
        / jnp.sqrt(float(dims_atom_double[-1]))
    )

    key, subkey = jax.random.split(key)
    params["atom_distance"]["b"] = (  # Renamed for clarity
        jax.random.uniform(subkey, shape=(1,)) / simulation_cell.scale / jnp.sqrt(1.0)
    )

    return params, key


def _init_atom_envelope(
    params: dict,
    key: chex.PRNGKey,
    simulation_cell: PyscfCell,
    distance_type: str,
    natom: int,
    rs: float,
    ensemble: str,
):
    """Initializes the atom_envelope parameters.

    Args:
        params: Parameter dictionary to update.
        key: JAX random key.
        simulation_cell: Simulation cell object.
        distance_type: Type of distance function.
        natom: Number of atoms.
        rs: Wigner-Seitz radius.
        ensemble: Ensemble type ('NVT' or 'NPT').

    Returns:
        Tuple of (params, key).

    """
    param_shape = (natom, natom, 1)
    # Initialize pi
    key, subkey_pi = jax.random.split(key)
    if ensemble == "NPT":
        params["atom_envelope"]["pi"] = (natom**0.3333) * rs / 1.414 * jnp.ones(
            param_shape
        ) + jax.random.normal(subkey_pi, shape=param_shape)
    elif ensemble == "NVT":
        params["atom_envelope"]["pi"] = jnp.ones(param_shape) + jax.random.normal(
            subkey_pi, shape=param_shape
        )
    else:
        raise ValueError(f"Unsupported ensemble: {ensemble}")
    params["atom_envelope"]["b"] = construct_periodic_2body_init_features(
        simulation_cell, distance_type
    )

    return params, key


def _init_elec_params(
    key: chex.PRNGKey,
    atoms: Array,
    spins: tuple[int, int],
    envelope_type: str,
    hidden_dims: tuple[tuple[int, int], ...],
    bias_orbitals: bool,
    use_last_layer: bool,
    full_det: bool,
    determinants: int,
    distance_type: str,
):
    """Initializes electron parameters for the network.

    Args:
        key: JAX random key.
        atoms: Atom positions.
        spins: Tuple of spin-up and spin-down electrons.
        envelope_type: Type of envelope function.
        hidden_dims: Dimensions of hidden layers.
        bias_orbitals: Whether to bias orbitals.
        use_last_layer: Whether to use the last layer.
        full_det: Whether to use full determinant.
        determinants: Number of determinants.
        distance_type: Type of distance function.

    Returns:
        Tuple of (params, key):
          params: Dictionary containing initialized electron parameters.
          key: Updated JAX random key.

    """
    natom = atoms.shape[0]
    in_dims = _get_input_dimensions(natom, distance_type)
    active_spin_channels = [spin for spin in spins if spin > 0]
    nchannels = len(active_spin_channels)
    # The input to layer L of the one-electron stream is from
    # construct_symmetric_features and shape (nelectrons, nfeatures), where  # noqa: ERA001
    # nfeatures is i) output from the previous one-electron layer; ii) the mean
    # for each spin channel from each layer; iii) the mean for each spin channel
    # from each two-electron layer. We don't create features for spin channels
    # which contain no electrons (i.e. spin-polarised systems).
    dims_one_in = [(nchannels + 1) * in_dims[0] + nchannels * in_dims[1]] + [
        (nchannels + 1) * hdim[0] + nchannels * hdim[1] for hdim in hidden_dims
    ]
    if not use_last_layer:
        dims_one_in[-1] = hidden_dims[-1][0]
    dims_one_out = [hdim[0] for hdim in hidden_dims]
    dims_two = [in_dims[1]] + [hdim[1] for hdim in hidden_dims]

    len_double = len(hidden_dims) if use_last_layer else len(hidden_dims) - 1
    params = {
        "single": [{} for _ in range(len(hidden_dims))],
        "double": [{} for _ in range(len_double)],
        "orbital": [],
        "envelope": [{} for _ in active_spin_channels],
    }
    for i, spin in enumerate(active_spin_channels):
        nparam = sum(spins) * determinants if full_det else spin * determinants
        params["envelope"][i]["pi"] = jnp.ones((natom, nparam))
        if envelope_type == "isotropic":
            params["envelope"][i]["sigma"] = jnp.ones((natom, nparam))
        elif envelope_type == "diagonal":
            params["envelope"][i]["sigma"] = jnp.ones((natom, 3, nparam))
        elif envelope_type == "full":
            params["envelope"][i]["sigma"] = jnp.tile(
                jnp.eye(6)[..., None, None], [1, 1, natom, nparam]
            )

    for i in range(len(hidden_dims)):
        key, subkey = jax.random.split(key)
        params["single"][i]["w"] = jax.random.normal(
            subkey, shape=(dims_one_in[i], dims_one_out[i])
        ) / jnp.sqrt(float(dims_one_in[i]))

        key, subkey = jax.random.split(key)
        params["single"][i]["b"] = jax.random.normal(subkey, shape=(dims_one_out[i],))

        if i < len_double:
            key, subkey = jax.random.split(key)
            params["double"][i]["w"] = jax.random.normal(
                subkey, shape=(dims_two[i], dims_two[i + 1])
            ) / jnp.sqrt(float(dims_two[i]))

            key, subkey = jax.random.split(key)
            params["double"][i]["b"] = jax.random.normal(
                subkey, shape=(dims_two[i + 1],)
            )

    for i, spin in enumerate(active_spin_channels):
        nparam = sum(spins) * determinants if full_det else spin * determinants
        key, subkey = jax.random.split(key)
        params["orbital"].append({})
        params["orbital"][i]["w"] = jax.random.normal(
            subkey, shape=(dims_one_in[-1], 2 * nparam)
        ) / jnp.sqrt(float(dims_one_in[-1]))
        if bias_orbitals:
            key, subkey = jax.random.split(key)
            params["orbital"][i]["b"] = jax.random.normal(subkey, shape=(2 * nparam,))
    return params, key


def _init_cell_params(simulation_cell: PyscfCell, mode="diag") -> Array:
    """Initializes cell parameters.

    Args:
        simulation_cell: Simulation cell object.
        mode: Lattice initialization mode ('diag', 'angle', 'partial_angle').

    Returns:
        Initialized cell parameters.

    """
    if mode == "diag":
        return jnp.diagonal(simulation_cell.lattice_vectors())
    elif mode == "angle":
        latvec = (
            simulation_cell.lattice_vectors()
        )  # assumming each row is a lattice vector
        a = jnp.linalg.norm(latvec[0])
        b = jnp.linalg.norm(latvec[1])
        c = jnp.linalg.norm(latvec[2])
        alpha = jnp.arccos(jnp.dot(latvec[1], latvec[2]) / (b * c))  # in rad
        beta = jnp.arccos(jnp.dot(latvec[0], latvec[2]) / (a * c))
        gamma = jnp.arccos(jnp.dot(latvec[0], latvec[1]) / (a * b))
        return jnp.array([a, b, c, alpha, beta, gamma])
    elif mode == "partial_angle":
        latvec = (
            simulation_cell.lattice_vectors()
        )  # assumming each row is a lattice vector
        a = jnp.linalg.norm(latvec[0])
        b = jnp.linalg.norm(latvec[1])
        c = jnp.linalg.norm(latvec[2])
        return jnp.array([a, b, c])
    else:
        raise ValueError(f"Invalid mode: {mode}")


def isotropic_envelope(ae: Array, params: dict) -> Array:
    """Computes an isotropic exponentially-decaying multiplicative envelope.

    Args:
        ae: Atom-electron vectors.
        params: Envelope parameters containing 'sigma' and 'pi'.

    Returns:
        The evaluated envelope.

    """
    return jnp.sum(jnp.exp(-jnp.abs(ae * params["sigma"])) * params["pi"], axis=1)


def diagonal_envelope(ae: Array, params: dict) -> Array:
    """Computes a diagonal exponentially-decaying multiplicative envelope.

    Args:
        ae: Atom-electron vectors.
        params: Envelope parameters containing 'sigma' and 'pi'.

    Returns:
        The evaluated envelope.

    """
    r_ae = jnp.linalg.norm(ae[..., None] * params["sigma"], axis=2)
    return jnp.sum(jnp.exp(-r_ae) * params["pi"], axis=1)


vdot = jax.vmap(jnp.dot, (0, 0))


def apply_covariance(x: Array, y: Array) -> Array:
    """Computes the product of covariance matrix y with vectors x.

    Equivalent to jnp.einsum('ijk,kmjn->ijmn', x, y).

    Args:
        x: Input vectors.
        y: Covariance matrices.

    Returns:
        Result of the multiplication.

    """
    i, _, _ = x.shape
    k, m, j, n = y.shape
    x = x.transpose((1, 0, 2))
    y = y.transpose((2, 0, 1, 3)).reshape((j, k, m * n))
    return vdot(x, y).reshape((j, i, m, n)).transpose((1, 0, 2, 3))


def full_envelope(ae: Array, params: dict) -> Array:
    """Computes a fully anisotropic exponentially-decaying multiplicative envelope.

    Args:
        ae: Atom-electron vectors.
        params: Envelope parameters containing 'sigma' and 'pi'.

    Returns:
        The evaluated envelope.

    """
    r_ae = apply_covariance(ae, params["sigma"])
    r_ae = curvature_tags_and_blocks.register_qmc(
        r_ae, ae, params["sigma"], type="full"
    )
    r_ae = jnp.linalg.norm(r_ae, axis=2)
    return jnp.sum(jnp.exp(-r_ae) * params["pi"], axis=1)


def output_envelope(ae: Array, params: dict) -> Array:
    """Computes a fully anisotropic envelope with a single output.

    Args:
        ae: Atom-electron vectors.
        params: Envelope parameters.

    Returns:
        The evaluated envelope (log scale).

    """
    sigma = jnp.expand_dims(params["sigma"], -1)
    ae_sigma = jnp.squeeze(apply_covariance(ae, sigma), axis=-1)
    r_ae = jnp.linalg.norm(ae_sigma, axis=2)
    return jnp.sum(jnp.log(jnp.sum(jnp.exp(-r_ae + params["pi"]), axis=1)))


def atom_envelop_2body_nonhybrid_full_dynamic(r_aa: Array, params: dict) -> Array:
    """Computes a non-hybrid full gaussian-decaying envelope function for dynamic centers.

    The inputs are expected to be non-hybrid.

    Args:
        r_aa: Distance matrix or features, shape (natom, natom, 1).
        params: Dictionary with 'pi' and 'b' parameters.
            params['pi']: shape (natom, natom, 1).
            params['b']: shape (natom, natom, 1) (dynamic centers from network).

    Returns:
        envelope: Log of envelope function value.

    """
    mat = -(params["pi"] ** 2) * (r_aa - params["b"]) ** 2
    return jnp.sum(mat)


def atom_envelop_2body_nonhybrid_full_fixed(
    r_aa: Array, params: dict, distance_aa: Array
) -> Array:
    """Computes a 2-body full fixed envelope function.

    Args:
        r_aa: Distance matrix or features, shape (natom, natom, 1).
        params: Dictionary with parameters
            params['pi']: shape (natom, natom, 1)
        distance_aa: Reference distances, shape (natom, natom, 1)

    Returns:
        envelope: Log of envelope function value

    """
    mat = -(params["pi"] ** 2) * (r_aa - distance_aa) ** 2
    return jnp.sum(mat)


def get_atom_wave_function(
    atom_center_dynamic: bool,
) -> Callable:
    """Returns the appropriate nuclear wavefunction envelope function based on configuration.

    Args:
        atom_center_dynamic: If True, Gaussian centers are learned by the network.
                             If False, they are fixed (e.g., to initial positions).

    Returns:
        Function implementing the specified nuclear wavefunction envelope.

    Raises:
        ValueError: If an invalid combination of options is provided.

    """
    if atom_center_dynamic:
        return atom_envelop_2body_nonhybrid_full_dynamic
    else:  # fixed cente
        return atom_envelop_2body_nonhybrid_full_fixed


def slogdet(x: Array) -> tuple[Array, Array]:
    """Computes sign and log of determinants of matrices.

    This is a jnp.linalg.slogdet with a special (fast) path for small matrices.

    Args:
      x: Square matrix.

    Returns:
      Tuple of (sign, logdet):
        sign: Sign of the determinant.
        logdet: Natural logarithm of the absolute value of the determinant.

    """
    if x.shape[-1] == 1:
        if x.dtype == jnp.complex64 or x.dtype == jnp.complex128:
            sign = x[..., 0, 0] / jnp.abs(x[..., 0, 0])
        else:
            sign = jnp.sign(x[..., 0, 0])
        logdet = jnp.log(jnp.abs(x[..., 0, 0]))
    else:
        sign, logdet = jnp.linalg.slogdet(x)

    return sign, logdet


def logdet_matmul(
    xs: Sequence[Array], w: Array | None = None
) -> tuple[Array, Array]:
    """Combines determinants and takes dot product with weights in log-domain.

    We use the log-sum-exp trick to reduce numerical instabilities.

    Args:
      xs: FermiNet orbitals in each determinant. Either of length 1 with shape
        (ndet, nelectron, nelectron) (full_det=True) or length 2 with shapes
        (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta) (full_det=False,
        determinants are factorised into block-diagonals for each spin channel).
      w: Weight of each determinant. If none, a uniform weight is assumed.

    Returns:
      Tuple of (phase_out, log_out):
        phase_out: Phase of the result.
        log_out: Logarithm of the absolute magnitude of the result.

    """
    # 1x1 determinants appear to be numerically sensitive and can become 0
    # (especially when multiple determinants are used with the spin-factored
    # wavefunction). Avoid this by not going into the log domain for 1x1 matrices.
    # Pass initial value to functools so det1d = 1 if all matrices are larger than
    # 1x1.
    det1d = functools.reduce(
        operator.mul, [x.reshape(-1) for x in xs if x.shape[-1] == 1], 1
    )
    # Pass initial value to functools so sign_in = 1, logdet = 0 if all matrices
    # are 1x1.
    phase_in, logdet = functools.reduce(
        lambda a, b: (a[0] * b[0], a[1] + b[1]),
        [slogdet(x) for x in xs if x.shape[-1] > 1],
        (1, 0),
    )

    # log-sum-exp trick
    maxlogdet = jnp.max(logdet)
    det = phase_in * det1d * jnp.exp(logdet - maxlogdet)
    result = jnp.sum(det) if w is None else jnp.matmul(det, w)[0]
    # return phase as a unit-norm complex number, rather than as an angle
    if result.dtype == jnp.complex64 or result.dtype == jnp.complex128:
        phase_out = jnp.angle(result)  # result / jnp.abs(result)
    else:
        phase_out = jnp.sign(result)
    log_out = jnp.log(jnp.abs(result)) + maxlogdet
    return phase_out, log_out


def linear_layer(x: Array, w: Array, b: Array | None = None) -> Array:
    """Evaluates a linear layer, x w + b.

    Args:
      x: Inputs.
      w: Weights.
      b: Optional bias. Only x w is computed if b is None.

    Returns:
      x w + b if b is given, x w otherwise.

    """
    y = jnp.dot(x, w)
    y = y + b if b is not None else y
    return y


vmap_linear_layer = jax.vmap(linear_layer, in_axes=(0, None, None), out_axes=0)


def residual(x: Array, y: Array) -> Array:
    """Computes a residual connection update.

    Args:
        x: Input tensor.
        y: Update tensor.

    Returns:
        (x + y) / sqrt(2) if shapes match, else y.

    """
    return (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y


def rezero(x: Array, y: Array, params: dict) -> Array:
    """Computes a rezero connection update.

    Args:
        x: Input tensor.
        y: Update tensor.
        params: Parameters containing 'alpha'.

    Returns:
        alpha * x + y.

    """
    return params["alpha"] * x + y


def elec_forward(
    h_one: Array, h_two: Array, params: dict, spins: tuple[int, int]
) -> Sequence[Array]:
    """Computes the forward pass for the electron stream of the FermiNet.

    Args:
        h_one: One-electron features.
        h_two: Two-electron features.
        params: Network parameters.
        spins: Tuple of spin-up and spin-down electrons.

    Returns:
        Output features split by spin channel.

    """
    for i in range(len(params["double"])):
        h_one_in = construct_symmetric_features(h_one, h_two, spins)
        # Execute next layer
        h_one_next = jnp.tanh(linear_layer(h_one_in, **params["single"][i]))
        h_two_next = jnp.tanh(
            vmap_linear_layer(h_two, params["double"][i]["w"], params["double"][i]["b"])
        )
        h_one = residual(h_one, h_one_next)
        h_two = residual(h_two, h_two_next)
    if len(params["double"]) != len(params["single"]):
        h_one_in = construct_symmetric_features(h_one, h_two, spins)
        h_one_next = jnp.tanh(linear_layer(h_one_in, **params["single"][-1]))
        h_one = residual(h_one, h_one_next)
        h_to_orbitals = h_one
    else:
        h_to_orbitals = construct_symmetric_features(h_one, h_two, spins)
    h_to_orbitals = jnp.split(h_to_orbitals, spins[0:1], axis=0)
    return h_to_orbitals


def get_init_distance_features(
    simulation_cell: PyscfCell = None,
    distance_type: str = "nu",
) -> tuple[Array]:
    """Get the initial distance features for the atom_envelope.

    Args:
        simulation_cell: Simulation cell object.
        distance_type: Distance type ('nu' or 'tri').

    Returns:
        Initial atom-atom distance features. Shape (natom, natom, 1).

    """
    initial_distance_features = construct_periodic_2body_init_features(
        simulation_cell=simulation_cell, distance_type=distance_type
    )  # Shape (natom, natom, 1)
    return initial_distance_features


def atom_forward(h_atom: Array, params: dict) -> Array:
    """Computes the forward pass for the atom stream.

    Args:
        h_atom: Atom features.
        params: Network parameters.

    Returns:
        Processed atom features.

    """
    for i in range(len(params["atom_double"])):
        h_atom_next = jnp.tanh(
            vmap_linear_layer(
                h_atom, params["atom_double"][i]["w"], params["atom_double"][i]["b"]
            )
        )
        h_atom = residual(h_atom, h_atom_next)
    return h_atom


def enforce_angle(x: Array) -> Array:
    """Enforces periodic boundary conditions on angles.

    Args:
        x: Input angles.

    Returns:
        Angles wrapped to [0, pi).

    """
    return jnp.mod(x, jnp.pi)


def get_jacobian(cellpar: Array, lattice_config: CrystalLatticeConfig) -> Array:
    """Creates the Jacobian matrix from fractional coordinates to Cartesian coordinates.

    Args:
        cellpar: Array containing lattice parameters.
            * lattice_config.mode == 'diag': [a, b, c].
            * lattice_config.mode == 'angle': [a, b, c, alpha, beta, gamma].
            * lattice_config.mode == 'partial_angle': [a, b, c].
        lattice_config: Configuration object for the crystal lattice.

    Returns:
        The Jacobian matrix.

    Raises:
        ValueError: If the lattice mode is not supported.

    """
    p_cell = cellpar.ravel()  # [a,b,c,alpha,beta,gamma] if mode == 'angle'
    if lattice_config.mode == "diag":
        trans_matrix = jnp.diag(p_cell)
    elif lattice_config.mode == "angle":
        a, b, c = p_cell[:3]  # lengths
        angles = enforce_angle(p_cell[3:])  # angles in radians
        # Cosine and sine terms
        cos_alpha, cos_beta, cos_gamma = jnp.cos(angles)
        _, sin_beta, sin_gamma = jnp.sin(angles)
        cosas = (cos_beta * cos_gamma - cos_alpha) / (sin_beta * sin_gamma)
        sinas = jnp.sqrt(1 - cosas**2)
        trans_matrix = jnp.array(
            [
                [a, 0, 0],
                [b * cos_gamma, b * sin_gamma, 0],
                [c * cos_beta, -c * sin_beta * cosas, c * sin_beta * sinas],
            ]
        )
    elif lattice_config.mode == "partial_angle":
        a, b, c = p_cell  # lengths
        angles = jnp.asarray(
            [lattice_config.alpha, lattice_config.beta, lattice_config.gamma]
        )
        # Cosine and sine terms
        cos_alpha, cos_beta, cos_gamma = jnp.cos(angles)
        _, sin_beta, sin_gamma = jnp.sin(angles)
        cosas = (cos_beta * cos_gamma - cos_alpha) / (sin_beta * sin_gamma)
        sinas = jnp.sqrt(1 - cosas**2)
        trans_matrix = jnp.array(
            [
                [a, 0, 0],
                [b * cos_gamma, b * sin_gamma, 0],
                [c * cos_beta, -c * sin_beta * cosas, c * sin_beta * sinas],
            ]
        )
    else:
        raise ValueError(f"mode {lattice_config.mode} is not supported.")
    return trans_matrix.astype(p_cell.dtype)


def get_inv_jacobian(cellpar: Array, lattice_config: CrystalLatticeConfig) -> Array:
    """Creates the inverse Jacobian matrix from Cartesian coordinates to fractional coordinates.

    Args:
        cellpar: Array containing lattice parameters.
            * lattice_config.mode == 'diag': [a, b, c].
            * lattice_config.mode == 'angle': [a, b, c, alpha, beta, gamma].
            * lattice_config.mode == 'partial_angle': [a, b, c].
        lattice_config: Configuration object for the crystal lattice.

    Returns:
        The inverse Jacobian matrix.

    Raises:
        ValueError: If the lattice mode is not supported.

    """
    p_cell = cellpar.ravel()
    if lattice_config.mode == "angle":
        inv_a, inv_b, inv_c = 1.0 / p_cell[:3]  # lengths
        angles = enforce_angle(p_cell[3:])  # angles in radians
        # Cosine and sine terms
        cos_alpha, cos_beta, cos_gamma = jnp.cos(angles)
        sin_angles = jnp.sin(angles[1:])
        csc_beta, csc_gamma = 1.0 / sin_angles
        cosas = (cos_beta * cos_gamma - cos_alpha) * csc_gamma * csc_beta
        csc_as = 1.0 / jnp.sqrt(1 - cosas**2)
        cot_gamma = cos_gamma * csc_gamma
        cot_beta = cos_beta * csc_beta
        cot_as = cosas * csc_as
        trans_matrix = jnp.array(
            [
                [inv_a, 0, 0],
                [-cot_gamma * inv_a, csc_gamma * inv_b, 0],
                [
                    -(cot_beta + cosas * cot_gamma) * csc_as * inv_a,
                    cot_as * csc_gamma * inv_b,
                    csc_beta * csc_as * inv_c,
                ],
            ]
        )
    elif lattice_config.mode == "diag":
        trans_matrix = jnp.diag(1.0 / p_cell)
    elif lattice_config.mode == "partial_angle":
        inv_a, inv_b, inv_c = 1.0 / p_cell  # lengths
        angles = jnp.asarray(
            [lattice_config.alpha, lattice_config.beta, lattice_config.gamma]
        )
        # Cosine and sine terms
        cos_alpha, cos_beta, cos_gamma = jnp.cos(angles)
        sin_angles = jnp.sin(angles[1:])
        csc_beta, csc_gamma = 1.0 / sin_angles
        cosas = (cos_beta * cos_gamma - cos_alpha) * csc_gamma * csc_beta
        csc_as = 1.0 / jnp.sqrt(1 - cosas**2)
        cot_gamma = cos_gamma * csc_gamma
        cot_beta = cos_beta * csc_beta
        cot_as = cosas * csc_as
        trans_matrix = jnp.array(
            [
                [inv_a, 0, 0],
                [-cot_gamma * inv_a, csc_gamma * inv_b, 0],
                [
                    -(cot_beta + cosas * cot_gamma) * csc_as * inv_a,
                    cot_as * csc_gamma * inv_b,
                    csc_beta * csc_as * inv_c,
                ],
            ]
        )
    else:
        raise ValueError(f"mode {lattice_config.mode} is not supported.")
    return trans_matrix.astype(p_cell.dtype)


def convert_to_simulation_cell(
    cellpar: Array, x: Array, lattice_config: CrystalLatticeConfig
) -> tuple[Array, Array]:
    """Converts the input x (fractional 0-1) to the simulation cell Cartesian coordinates.

    Args:
        cellpar: Parameters of the simulation cell.
        x: Input x in fractional coordinates [0, 1]. Shape ((nelectrons+natom)*3,).
        lattice_config: Configuration object for the crystal lattice.

    Returns:
        Tuple of (converted_x, jacobian):
          converted_x: Cartesian coordinates. Shape ((nelectrons+natom)*3,).
          jacobian: The Jacobian matrix used for conversion.

    """
    x = jnp.reshape(x, (-1, 3))  # (nelectrons+natom,3)
    jacobian = get_jacobian(cellpar, lattice_config)
    return jnp.dot(x, jacobian).ravel(), jacobian


def eval_phase(
    x: Array,
    klist: Sequence[Array],
    ndim: int = 3,
    spins: tuple[int, int] | None = None,
    full_det: bool = False,
) -> Sequence[Array]:
    """Evaluates the phase factor exp(i k.r) for the wavefunction.

    Args:
        x: Particle positions.
        klist: List of k-vectors.
        ndim: Dimension of the system.
        spins: Tuple of spin-up and spin-down electrons.
        full_det: Whether to use full determinant.

    Returns:
        List of phase factors for each spin channel.

    """
    x = jnp.reshape(x, [-1, ndim])
    xs = jnp.split(x, spins[0:1], axis=-2)
    if full_det:
        klist = jnp.concatenate(klist, axis=0)
        kdot_xs = [jnp.matmul(x, klist.T) for x, ne in zip(xs, spins) if ne > 0]
    else:
        kdot_xs = [
            jnp.matmul(x, kpt.T) for x, kpt, ne in zip(xs, klist, spins) if ne > 0
        ]
    phases = [jnp.exp(1j * kdot_x) for kdot_x in kdot_xs]
    return phases
