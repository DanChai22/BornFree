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

"""Implementation of Fermionic Neural Network for quantum systems in JAX."""

import functools
from collections import namedtuple

import chex
import jax.numpy as jnp
from jax import Array
from pyscf.pbc.gto import Cell as PyscfCell

from BornFree.network import network_block


def init_solid_fermi_net_params(
    key: chex.PRNGKey,
    data: Array | None,
    atoms: Array,
    spins: tuple[int, int],
    envelope_type: str = "full",
    simulation_cell: PyscfCell | None = None,
    bias_orbitals: bool = False,
    use_last_layer: bool = False,
    full_det: bool = True,
    hidden_dims: network_block.FermiLayers = (
        (256, 32, 32),
        (256, 32, 32),
        (256, 32, 1),
    ),
    determinants: int = 16,
    distance_type: str = "nu",
):
    """Initializes combined parameters for the full quantum system network (NVT Quantum).

    Args:
      key: JAX RNG state.
      data: Optional data (unused).
      atoms: (natom, 3) array of atom positions.
      spins: Tuple of the number of spin-up and spin-down electrons.
      envelope_type: Envelope type for electronic orbitals.
      simulation_cell: Simulation cell for periodic boundary conditions.
      bias_orbitals: If true, include bias in final orbital-shaping layer.
      use_last_layer: If true, combine one- and two-electron streams for orbitals.
      full_det: If true, use dense determinants across all electrons.
      hidden_dims: Network architecture dimensions.
      determinants: Number of determinants for electronic part.
      distance_type: Type of distance function ('nu', etc).

    Returns:
      PyTree of combined network parameters.

    """
    del data
    rs = network_block.eval_rs(simulation_cell)
    elec_params, key = network_block._init_elec_params(
        key,
        atoms,
        spins,
        envelope_type,
        hidden_dims,
        bias_orbitals,
        use_last_layer,
        full_det,
        determinants,
        distance_type,
    )

    # Initialize atom parameters
    atom_params, key = network_block._init_atom_params(
        key,
        simulation_cell,
        distance_type,
        hidden_dims,
        rs,
        ensemble="NVT",
    )

    # Combine parameters
    params = {**elec_params, **atom_params}
    return params


def get_atom_input_and_target_distance(
    x: Array,
    simulation_cell: PyscfCell | None = None,
    distance_type: str = "nu",
):
    """Computes atom input features and target residual distances.

    Args:
        x: Particle positions.
        simulation_cell: Simulation cell object.
        distance_type: Distance function type.

    Returns:
        Tuple of (h_atom_input, target_residual_distance).

    """
    _, _, aa_, _, _, r_aa = network_block.construct_periodic_input_features(
        x, simulation_cell=simulation_cell, distance_type=distance_type
    )
    h_atom_input = jnp.concatenate((r_aa, aa_), axis=2)  # Shape (natom, natom, 7)
    target_residual_distance = r_aa  # Shape (natom, natom, 1)
    return h_atom_input, target_residual_distance


def solid_fermi_net_atom_features(
    params,
    x,
    simulation_cell: PyscfCell | None = None,
    atom_center_dynamic: bool = True,
    distance_type: str = "nu",
    is_rezero: bool = False,
):
    """Generic forward evaluation for atomic/nuclear features and envelope calculation.

    Args:
        params: Network parameters.
        x: Particle positions.
        simulation_cell: Simulation cell object.
        atom_center_dynamic: Whether atom centers are dynamic.
        distance_type: Distance function type.
        is_rezero: Whether to use rezero connection.

    Returns:
        Log of the nuclear wavefunction envelope.

    """
    h_atom_input, target_residual_distance = get_atom_input_and_target_distance(
        x, simulation_cell=simulation_cell, distance_type=distance_type
    )
    h_atom_processed = network_block.atom_forward(h_atom_input, params)
    predicted_dist = network_block.vmap_linear_layer(
        h_atom_processed, params["atom_distance"]["w"], params["atom_distance"]["b"]
    )  # Shape (natom, 1) or (natom, natom, 1)
    assert predicted_dist.shape == target_residual_distance.shape

    if is_rezero:
        final_distances_for_envelope = network_block.rezero(
            predicted_dist, target_residual_distance, params
        )
    else:
        final_distances_for_envelope = network_block.residual(
            predicted_dist, target_residual_distance
        )

    envelope_func = network_block.get_atom_wave_function(
        atom_center_dynamic=atom_center_dynamic,
    )

    args_for_envelope = [final_distances_for_envelope, params["atom_envelope"]]
    if not atom_center_dynamic:  # Only fixed types require the 3rd argument
        initial_distance_features = network_block.get_init_distance_features(
            simulation_cell=simulation_cell,
            distance_type=distance_type,
        )
        args_for_envelope.append(initial_distance_features)

    log_gaussian = envelope_func(*args_for_envelope)
    return log_gaussian


def solid_fermi_net_electron_orbitals(
    params,
    x,
    simulation_cell: PyscfCell | None = None,
    klist: tuple[Array, Array] | None = None,
    spins: tuple[int, int] | None = None,
    envelope_type: str | None = None,
    full_det: bool = False,
    distance_type: str = "nu",
):
    """Forward evaluation of the Solid Neural Network up to the orbitals.

    Args:
      params: A dictionary of parameters.
      x: The input data, a 3N dimensional vector.
      simulation_cell: PySCF object of simulation cell.
      klist: Tuple with occupied k points of the spin up and spin down electrons.
      spins: Tuple with number of spin up and spin down electrons.
      envelope_type: A string that specifies kind of envelope ('isotropic', 'diagonal', 'full').
      full_det: If true, the determinants are dense, rather than block-sparse.
      distance_type: Type of distance function to use.

    Returns:
      Tuple of (orbitals, to_env):
        orbitals: Orbital matrices.
        to_env: Input variables for the envelope function.

    """
    # ae_: shape (nelec, natom, 3); r_ae: shape (nelec, natom, 1)
    # ee_: shape (nelec, nelec, 3); r_ee: shape (nelec, nelec, 1)
    # aa_: shape (natom, natom, 3); r_aa: shape (natom, natom, 1)
    ae_, ee_, _, r_ae, r_ee, _ = network_block.construct_periodic_input_features(
        x, simulation_cell=simulation_cell, distance_type=distance_type
    )
    natom = simulation_cell.natm
    elec_coord = x[natom * 3 :]
    ae = jnp.concatenate((r_ae, ae_), axis=2)
    ae = jnp.reshape(ae, [jnp.shape(ae)[0], -1])
    ee = jnp.concatenate((r_ee, ee_), axis=2)
    # which variable do we pass to envelope?
    to_env = r_ae if envelope_type == "isotropic" else ae_

    if envelope_type == "isotropic":
        envelope = network_block.isotropic_envelope
    elif envelope_type == "diagonal":
        envelope = network_block.diagonal_envelope
    elif envelope_type == "full":
        envelope = network_block.full_envelope

    h_one = ae  # single-electron features
    h_two = ee  # two-electron features
    h_to_orbitals = network_block.elec_forward(h_one, h_two, params, spins)

    active_spin_channels = [spin for spin in spins if spin > 0]
    orbitals = [
        network_block.linear_layer(h, **p)
        for h, p in zip(h_to_orbitals, params["orbital"])
    ]

    for i, _ in enumerate(active_spin_channels):
        nparams = params["orbital"][i]["w"].shape[-1] // 2
        orbitals[i] = orbitals[i][..., :nparams] + 1j * orbitals[i][..., nparams:]
    if envelope_type in ["isotropic", "diagonal", "full"]:
        orbitals = [
            envelope(te, param) * orbital
            for te, orbital, param in zip(
                jnp.split(to_env, active_spin_channels[:-1], axis=0),
                orbitals,
                params["envelope"],
            )
        ]
    # Reshape into matrices and drop unoccupied spin channels.
    orbitals = [
        jnp.reshape(orbital, [spin, -1, sum(spins) if full_det else spin])
        for spin, orbital in zip(active_spin_channels, orbitals)
        if spin > 0
    ]
    orbitals = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals]
    phases = network_block.eval_phase(
        elec_coord, klist=klist, ndim=3, spins=spins, full_det=full_det
    )

    orbitals = [orb * p[None, :, :] for orb, p in zip(orbitals, phases)]
    if full_det:
        orbitals = [jnp.concatenate(orbitals, axis=1)]
    return orbitals, to_env


def eval_func(
    params,
    x,
    klist: tuple[Array, Array] | None = None,
    simulation_cell: PyscfCell | None = None,
    spins: tuple[int, int] | None = None,
    envelope_type: str = "full",
    atom_center_dynamic: bool = True,
    is_rezero: bool = False,
    full_det: bool = False,
    distance_type: str = "nu",
    method_name: str = "eval_logabs_network",
    mcmc: str | None = None,
):
    """Evaluates the combined quantum wavefunction (NVT Quantum).

    Args:
        params: Parameter dictionary for network.
        x: Particle coordinates (combined nuclear and electronic), 3N dimensional vector.
        klist: K-points for periodic boundary conditions.
        simulation_cell: Simulation cell object.
        spins: Tuple of spin-up and spin-down electron counts.
        envelope_type: Electronic orbital envelope type.
        atom_center_dynamic: Whether atom centers are dynamic.
        is_rezero: Whether to use rezero.
        full_det: If true, use dense determinants across all electrons.
        distance_type: Type of distance function ('nu', etc.).
        method_name: Output format specification.
        mcmc: MCMC mode specification.

    Returns:
        Wavefunction evaluation according to method_name and mcmc parameters.

    """
    orbitals, _ = solid_fermi_net_electron_orbitals(
        params,
        x,
        klist=klist,
        simulation_cell=simulation_cell,
        spins=spins,
        envelope_type=envelope_type,
        full_det=full_det,
        distance_type=distance_type,
    )

    # Get the appropriate nuclear wavefunction implementation
    log_gaussian_atom = solid_fermi_net_atom_features(
        params,
        x,
        simulation_cell=simulation_cell,
        atom_center_dynamic=atom_center_dynamic,
        distance_type=distance_type,
        is_rezero=is_rezero,
    )

    # Process electronic wavefunction according to method_name
    if method_name == "eval_logabs_network":
        _, wf_elec = network_block.logdet_matmul(orbitals)
    elif method_name == "eval_log_network":
        phase, slogdet = network_block.logdet_matmul(orbitals)
        wf_elec = slogdet + 1j * phase
    elif method_name == "eval_phase_and_logabs_network":
        wf_elec = network_block.logdet_matmul(orbitals)
    elif method_name == "eval_orbitals":
        wf_elec = orbitals
    else:
        raise ValueError(f"Unrecognized method name: {method_name}")

    # Return results based on mcmc parameter
    if mcmc == "electron":
        result = wf_elec
    elif mcmc == "atom":
        result = log_gaussian_atom
    elif mcmc is None:
        result = wf_elec + log_gaussian_atom  # Combined wavefunction
    else:
        raise ValueError(f"Unrecognized mcmc mode: {mcmc}")

    return result


def make_solid_fermi_net(
    envelope_type: str = "full",
    atom_center_dynamic: bool = True,
    is_rezero: bool = False,
    bias_orbitals: bool = False,
    use_last_layer: bool = False,
    klist=None,
    simulation_cell: PyscfCell = None,
    full_det: bool = True,
    hidden_dims: network_block.FermiLayers = ((256, 32), (256, 32), (256, 32)),
    determinants: int = 16,
    distance_type: str = "nu",
    method_name: str = "eval_log_network",
    mcmc: str | None = None,
):
    """Creates a neural network quantum wavefunction model (NVT Quantum).

    Args:
        envelope_type: Electronic orbital envelope type.
        atom_center_dynamic: Whether atom centers are dynamic.
        is_rezero: Whether to use rezero.
        bias_orbitals: Whether to use bias in orbital layer.
        use_last_layer: Whether to use two-electron stream in final layer.
        klist: K-points for periodic boundary conditions.
        simulation_cell: Simulation cell object.
        full_det: If true, use dense determinants across all electrons.
        hidden_dims: Network architecture dimensions.
        determinants: Number of determinants for electronic part.
        distance_type: Type of distance function ('nu', etc.).
        method_name: Output format specification.
        mcmc: MCMC mode specification.

    Returns:
        A namedtuple with 'init' and 'apply' methods:
            - init: Function to initialize network parameters
            - apply: Function to evaluate the wavefunction

    """
    valid_methods = [
        "eval_logabs_network",
        "eval_log_network",
        "eval_orbitals",
        "eval_phase_and_logabs_network",
    ]
    if method_name not in valid_methods:
        raise ValueError(f"Method name must be one of: {valid_methods}")
    # Create named tuple with init and apply methods
    method = namedtuple("method", ["init", "apply"])

    # Initialize parameters function
    init = functools.partial(
        init_solid_fermi_net_params,
        atoms=jnp.asarray(simulation_cell.atom_coords(), dtype=simulation_cell.a.dtype),
        spins=simulation_cell.nelec,
        envelope_type=envelope_type,
        simulation_cell=simulation_cell,
        bias_orbitals=bias_orbitals,
        use_last_layer=use_last_layer,
        full_det=full_det,
        hidden_dims=hidden_dims,
        determinants=determinants,
        distance_type=distance_type,
    )

    # Apply function to evaluate wavefunction
    network = functools.partial(
        eval_func,
        simulation_cell=simulation_cell,
        klist=klist,
        spins=simulation_cell.nelec,
        envelope_type=envelope_type,
        atom_center_dynamic=atom_center_dynamic,
        is_rezero=is_rezero,
        full_det=full_det,
        distance_type=distance_type,
        method_name=method_name,
        mcmc=mcmc,
    )

    method.init = init
    method.apply = network
    return method
