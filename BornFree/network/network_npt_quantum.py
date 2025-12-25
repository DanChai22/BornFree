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
import functools
from collections import namedtuple

import chex
import jax.numpy as jnp
from jax import Array
from pyscf.pbc.gto import Cell as PyscfCell

from BornFree.network import network_block


def init_solid_fermi_net_params(
    key: chex.PRNGKey,
    data,
    atoms: Array,
    spins: tuple[int, int],
    envelope_type: str = "full",
    simulation_cell: PyscfCell = None,
    unit_cell: PyscfCell = None,
    bias_orbitals: bool = False,
    use_last_layer: bool = False,
    full_det: bool = True,
    hidden_dims: network_block.FermiLayers = (
        (256, 32, 32),
        (256, 32, 32),
        (256, 32, 1),
    ),
    determinants: int = 16,
    lattice_mode: str = "angle",
    distance_type="nu",
) -> dict:
    """Initializes parameters for the NPT ensemble.

    This function creates and initializes all parameters needed for the NPT ensemble,
    including weights and biases for the neural network layers, envelope functions,
    cell parameters.

    Args:
        key: JAX RNG state.
        data: Optional data (unused).
        atoms: (natom, 3) array of atom positions.
        spins: Tuple of the number of spin-up and spin-down electrons.
        envelope_type: Envelope to use to impose orbitals go to zero at infinity.
        simulation_cell: PySCF simulation cell object.
        unit_cell: PySCF unit cell object.
        bias_orbitals: If true, include a bias in the final linear layer to shape the outputs into orbitals.
        use_last_layer: If true, the outputs of the one- and two-electron streams are combined.
        full_det: If true, evaluate determinants over all electrons.
        hidden_dims: Tuple of pairs, where each pair contains the number of hidden units.
        determinants: Number of determinants to use.
        lattice_mode: Mode for lattice parameters initialization.
        distance_type: Type of distance function to use.

    Returns:
        Dictionary containing initialized parameters for:
        - Neural network weights and biases
        - Envelope functions
        - Cell parameters
        - Atomic parameters

    """
    del data
    rs = network_block.eval_rs(simulation_cell)
    cell_params = {}
    cell_params["cell"] = network_block._init_cell_params(simulation_cell, lattice_mode)
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
    atom_params, key = network_block._init_atom_params(
        key, unit_cell, distance_type, hidden_dims, rs, ensemble="NPT"
    )
    params = {**elec_params, **atom_params, **cell_params}
    return params


def get_atom_input_and_target_distance(
    x: Array,
    simulation_cell: PyscfCell = None,
    distance_type: str = "nu",
) -> tuple[Array, Array]:
    """Computes atom input features and target residual distances.

    Args:
        x: Particle positions.
        simulation_cell: Simulation cell object.
        distance_type: Distance function type.

    Returns:
        Tuple of (h_atom_input, target_residual_distance).

    """
    _, _, aa_, _, _, r_aa = network_block.construct_periodic_input_features_npt_quantum(
        x, simulation_cell=simulation_cell, distance_type=distance_type
    )
    h_atom_input = jnp.concatenate((r_aa, aa_), axis=2)  # Shape (natom, natom, 7)
    target_residual_distance = r_aa  # Shape (natom, natom, 1)
    return h_atom_input, target_residual_distance


def solid_fermi_net_atom_features(
    params,
    x: Array,
    simulation_cell: PyscfCell = None,
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
        x,
        simulation_cell=simulation_cell,
        distance_type=distance_type,
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
    simulation_cell: PyscfCell = None,
    klist=None,
    atom_center_dynamic: bool = True,
    spins: tuple[int, int] = (None, None),
    envelope_type: str | None = None,
    full_det=False,
    distance_type="nu",
):
    """Forward evaluation of the Solid Neural Network up to the orbitals.

    Args:
      params: A dictionary of parameters.
      x: The input data, a 3N dimensional vector.
      simulation_cell: PySCF object of simulation cell.
      klist: Tuple with occupied k points of the spin up and spin down electrons.
      atom_center_dynamic: Whether atom centers are dynamic.
      spins: Tuple with number of spin up and spin down electrons.
      envelope_type: A string that specifies kind of envelope ('isotropic', 'diagonal', 'full').
      full_det: If true, the determinants are dense, rather than block-sparse.
      distance_type: Type of distance function to use.

    Returns:
      Tuple of (orbitals, to_env):
        orbitals: Orbital matrices.
        to_env: Input variables for the envelope function.

    """
    if atom_center_dynamic:
        ae_, ee_, _, r_ae, r_ee, _ = (
            network_block.construct_periodic_input_features_npt_quantum(
                x, simulation_cell=simulation_cell, distance_type=distance_type
            )
        )
    else:
        ae_, ee_, _, r_ae, r_ee, _ = (
            network_block.construct_periodic_input_features_fixed_npt_quantum(
                x, simulation_cell=simulation_cell, distance_type=distance_type
            )
        )

    natom = simulation_cell.natm
    elec_coord = x[natom * 3 :]
    ae = jnp.concatenate((r_ae, ae_), axis=2)
    ae = jnp.reshape(ae, [jnp.shape(ae)[0], -1])
    ee = jnp.concatenate((r_ee, ee_), axis=2)
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

    for i in range(len(active_spin_channels)):
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
    klist=None,
    simulation_cell=None,
    spins=(None, None),
    envelope_type="full",
    atom_center_dynamic=True,
    is_rezero=False,
    full_det=False,
    distance_type="nu",
    method_name="eval_logabs_network",
    mcmc=None,
):
    """Generates the wavefunction of the simulation cell (NPT Ensemble).

    Args:
        params: Parameter dict.
        x: The input data, a 3N dimensional vector.
        klist: Tuple with occupied k points of the spin up and spin down electrons.
        simulation_cell: PySCF object of simulation cell.
        spins: Tuple with number of spin up and spin down electrons.
        envelope_type: Envelope type string.
        atom_center_dynamic: Whether atom centers are dynamic.
        is_rezero: Whether to use rezero.
        full_det: Specify the mode of wavefunction, spin diagonalized or not.
        distance_type: Distance function type.
        method_name: Specify the returned function of wavefunction.
        mcmc: MCMC mode ('electron', 'atom', or None for both).

    Returns:
        Required wavefunction value.

    """
    orbitals, _ = solid_fermi_net_electron_orbitals(
        params,
        x,
        klist=klist,
        simulation_cell=simulation_cell,
        atom_center_dynamic=atom_center_dynamic,
        spins=spins,
        envelope_type=envelope_type,
        full_det=full_det,
        distance_type=distance_type,
    )
    log_wf_atom = solid_fermi_net_atom_features(
        params,
        x,
        simulation_cell=simulation_cell,
        atom_center_dynamic=atom_center_dynamic,
        distance_type=distance_type,
        is_rezero=is_rezero,
    )
    if method_name == "eval_logabs_network":
        _, log_wf_elec = network_block.logdet_matmul(orbitals)
    elif method_name == "eval_log_network":
        phase, slogdet = network_block.logdet_matmul(orbitals)
        log_wf_elec = slogdet + 1j * phase
    elif method_name == "eval_phase_and_logabs_network":
        log_wf_elec = network_block.logdet_matmul(orbitals)
    elif method_name == "eval_orbitals":
        log_wf_elec = orbitals
    else:
        raise ValueError("Unrecognized method name")

    if mcmc == "electron":
        result = log_wf_elec
    elif mcmc == "atom":
        result = log_wf_atom
    elif mcmc is None:
        result = log_wf_elec + log_wf_atom

    return result


def make_solid_fermi_net(
    envelope_type: str = "full",
    atom_center_dynamic: bool = True,
    is_rezero: bool = False,
    bias_orbitals: bool = False,
    use_last_layer: bool = False,
    klist=None,
    simulation_cell: PyscfCell = None,
    unit_cell: PyscfCell = None,
    full_det: bool = True,
    hidden_dims: network_block.FermiLayers = ((256, 32), (256, 32), (256, 32)),
    lattice_mode: str = "angle",
    determinants: int = 16,
    distance_type: str = "nu",
    method_name: str = "eval_log_network",
    mcmc: str | None = None,
):
    """Creates a Fermionic Neural Network for NPT ensemble.

    Args:
        envelope_type: Specify envelope type.
        atom_center_dynamic: Whether atom centers are dynamic.
        is_rezero: Whether to use rezero.
        bias_orbitals: Whether to contain bias in the last layer of orbitals.
        use_last_layer: Whether to use two-electron feature in the last layer.
        klist: Occupied k points from HF.
        simulation_cell: Simulation cell.
        unit_cell: Unit cell.
        full_det: Specify the mode of wavefunction, spin diagonalized or not.
        hidden_dims: Specify the dimension of one-electron and two-electron layer.
        lattice_mode: Lattice initialization mode.
        determinants: The number of determinants used.
        distance_type: Distance function type.
        method_name: Specify the returned function.
        mcmc: MCMC mode.

    Returns:
        A namedtuple with 'init' and 'apply' methods.

    """
    if method_name not in [
        "eval_logabs_network",
        "eval_log_network",
        "eval_orbitals",
        "eval_phase_and_logabs_network",
    ]:
        raise ValueError("Method name is not in class dir.")

    method = namedtuple("method", ["init", "apply"])
    init = functools.partial(
        init_solid_fermi_net_params,
        atoms=unit_cell.atom_coords(),
        spins=unit_cell.nelec,
        envelope_type=envelope_type,
        simulation_cell=simulation_cell,
        unit_cell=unit_cell,
        bias_orbitals=bias_orbitals,
        use_last_layer=use_last_layer,
        full_det=full_det,
        hidden_dims=hidden_dims,
        determinants=determinants,
        lattice_mode=lattice_mode,
        distance_type=distance_type,
    )

    network = functools.partial(
        eval_func,
        simulation_cell=unit_cell,
        klist=klist,
        spins=unit_cell.nelec,
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
