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
    bias_orbitals: bool = False,
    use_last_layer: bool = False,
    full_det: bool = True,
    hidden_dims: network_block.FermiLayers = ((256, 32), (256, 32), (256, 32)),
    determinants: int = 16,
    distance_type="nu",
):
    """Initializes parameters for the Fermionic Neural Network (NVT Fixed).

    This function creates and initializes all parameters needed for the FermiNet,
    including weights and biases for the one-electron and two-electron streams,
    orbital shaping parameters, and envelope function parameters.

    Args:
        key: JAX RNG state for reproducible initialization.
        data: Additional data for initialization (currently unused).
        atoms: Array of shape (natom, 3) containing atom positions.
        spins: Tuple of (n_up, n_down) specifying number of spin-up and spin-down electrons.
        envelope_type: Type of envelope function to use ('full', 'isotropic', or None).
        bias_orbitals: Whether to include bias in the final orbital-shaping layer.
        use_last_layer: Whether to combine one- and two-electron streams in final layer.
        full_det: If True, evaluate determinants over all electrons; if False, use spin blocks.
        hidden_dims: Tuple of (one_electron_dim, two_electron_dim) pairs for each layer.
        determinants: Number of determinants in the wavefunction.
        distance_type: Type of distance function to use ('nu' for minimum image convention).

    Returns:
        PyTree containing all network parameters organized by component.

    """
    del data
    params, _ = network_block._init_elec_params(
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
    return params


def solid_fermi_net_orbitals(
    params,
    x,
    simulation_cell: PyscfCell = None,
    klist=None,
    spins=(None, None),
    envelope_type=None,
    full_det=False,
    distance_type="nu",
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
    ae_, ee_, r_ae, r_ee = network_block.construct_periodic_input_features_nvt_fixed(
        x, simulation_cell=simulation_cell, distance_type=distance_type
    )
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

    active_spin_channels = [spin for spin in spins if spin > 0]  # shape (nalpha, nbeta)
    orbitals = [
        network_block.linear_layer(h, **p)
        for h, p in zip(h_to_orbitals, params["orbital"])
    ]

    for i, spin in enumerate(active_spin_channels):
        nparams = params["orbital"][i]["w"].shape[-1] // 2
        orbitals[i] = (
            orbitals[i][..., :nparams] + 1j * orbitals[i][..., nparams:]
        )  # shape (nalpha, nparams)

    if envelope_type in ["isotropic", "diagonal", "full"]:
        orbitals = [
            envelope(te, param) * orbital
            for te, orbital, param in zip(
                jnp.split(to_env, active_spin_channels[:-1], axis=0),
                orbitals,
                params["envelope"],
            )
        ]  # shape (nalpha, nparams)
    # Reshape into matrices and drop unoccupied spin channels.
    orbitals = [
        jnp.reshape(orbital, [spin, -1, sum(spins) if full_det else spin])
        for spin, orbital in zip(active_spin_channels, orbitals)
        if spin > 0
    ]  # shape (nalpha, ndet, nalpha)
    orbitals = [
        jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals
    ]  # shape (ndet, nalpha, nalpha)
    phases = network_block.eval_phase(
        x, klist=klist, ndim=3, spins=spins, full_det=full_det
    )

    orbitals = [orb * p[None, :, :] for orb, p in zip(orbitals, phases)]
    if full_det:
        orbitals = [jnp.concatenate(orbitals, axis=1)]
    return orbitals, to_env


def eval_func(
    params,
    x,
    klist=None,
    simulation_cell: PyscfCell = None,
    spins=(None, None),
    envelope_type="full",
    full_det=False,
    distance_type="nu",
    method_name="eval_slogdet",
):
    """Generates the wavefunction of simulation cell (NVT Fixed).

    Args:
        params: Parameter dict.
        x: The input data, a 3N dimensional vector.
        klist: Tuple with occupied k points of the spin up and spin down electrons.
        simulation_cell: PySCF object of simulation cell.
        spins: Tuple with number of spin up and spin down electrons.
        envelope_type: Envelope type string.
        full_det: Specify the mode of wavefunction, spin diagonalized or not.
        distance_type: Distance function type.
        method_name: Specify the returned function of wavefunction.

    Returns:
        Required wavefunction value.

    """
    orbitals, to_env = solid_fermi_net_orbitals(
        params,
        x,
        klist=klist,
        simulation_cell=simulation_cell,
        spins=spins,
        envelope_type=envelope_type,
        distance_type=distance_type,
        full_det=full_det,
    )
    if method_name == "eval_logabs_network":
        _, result = network_block.logdet_matmul(orbitals)
    elif method_name == "eval_log_network":
        sign, slogdet = network_block.logdet_matmul(orbitals)
        result = slogdet + 1j * sign
    elif method_name == "eval_phase_and_logabs_network":
        result = network_block.logdet_matmul(orbitals)
    elif method_name == "eval_orbitals":
        result = orbitals
    else:
        raise ValueError("Unrecognized method name")

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
    distance_type="nu",
    method_name="eval_log_network",
):
    """Creates a Fermionic Neural Network (NVT Fixed).

    Args:
        envelope_type: Specify envelope.
        atom_center_dynamic: Whether atom centers are dynamic (unused in fixed).
        is_rezero: Whether to use rezero (unused in fixed).
        bias_orbitals: Whether to contain bias in the last layer of orbitals.
        use_last_layer: Whether to use two-electron feature in the last layer.
        klist: Occupied k points from HF.
        simulation_cell: Simulation cell.
        full_det: Specify the mode of wavefunction, spin diagonalized or not.
        hidden_dims: Specify the dimension of one-electron and two-electron layer.
        determinants: The number of determinants used.
        distance_type: Distance function type.
        method_name: Specify the returned function.

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

    del (
        atom_center_dynamic,
        is_rezero,
    )

    method = namedtuple("method", ["init", "apply"])
    init = functools.partial(
        init_solid_fermi_net_params,
        atoms=jnp.asarray(simulation_cell.atom_coords(), dtype=simulation_cell.a.dtype),
        spins=simulation_cell.nelec,
        envelope_type=envelope_type,
        bias_orbitals=bias_orbitals,
        use_last_layer=use_last_layer,
        full_det=full_det,
        hidden_dims=hidden_dims,
        determinants=determinants,
        distance_type=distance_type,
    )
    network = functools.partial(
        eval_func,
        simulation_cell=simulation_cell,
        klist=klist,
        spins=simulation_cell.nelec,
        envelope_type=envelope_type,
        full_det=full_det,
        distance_type=distance_type,
        method_name=method_name,
    )
    method.init = init
    method.apply = network
    return method
