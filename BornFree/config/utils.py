# Copyright (c) 2025 Shengdu Chai
#
# Licensed under the Apache License, Version 2.0.


from BornFree import base_config
from BornFree.constants import STRUCTURE_RESULT_DIR


def _get_base_path(config: base_config.BornFreeConfig) -> str:
    """Generate base directory path from configuration."""
    return (
        f"{STRUCTURE_RESULT_DIR}/"
        f"{config.ensemble}/"
        f"{config.nuclear_treatment}/"
        f"{config.mcmc.mcmc_type}/"
        f"structure_{config.crystal.structure}/"
        f"{config.network.detnet.distance_type}_embedding"
    )


def _get_network_components(config: base_config.BornFreeConfig) -> list[str]:
    """Generate network configuration components."""
    return [
        f"{config.network.detnet.atom_center_dynamic}",
        f"{config.network.detnet.determinants}",
        f"rezero{config.network.detnet.is_rezero}",
    ]


def _get_kpts_str(config: base_config.BornFreeConfig) -> str:
    """Generate k-points string (e.g., "222" for [2, 2, 2])."""
    return "".join(str(n) for n in config.crystal.kpts.number)


def _get_crystal_components(config: base_config.BornFreeConfig, include_lattice_mode: bool = False) -> list[str]:
    """Generate crystal structure parameter components."""
    components = [
        f"{config.crystal.structure}",
        "".join(map(str, config.crystal.ncopy)),
        f"rs{config.crystal.rs:.2f}",
        f"twist({','.join(f'{x:.3f}' for x in config.network.twist)})",
    ]
    if include_lattice_mode:
        components.append(f"{config.crystal.lattice.mode}")

    return components


def _get_training_components(config: base_config.BornFreeConfig) -> list[str]:
    """Generate training parameter components."""
    return [
        f"{config.batch_size}",
        f"{config.optim.lr.rate}",
    ]


def _get_mcmc_components(config: base_config.BornFreeConfig) -> list[str]:
    """Generate MCMC parameter components."""
    return [
        f"atomwidth{config.mcmc.atom_move_width}",
        f"elecwidth{config.mcmc.elec_move_width}",
        f"burnin{config.mcmc.burn_in}",
        f"steps{config.mcmc.steps}",
        f"iter{config.mcmc.iter}",
    ]


def _get_annealing_components(config: base_config.BornFreeConfig, include_final_temp: bool = False) -> list[str]:
    """Generate annealing parameter components."""
    components = [
        f"{config.mcmc.annealing.annealing_type}",
        f"annealing_steps{config.mcmc.annealing.annealing_steps}",
        f"cell_width{config.mcmc.annealing.cell_annealing_width}",
        f"l_sampling{config.mcmc.annealing.local_sampling}",
        f"l_steps{config.mcmc.annealing.local_steps}",
        f"beta{config.mcmc.annealing.beta}",
        f"initial_temp{config.mcmc.annealing.initial_temp}",
    ]
    if include_final_temp:
        components.append(f"final_temp{config.mcmc.annealing.final_temp}")

    return components


def get_save_name(config: base_config.BornFreeConfig) -> str:
    """Generate a descriptive save name for file system storage.

    Args:
        config: BornFreeConfig object containing all configuration parameters

    Returns:
        str: Formatted save name for file system storage

    """
    base_path = _get_base_path(config)
    network_str = "_".join(_get_network_components(config))
    kpts_str = _get_kpts_str(config)

    # Determine folder based on ensemble type
    if config.ensemble == "NPT":
        strategy_components = [
            f"{config.strategy.warmup_steps}",
            f"{config.strategy.opt_steps}",
            f"{config.strategy.geo_opt_steps}",
        ]
        strategy_str = "_".join(strategy_components)
        folder_name = f"{base_path}/{network_str}_nk{kpts_str}/{config.crystal.natm}atoms/{strategy_str}"
    else:
        folder_name = f"{base_path}/{network_str}_nk{kpts_str}"

    # Geometry parameters
    geo_str = "_".join([
        f"{config.target_pressure}",
        f"{config.crystal.lattice.mode}",
    ])

    # Collect all parameter strings
    crystal_params = "_".join(_get_crystal_components(config, include_lattice_mode=False))
    training_str = "_".join(_get_training_components(config))
    mcmc_str = "_".join(_get_mcmc_components(config))
    annealing_str = "_".join(_get_annealing_components(config, include_final_temp=False))

    # Construct final file name
    file_name = f"{folder_name}/{crystal_params}_{geo_str}_{training_str}_{mcmc_str}_{annealing_str}_{config.precision}"

    return file_name
