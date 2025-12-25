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

import contextlib
import dataclasses
import datetime
import os
from collections.abc import Sequence
from typing import Any

import ml_collections
import numpy as np
import pandas as pd
import yaml
from absl import logging

from BornFree import base_config
from BornFree.config import utils


class Writer(contextlib.AbstractContextManager):
    """Write data to CSV files with optional stdout logging.

    This class provides a context manager for writing tabular data to CSV files.
    It can also log the data to stdout if desired.
    """

    def __init__(
        self,
        name: str,
        schema: Sequence[str],
        directory: str = "logs/",
        iteration_key: str | None = "t",
        log: bool = True,
    ):
        """Initialize a CSV writer.

        Args:
            name: File name for CSV (without extension)
            schema: Sequence of column keys corresponding to each data item
            directory: Directory path to write file to
            iteration_key: If not None, include iteration index as first column
                           with this key
            log: Also log each entry to stdout via absl logging

        """
        self._schema = schema

        # Create directory if it doesn't exist
        if not os.path.isdir(directory):
            with contextlib.suppress(FileExistsError):
                os.makedirs(directory)
        self._filename = os.path.join(directory, name + ".csv")
        self._iteration_key = iteration_key
        self._log = log

    def __enter__(self):
        should_add_header = not os.path.exists(self._filename)

        self._file = open(self._filename, "a+", encoding="utf-8")

        if should_add_header:
            # write top row of csv
            if self._iteration_key:
                self._file.write(f"{self._iteration_key},")
            self._file.write(",".join(self._schema) + "\n")
        return self

    def write(self, t: int, **data: Any):
        """Write data row to CSV file and optionally log to stdout.

        Args:
            t: Iteration index
            **data: Data items with keys matching the schema

        Raises:
            ValueError: If data contains keys not in the schema

        """
        row = [str(data.get(key, "")) for key in self._schema]
        if self._iteration_key:
            row.insert(0, str(t))
        for key in data:
            if key not in self._schema:
                raise ValueError(f"Not a recognized key for writer: {key}")

        # write the data to csv
        self._file.write(",".join(row) + "\n")

        # write the data to abseil logs
        if self._log:
            logging.info("Iteration %s: %s", t, data)

    def flush(self):
        """Flush the CSV file buffer to disk."""
        self._file.flush()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        self._file.close()


def clean_csv(ckpt_save_path: str, name: str, train_schema: list[str] | None = None) -> pd.DataFrame:
    """Clean a CSV file by removing duplicates and fixing headers.

    Args:
        ckpt_save_path: Directory containing the CSV file
        name: Base name of the CSV file (without extension)
        train_schema: Optional custom header for the CSV file

    Returns:
        Cleaned pandas DataFrame

    Raises:
        ValueError: If the number of columns doesn't match the provided schema

    """
    file_path = os.path.join(ckpt_save_path, f"{name}.csv")
    print(f"Starting cleaning of {name}")

    # Load CSV file
    if train_schema:
        # Read without header if custom schema provided
        df = pd.read_csv(file_path, header=None)

        # Validate column count
        if len(df.columns) != len(train_schema):
            print(f"File: {file_path}")
            print(f"Number of columns: {len(df.columns)}")
            print(f"Custom header length: {len(train_schema)}")
            print(f"Custom header: {train_schema}")
            raise ValueError("The number of columns does not match the custom header.")

        # Apply custom header
        df.columns = train_schema
    else:
        # Use existing header
        df = pd.read_csv(file_path)

    # Clean data
    df.drop_duplicates(subset=["step"], keep="last", inplace=True)
    df.drop(df[df["step"] == "step"].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Save cleaned file
    df.to_csv(file_path, index=False)
    print(f"{name} cleaned successfully")

    return df


def restructure_dict(data):
    """Restructure dictionary by converting list values to indexed dictionaries.

    Args:
        data: Dictionary that may contain list values

    Returns:
        Dictionary with list values converted to dictionaries with layer_i keys
    """
    new_dict = {}

    for key, value in data.items():
        if isinstance(value, list):
            # Convert list into a dictionary with indexed keys
            new_dict[key] = {f"layer_{i}": v for i, v in enumerate(value)}
        else:
            new_dict[key] = value  # Keep original structure for non-list elements

    return new_dict


def _get_wandb_training_components(config: base_config.BornFreeConfig) -> list[str]:
    """Generate training parameter components."""
    # Get base training components and reformat for wandb
    base_components = utils._get_training_components(config)

    components = [
        f"batch{base_components[0]}",  # batch{batch_size}
        f"{config.optim.optimizer}",  # optimizer (not in base)
        f"lr{float(base_components[1]):.3f}",  # lr with 3 decimal places
    ]

    return components


def _get_wandb_annealing_components(config: base_config.BornFreeConfig) -> list[str]:
    """Generate annealing parameter components."""
    # Get base annealing components from utils and modify format
    base_components = utils._get_annealing_components(config, include_final_temp=False)

    # Replace abbreviated names with full names
    components = []
    for component in base_components:
        modified_component = component.replace("l_sampling", "local_sampling")
        modified_component = modified_component.replace("l_steps", "local_steps")
        components.append(modified_component)

    # Add final_temp
    components.append(f"final_temp{config.mcmc.annealing.final_temp}")

    return components


def _get_wandb_npt_components(config: base_config.BornFreeConfig) -> list[str]:
    """Generate NPT-specific components."""
    components = [
        f"{config.target_pressure:.2f}GPa",
        f"{config.crystal.lattice.mode}",
        f"warmupsteps{config.strategy.warmup_steps}",
        f"optsteps{config.strategy.opt_steps}",
        f"geooptsteps{config.strategy.geo_opt_steps}",
    ]
    return components


def get_wandb_save_name(config: base_config.BornFreeConfig) -> str:
    """Generate a descriptive save name for wandb logging.

    The name is constructed from several components in the following order:
    1. Basic configuration (ensemble type, nuclear treatment)
    2. Network architecture details
    3. Crystal structure information
    4. Training parameters (batch size, optimizer)
    5. MCMC parameters
    6. NPT-specific parameters (if applicable)
    7. Additional flags (inference, debug mode)

    Args:
        config: BornFreeConfig object containing all configuration parameters

    Returns:
        str: Formatted save name for wandb logging

    """
    components = []
    basic_components = [
        f"{config.ensemble}",
        f"{config.nuclear_treatment}",
        f"{config.network.detnet.distance_type}",
    ]

    # Collect all components
    components.extend(basic_components)
    components.extend(utils._get_network_components(config))
    components.extend(utils._get_crystal_components(config, include_lattice_mode=False))
    components.extend(_get_wandb_training_components(config))
    components.extend(utils._get_mcmc_components(config))
    components.extend(_get_wandb_annealing_components(config))

    # Add NPT-specific parameters if applicable
    if config.ensemble == "NPT":
        components.extend(_get_wandb_npt_components(config))

    # Add precision
    components.append(f"{config.precision}")

    # Add additional flags with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.infer:
        components.append(f"infer_{timestamp}")
    if config.debug.deterministic:
        components.append(f"debug_{timestamp}")

    return "_".join(components)


def convert_data_for_yaml(obj):
    """Recursively convert tuples to lists and NumPy objects to Python types."""
    if isinstance(obj, tuple):
        return {"__tuple__": [convert_data_for_yaml(v) for v in obj]}  # Mark tuples
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    elif isinstance(obj, np.generic):
        return obj.item()  # Convert NumPy scalars to Python types
    elif isinstance(obj, dict):
        return {k: convert_data_for_yaml(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_data_for_yaml(v) for v in obj]
    return obj


def mlcollections_to_dict(nested_obj: Any) -> dict:
    """Convert ml_collections.ConfigDict to regular dictionary.

    Args:
        nested_obj: Object to convert (ConfigDict, dict, or other)

    Returns:
        dict: Converted dictionary

    """
    if isinstance(nested_obj, ml_collections.ConfigDict | dict):
        return {key: mlcollections_to_dict(value) for key, value in nested_obj.items()}
    else:
        return nested_obj


def dict_to_mlcollections(d: dict) -> ml_collections.ConfigDict:
    """Convert dictionary to ml_collections.ConfigDict.

    Args:
        d: Dictionary to convert

    Returns:
        ml_collections.ConfigDict: Converted ConfigDict

    """
    config = ml_collections.ConfigDict()
    for key, value in d.items():
        if isinstance(value, dict):
            config[key] = dict_to_mlcollections(value)
        else:
            config[key] = value
    return config


def load_yaml_to_config(base_folder: str, keys: list) -> ml_collections.ConfigDict:
    """Load config.yaml from base folder and convert to ml_collections.ConfigDict.

    Args:
        base_folder: Path to folder containing config.yaml
        keys: List of keys to extract from the configuration

    Returns:
        ml_collections.ConfigDict: Configuration loaded from yaml

    Raises:
        FileNotFoundError: If config.yaml is not found in base_folder

    """
    yaml_path = os.path.join(base_folder, "config.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found at {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        yaml_config = yaml.unsafe_load(f)

    # Convert yaml_config to dict if it's a BornFreeConfig object
    if hasattr(yaml_config, "__dict__"):
        yaml_config = yaml_config.__dict__
    elif isinstance(yaml_config, dict) and "__BornFreeConfig__" in yaml_config:
        # Handle case where yaml contains BornFreeConfig type information
        yaml_config = {key: value for key, value in yaml_config.items() if key != "__BornFreeConfig__"}

    return dict_to_mlcollections({key: yaml_config.get(key) for key in keys})


def save_config_to_yaml(config: base_config.BornFreeConfig, save_path: str) -> None:
    """Save ml_collections.ConfigDict to yaml file.

    Args:
        config: Configuration to save
        save_path: Path to save the yaml file

    Raises:
        ValueError: If save_path doesn't end with .yaml

    """
    if not save_path.endswith(".yaml"):
        raise ValueError("Save path must end with .yaml")

    # Convert to dict first
    config_dict = mlcollections_to_dict(config)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # Save to yaml
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(convert_data_for_yaml(config_dict), f, default_flow_style=False)


def _dataclass_to_dict_for_configdict(instance: Any) -> Any:
    """Recursively converts a dataclass instance to a dictionary.

    Non-dataclass objects (like numpy arrays, PySCF Cell instances) are returned as is,
    as ml_collections.ConfigDict can store them directly.
    """
    if dataclasses.is_dataclass(instance) and not isinstance(instance, type):
        result = {}
        for field_info in dataclasses.fields(instance):
            value = getattr(instance, field_info.name)
            result[field_info.name] = _dataclass_to_dict_for_configdict(value)
        return result
    elif isinstance(instance, list):
        return [_dataclass_to_dict_for_configdict(item) for item in instance]
    elif isinstance(instance, tuple):
        return tuple(_dataclass_to_dict_for_configdict(item) for item in instance)
    else:
        return instance


def update_multiple_keys_from_yaml(
    base_cfg: ml_collections.ConfigDict, yaml_path: str, target_keys_list: list[str]
) -> ml_collections.ConfigDict:
    """Loads configuration from a YAML file and updates multiple specified keys in the base_cfg.

    The YAML file is expected to deserialize into a BornFreeConfig object (or a
    compatible structure) at its root. For each key in target_keys_list,
    the corresponding part is extracted from the loaded YAML object and used
    to update base_cfg.

    Args:
        base_cfg: The base ml_collections.ConfigDict object to update.
        yaml_path: Path to the YAML configuration file.
        target_keys_list: A list of strings, where each string is a key
                          (potentially dot-separated for nesting, e.g., "optim"
                          or "system.pyscf_cell") to update in base_cfg using
                          data from the corresponding path in the loaded YAML object.

    Returns:
        The updated ml_collections.ConfigDict object.

    """
    try:
        with open(yaml_path, encoding="utf-8") as f:
            # Using yaml.unsafe_load, consistent with writers.py.
            # WARNING: Ensure YAML is from a trusted source.
            loaded_object_from_yaml = yaml.unsafe_load(f)
    except FileNotFoundError:
        print(f"ERROR: YAML configuration file '{yaml_path}' not found.")
        raise
    except yaml.YAMLError as e:
        print(f"ERROR: Could not parse YAML file '{yaml_path}': {e}")
        raise
    except Exception as e:  # Catch other potential errors (e.g., module not found for tags)
        print(f"ERROR: An unexpected error occurred while loading YAML '{yaml_path}': {e}")
        raise

    if not loaded_object_from_yaml:
        print(f"WARNING: YAML file '{yaml_path}' is empty or parsed to None. No updates will be performed.")
        return base_cfg

    was_locked = base_cfg.is_locked
    if was_locked:
        base_cfg.unlock()

    try:
        for target_key in target_keys_list:
            source_for_update_value: Any
            current_part_from_yaml = loaded_object_from_yaml
            key_segments = target_key.split(".")

            # 1. Extract data for the current target_key from loaded_object_from_yaml
            try:
                for i, key_segment in enumerate(key_segments):
                    path_so_far = ".".join(key_segments[: i + 1])
                    if dataclasses.is_dataclass(current_part_from_yaml) and not isinstance(
                        current_part_from_yaml, type
                    ):
                        if not hasattr(current_part_from_yaml, key_segment):
                            raise AttributeError(
                                f"Field '{key_segment}' (path: '{path_so_far}') not found in "
                                f"dataclass {type(current_part_from_yaml)}."
                            )
                        current_part_from_yaml = getattr(current_part_from_yaml, key_segment)
                    elif isinstance(current_part_from_yaml, dict):
                        if key_segment not in current_part_from_yaml:
                            raise KeyError(f"Key '{key_segment}' (path: '{path_so_far}') not found in dict.")
                        current_part_from_yaml = current_part_from_yaml[key_segment]
                    else:
                        raise ValueError(
                            f"Cannot traverse using key '{key_segment}' (path: '{path_so_far}'). "
                            f"Part is type '{type(current_part_from_yaml)}', not traversable."
                        )
                source_for_update_value = current_part_from_yaml
            except (AttributeError, KeyError, ValueError) as e:
                print(f"WARNING: Could not extract data for key '{target_key}' from YAML: {e}. Skipping this key.")
                continue  # Move to the next key in target_keys_list

            # 2. Convert extracted data to a ConfigDict-friendly payload
            update_payload = _dataclass_to_dict_for_configdict(source_for_update_value)

            # 3. Update base_cfg at the current target_key
            cfg_ptr = base_cfg
            try:
                for i, k_segment in enumerate(key_segments[:-1]):  # Navigate to parent in base_cfg
                    path_so_far_cfg = ".".join(key_segments[: i + 1])
                    if k_segment not in cfg_ptr:
                        cfg_ptr[k_segment] = ml_collections.ConfigDict()
                    elif not isinstance(cfg_ptr[k_segment], ml_collections.ConfigDict):
                        if isinstance(cfg_ptr[k_segment], dict):
                            cfg_ptr[k_segment] = ml_collections.ConfigDict(cfg_ptr[k_segment])
                        else:
                            print(
                                f"WARNING: Overwriting non-ConfigDict/dict at '{path_so_far_cfg}' "
                                f"in base_cfg for key '{target_key}'."
                            )
                            cfg_ptr[k_segment] = ml_collections.ConfigDict()
                    cfg_ptr = cfg_ptr[k_segment]

                final_key_segment = key_segments[-1]
                if (
                    final_key_segment in cfg_ptr
                    and isinstance(cfg_ptr.get(final_key_segment), ml_collections.ConfigDict)
                    and isinstance(update_payload, dict)
                ):
                    cfg_ptr[final_key_segment].update(update_payload)
                else:
                    cfg_ptr[final_key_segment] = update_payload
            except Exception as e:
                print(f"WARNING: Failed to update base_cfg for key '{target_key}': {e}. Skipping this key.")
                continue  # Move to the next key

    finally:
        if was_locked:
            base_cfg.lock()

    return base_cfg
