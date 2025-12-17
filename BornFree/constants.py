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

"""Constants and utility functions for BornFree framework.

This module contains directory paths, physical constants, and parallel computing
utilities used throughout the BornFree package.
"""

import functools
import os
from pathlib import Path
from typing import TypeVar

import jax
import kfac_jax
from jax import core

ROOT_DIR = Path(__file__).parent.parent.resolve()
HYDROGEN_CIF_DIR = os.getenv("BORNFREE_CIF_DIR", str(ROOT_DIR / "Hydrogen_cif"))
OUTPUT_DIR = os.getenv("BORNFREE_OUTPUT_DIR", str(ROOT_DIR / "outputs"))
STRUCTURE_RESULT_DIR = os.path.join(OUTPUT_DIR, "structure_result")
LOG_DIR = os.getenv("BORNFREE_LOG_DIR", str(ROOT_DIR / "log"))
TEST_RESULT_DIR = os.path.join(OUTPUT_DIR, "test_result")

for dir_path in [
    OUTPUT_DIR,
    STRUCTURE_RESULT_DIR,
    LOG_DIR,
    TEST_RESULT_DIR,
]:
    os.makedirs(dir_path, exist_ok=True)

T = TypeVar("T")

PMAP_AXIS_NAME = "qmc_pmap_axis"

pmap = functools.partial(jax.pmap, axis_name=PMAP_AXIS_NAME)


def wrap_if_pmap(p_func):
    """Wrap function to conditionally apply pmap operations.

    Args:
        p_func: Function to wrap

    Returns:
        Wrapped function that only applies pmap when axis_name is valid
    """
    def p_func_if_pmap(obj, axis_name):
        try:
            core.axis_frame(axis_name)
            return p_func(obj, axis_name)
        except NameError:
            return obj

    return p_func_if_pmap


# Shortcut for kfac utils
psum = functools.partial(kfac_jax.utils.psum_if_pmap, axis_name=PMAP_AXIS_NAME)
pmean = functools.partial(kfac_jax.utils.pmean_if_pmap, axis_name=PMAP_AXIS_NAME)
all_gather = functools.partial(kfac_jax.utils.wrap_if_pmap(jax.lax.all_gather))

# Physical constants
# Atomic mass unit in electron mass (m_u / m_e)
# Used for converting nuclear masses to atomic units
ATOM_MASS = 1836.65  # atomic mass unit in electron masses
DEFAULT_MOLECULE_THRESHOLD_MAX = 1.6  # Angstroms
DEFAULT_MOLECULE_THRESHOLD_MIN = 1.2  # Angstroms
DEFAULT_ORIENTATION_BINS = 100
BOHR_TO_ANGSTROM = 0.529177
# Default XRD parameters
DEFAULT_XRD_WAVELENGTH = 0.7107
DEFAULT_XRD_RANGE = (22, 38)
DEFAULT_XRD_STEP = 0.01
DEFAULT_XRD_GAMMA = 0.01
