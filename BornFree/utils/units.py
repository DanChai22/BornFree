# Copyright 2020 DeepMind Technologies Limited.
#
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


from typing import TypeVar

import numpy as np

# Type for numerical values (scalar or array)
NumericalLike = TypeVar("NumericalLike", float, np.ndarray)

# Fundamental constants
# ---------------------
# 1 Bohr = 0.52917721067 (12) x 10^{-10} m
# https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
ANGSTROM_BOHR = 0.52917721067  # Angstrom per Bohr
BOHR_ANGSTROM = 1.0 / ANGSTROM_BOHR  # Bohr per Angstrom

# 1 Hartree = 627.509474 kcal/mol
# https://en.wikipedia.org/wiki/Hartree
KCAL_HARTREE = 627.509474  # kcal/mol per Hartree
HARTREE_KCAL = 1.0 / KCAL_HARTREE  # Hartree per kcal/mol

# Derived constants
# ----------------
BOHR_METER = 5.29177e-11  # m per Bohr
HARTREE_EV = 27.2114  # eV per Hartree
EV_J = 1.60218e-19  # J per eV
NPERM2_MBAR = 1e-5  # mbar per N/m^2

# Pressure conversion constants
HABOHR3_MBAR = HARTREE_EV * EV_J / (BOHR_METER**3) * NPERM2_MBAR / 1e6  # mbar per Ha/Bohr^3
MBAR_HABOHR3 = 1.0 / HABOHR3_MBAR  # Ha/Bohr^3 per mbar
GPA_MBAR = 1e-2  # mbar per GPa
GPA_HABOHR3 = GPA_MBAR * MBAR_HABOHR3  # Ha/Bohr^3 per GPa
HABOHR3_GPA = 1.0 / GPA_HABOHR3  # GPa per Ha/Bohr^3


# Conversion functions
# -------------------
def mbar2habohr3(x_m: NumericalLike) -> NumericalLike:
    """Convert pressure from mbar to Hartree/Bohr^3."""
    return x_m * MBAR_HABOHR3


def habohr32mbar(x_h: NumericalLike) -> NumericalLike:
    """Convert pressure from Hartree/Bohr^3 to mbar."""
    return x_h * HABOHR3_MBAR


def habohr32gpa(x_h: NumericalLike) -> NumericalLike:
    """Convert pressure from Hartree/Bohr^3 to GPa."""
    return x_h * HABOHR3_GPA


def gpa2habohr3(x_g: NumericalLike) -> NumericalLike:
    """Convert pressure from GPa to Hartree/Bohr^3."""
    return x_g * GPA_HABOHR3


def bohr2angstrom(x_b: NumericalLike) -> NumericalLike:
    """Convert length from Bohr to Angstrom."""
    return x_b * ANGSTROM_BOHR


def angstrom2bohr(x_a: NumericalLike) -> NumericalLike:
    """Convert length from Angstrom to Bohr."""
    return x_a * BOHR_ANGSTROM


def hartree2kcal(x_h: NumericalLike) -> NumericalLike:
    """Convert energy from Hartree to kcal/mol."""
    return x_h * KCAL_HARTREE


def kcal2hartree(x_k: NumericalLike) -> NumericalLike:
    """Convert energy from kcal/mol to Hartree."""
    return x_k * HARTREE_KCAL


def pressure_estimator(kinetic: float, potential: float, volume: float) -> tuple[float, float]:
    """Calculate pressure and PV term from energy components using the virial theorem.

    This function computes the pressure and PV term using the quantum mechanical
    virial theorem: PV = (2T + V)/3, where T is kinetic energy and V is potential energy.
    This relation is exact for systems with Coulomb interactions.

    Args:
        kinetic: Kinetic energy in Hartree.
        potential: Potential energy in Hartree.
        volume: Volume in Bohr^3.

    Returns:
        A tuple containing:
            - pv: PV term in Hartree.
            - pressure: Pressure in GPa.

    Raises:
        ValueError: If volume is not positive.
        ZeroDivisionError: If volume is zero.

    """
    if volume <= 0:
        raise ValueError("Volume must be positive")

    # Calculate PV term using virial theorem: PV = (2K + V)/3
    pv = (2 * kinetic + potential) / 3

    # Convert to GPa
    pressure = habohr32gpa(pv / volume)

    return pv, pressure
