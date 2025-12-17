# Copyright (c) 2025 Shengdu Chai
#
# Licensed under the Apache License, Version 2.0.

import logging
import os

import ase.io
import numpy as np
from ase.atoms import Atoms
from ase.visualize import view
from pyscf.pbc.gto import Cell as PyscfCell

from BornFree import supercell
from BornFree.config import (
    bcc_config,
    read_cif,
)
from BornFree.constants import HYDROGEN_CIF_DIR


class SimulationCellGenerator:
    """Generator for various types of simulation cells used in quantum Monte Carlo calculations.

    This class provides a unified interface for creating different crystal structures
    and simulation cells, including body-centered cubic (bcc) and hydrogen structures from CIF files.

    The generator supports:
    - Built-in crystal structures (bcc)
    - Hydrogen structures from CIF files
    - Supercell generation with specified copy numbers
    - Various basis sets for electronic structure calculations

    Attributes:
        ncopy: Array specifying the number of unit cell replications in each direction.
        basis: Basis set name for electronic structure calculations.
        gen_k: Whether to generate special k-points for periodic calculations.
        identity: Whether to use identity transformation for CIF structures.

    """

    def __init__(
        self,
        ncopy: np.ndarray,
        basis: str = "sto-3g",
        gen_k: bool = False,
        identity: bool = False,
    ) -> None:
        """Initialize the simulation cell generator.

        Args:
            ncopy: Array of shape (3,) specifying the number of unit cell copies
                in each spatial direction (x, y, z).
            basis: Name of the basis set to use for electronic structure calculations.
                Default is 'sto-3g'.
            gen_k: Whether to generate k-points for periodic boundary condition
                calculations. Default is False.
            identity: Whether to use identity transformation when reading CIF files.
                Default is False.

        Raises:
            ValueError: If ncopy is not a 3-element array.
            TypeError: If basis is not a string.

        """
        if not isinstance(ncopy, np.ndarray) or ncopy.size != 3:
            raise ValueError("ncopy must be a numpy array of size 3")
        if not isinstance(basis, str):
            raise TypeError("basis must be a string")

        self.ncopy = ncopy
        self.basis = basis
        self.gen_k = gen_k
        self.identity = identity

    def get_bcc_cell(self, rs: float) -> PyscfCell:
        """Generate a body-centered cubic (BCC) crystal structure simulation cell.

        Creates a BCC lattice where atoms are located at the corners and center
        of cubic unit cells.

        Args:
            rs: Wigner-Seitz radius in atomic units (Bohr radii).

        Returns:
            PySCF Cell object representing the BCC crystal structure.

        Raises:
            ValueError: If rs is not positive.

        """
        if rs <= 0:
            raise ValueError("rs must be positive")
        return bcc_config.make_cell(rs, self.ncopy, self.basis, self.gen_k)

    def get_cif_cell(self, rs: float, cif_path: str) -> PyscfCell:
        """Generate a simulation cell from a CIF (Crystallographic Information File).

        Reads crystal structure information from a CIF file and creates a
        corresponding simulation cell with the specified density.

        Args:
            rs: Wigner-Seitz radius in atomic units (Bohr radii).
            cif_path: Path to the CIF file containing crystal structure information.

        Returns:
            PySCF Cell object representing the crystal structure from the CIF file.

        Raises:
            FileNotFoundError: If the CIF file is not found at the specified path.
            ValueError: If rs is not positive or CIF file is invalid.
            IOError: If there's an error reading the CIF file.

        """
        if rs <= 0:
            raise ValueError("rs must be positive")
        if not os.path.isfile(cif_path):
            logging.error(f"CIF file not found at path: {cif_path}")
            raise FileNotFoundError(f"CIF file not found at path: {cif_path}")

        try:
            ase_cell = ase.io.read(cif_path)
        except Exception as e:
            raise OSError(f"Error reading CIF file {cif_path}: {e}")

        return read_cif.make_cell(
            ase_cell, rs, self.ncopy, self.basis, self.gen_k, self.identity
        )

    def get_cell(
        self,
        rs: float,
        structure: str,
        cif_path: str = "",
    ) -> PyscfCell:
        """Generate a simulation cell for the specified crystal structure.

        This is the main interface method that dispatches to appropriate
        structure-specific generation methods based on the structure parameter.

        Supported structures:
        - 'bcc': Body-centered cubic structure
        - 'csiv', 'cmca4', 'cmca12', 'p21c', 'pca21', 'fmmm':
          Hydrogen structures from CIF files

        Args:
            rs: Wigner-Seitz radius in atomic units (Bohr radii).
            structure: Name of the crystal structure to generate.
            cif_path: Path to CIF file (required for CIF-based structures).
                If not provided for supported structures, uses default path.

        Returns:
            PySCF Cell object representing the requested crystal structure.

        Raises:
            ValueError: If structure name is invalid or rs is not positive.
            FileNotFoundError: If required CIF file is not found.

        """
        if rs <= 0:
            raise ValueError("rs must be positive")

        if structure == "bcc":
            cell = self.get_bcc_cell(rs)
        elif structure in [
            "p21c8",
            "p21c24",
            "pca21",
            "hcpy",
            "cmcm",
            "p63m",
            "p63mmc",
        ]:
            # If cif_path is not provided, use default path
            if not cif_path:
                cif_path = os.path.join(HYDROGEN_CIF_DIR, f"{structure}.cif")
            cell = self.get_cif_cell(rs, cif_path)
        else:
            raise ValueError(f"Invalid structure: {structure}")
        return cell

    def get_simulation_cell(
        self,
        rs: float,
        structure: str,
        cif_path: str = "",
    ) -> PyscfCell:
        """Generate a supercell for quantum Monte Carlo simulations.

        This method creates a supercell from the base unit cell, which is
        necessary for finite-size quantum Monte Carlo calculations with
        periodic boundary conditions.

        Args:
            rs: Wigner-Seitz radius in atomic units (Bohr radii).
            structure: Name of the crystal structure to generate.
            cif_path: Path to CIF file (required for CIF-based structures).

        Returns:
            PySCF Cell object representing the supercell ready for QMC simulations.

        Raises:
            ValueError: If structure name is invalid or rs is not positive.
            FileNotFoundError: If required CIF file is not found.

        """
        cell = self.get_cell(rs, structure, cif_path)
        return supercell.get_supercell(cell, np.diag([1, 1, 1]))

    @staticmethod
    def pyscf_cell_to_ase(cell: PyscfCell, is_view: bool = False) -> Atoms:
        """Convert a PySCF Cell object to an ASE Atoms object.

        This utility method facilitates interoperability between PySCF
        (used for electronic structure calculations) and ASE (Atomic Simulation
        Environment, used for structure manipulation and visualization).

        Args:
            cell: PySCF Cell object to convert.
            is_view: Whether to display the structure using ASE's 3D viewer.
                Default is False.

        Returns:
            ASE Atoms object with the same structure as the input PySCF cell.

        Raises:
            AttributeError: If the PySCF cell lacks required methods or attributes.

        """
        symbols = [cell.atom_pure_symbol(i) for i in range(cell.natm)]
        positions = cell.atom_coords()  # Returns atomic positions in Bohr
        lattice_vectors = cell.lattice_vectors()  # Returns lattice vectors in Bohr
        ase_cell = Atoms(
            symbols=symbols, positions=positions, cell=lattice_vectors, pbc=True
        )
        if is_view:
            view(ase_cell, viewer="x3d")
        return ase_cell
