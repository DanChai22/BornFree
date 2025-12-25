# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This file may have been modified by Shengdu Chai.
# Modifications Copyright (c) 2025 Shengdu Chai

from setuptools import find_packages, setup

REQUIRED_PACKAGES = [
    "jax==0.4.35",
    "jaxlib==0.4.34",
    "jax-cuda12-plugin[with_cuda]==0.4.36",
    "jax-cuda12-pjrt==0.4.36",
    "kfac_jax==0.0.6",
    "folx",
    "numpy~=2.0",
    "scipy",
    "pandas",
    "omegaconf~=2.3",
    "universal_pathlib~=0.2.2",
    "ml_collections",
    "pyscf==2.8.0",
    "ase",
    "pymatgen==2025.2.18",
    "matplotlib",
    "wandb",
    "tables",
    "h5py",
    "absl-py",
    "attrs",
    "dataclasses",
    "networkx",
    "ordered-set",
    "typing",
    "chex",
    "optax",
    "pymatviz",
    # NVIDIA CUDA packages (for GPU support)
    "nvidia-cuda-cupti-cu12==12.8.90",
    "nvidia-cuda-nvcc-cu12==12.8.93",
    "nvidia-cuda-runtime-cu12==12.8.90",
]

EXTRAS_REQUIRE = {
    "dev": [
        "pytest",
        "flake8",
        "black",
        "jupyter",
    ],
}

setup(
    name="BornFree",
    version="1.0.0",
    description="Beyond Born-Oppenheimer Real-space Neural-network Framework for \
    Enthalpy Extremization: A first-principles quantum Monte Carlo study of \
    high-pressure solid hydrogen",
    author="Shengdu Chai",
    author_email="sdchai24@m.fudan.edu.cn",
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRAS_REQUIRE,
    packages=find_packages(),
    scripts=["bin/bornfree"],
    license="Apache 2.0",
    python_requires=">=3.11",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    dependency_links=["https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"],
)
