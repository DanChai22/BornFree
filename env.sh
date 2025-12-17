# conda create -n BornFree python=3.11
pip install jax==0.4.35 jaxlib==0.4.34 "jax-cuda12-plugin[with_cuda]"==0.4.34 jax-cuda12-pjrt==0.4.34 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install kfac_jax==0.0.6
pip install numpy~=2.0
pip install omegaconf~=2.3
pip install universal_pathlib~=0.2.2
pip install ml_collections
pip install pyscf==2.8.0
pip install ase
pip install pandas matplotlib
pip install pymatgen==2025.2.18
pip install folx
pip install wandb
pip install tables
pip install pymatviz
pip install jupyter
pip install --force-reinstall nvidia-cuda-cupti-cu12==12.8.90 nvidia-cuda-nvcc-cu12==12.8.93 nvidia-cuda-runtime-cu12==12.8.90
pip install optax