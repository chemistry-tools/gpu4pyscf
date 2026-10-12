# Native RTX 3080 installation

Build the complete fork when using its numerical Python sources. The installed-release
compatibility adapter is a separate experiment; it is not needed by a native installation.
The RTX 3080 has compute capability 8.6, so this build contains native `sm_86` code.

## Prerequisites

Use Linux x86_64, Python 3.12, GCC/G++/gfortran 14, CMake, Ninja and the CUDA 13.0.2
development toolkit. The toolkit must include NVCC, NVVM, CRT, CCCL headers and both
`libcudart_static.a` and `libcudadevrt.a`. Python CUDA runtime wheels alone do not supply
every build dependency. NVIDIA's [redistribution manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.0.2.json)
provides individually downloadable components and their checksums. A private toolkit avoids
changing the host's installed CUDA runtime.

Install Python dependencies in a separate environment. The native integration checks use
PySCF 2.14.0, CuPy 14.2.0, cuTENSOR 2.3.1, GPU LibXC 0.8.1, NumPy 2.5.3, SciPy 1.18.1
and ASE 3.27.0. LibXC remains the normal external package dependency.

```sh
python3.12 -m venv /path/native-venv
/path/native-venv/bin/python -m pip install \
  pyscf==2.14.0 cupy-cuda13x==14.2.0 cutensor-cu13==2.3.1 \
  gpu4pyscf-libxc-cuda13x==0.8.1 numpy==2.5.3 scipy==1.18.1 ase==3.27.0 \
  pyscf-dispersion==1.5.0 geometric==1.1.1 threadpoolctl==3.7.0 \
  setuptools==84.0.0 wheel cmake ninja pytest==9.1.1
```

## Build and install

Set the actual toolkit, environment and checkout paths before running these commands.
Follow the GPU 1, four-CPU and sequential-job rules in [RUNNING.md](RUNNING.md), including
the shared benchmark lock on the designated host. Choose a fresh wheel directory.

```sh
export CUDA_PATH=/path/cuda-13.0.2
export PATH=/path/native-venv/bin:$CUDA_PATH/bin:$PATH
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4 OMP_THREAD_LIMIT=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export BLIS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 OMP_DYNAMIC=FALSE OMP_MAX_ACTIVE_LEVELS=1
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:/path/native-venv/lib/python3.12/site-packages/cutensor/lib
export CMAKE_BUILD_PARALLEL_LEVEL=4
export CMAKE_CONFIGURE_ARGS="-G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHITECTURES=86-real '-DCMAKE_C_FLAGS=-O3 -march=native' '-DCMAKE_CXX_FLAGS=-O3 -march=native' '-DCMAKE_Fortran_FLAGS=-O3 -march=native' -DCMAKE_CUDA_FLAGS_RELEASE=-O3"
cd /path/gpu4pyscf
taskset -c 4-7 python -m pip wheel --no-deps --no-build-isolation --wheel-dir /path/new-wheels .
taskset -c 4-7 python -m pip install --no-deps --force-reinstall /path/new-wheels/gpu4pyscf_cuda13x-*.whl
```

The package build honors `CMAKE_BUILD_PARALLEL_LEVEL`; without a supplied job setting it
defaults to four jobs. Quoted CMake arguments preserve flags containing spaces. OpenMP,
solvent support, CUTLASS and the other fork libraries remain enabled. `BUILD_MARCH_NATIVE`
stays disabled because that upstream option also enables global fast math; the explicit
host flags above preserve the existing floating-point rules. `86-real` wheels target
compute capability 8.6 and must be rebuilt for a different GPU architecture.

## Check the installed package

Run from outside the checkout, without a checkout `PYTHONPATH`. Confirm that `gpu4pyscf`
imports from the new environment's `site-packages`, and retain its native library hashes.
Run the opt-in `test_ase_gpu.py`, `test_vv10_response.py` and `test_native_grid_response.py` checks with
`DFT_GPU_TEST=1`. The former checks energies, full-response forces, geometry changes, SCF
reuse and finite differences; the latter compares fused fields and derivatives with the
newly built native calls. Native streaming tests compare density/products and differentiate
energies; set `DFT_REFERENCE_GRADIENT` to a retained matching native baseline source for
complete-force comparisons. `test_force_response.py` uses the older wheel compatibility
adapter and is not the native installation check. [PERFORMANCE.md](PERFORMANCE.md) records
the measured build and numerical validation separately from older wheel results.
