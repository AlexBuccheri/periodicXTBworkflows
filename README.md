# Data Analysis Repository for QCore-xTB Comparison to Existing GFN1-xTB Periodic Implementation

This repository has two purposes: Workflows and data analysis.

* Workflows were intended as an automated means to run repeatable calculations consistently 
  * The instructions are incomplete

* Data analysis was intended for reproduciblility of publication plots and tables
  * This is largely complete 

**Workflows**

* A dockerfile is provided to install DFTB+ and TBLite locally, for running TBLite workflows within a container.
* QE workflows are present, however no docker build is provided
* QCore is available on [conda](https://anaconda.org/Entos/qcore), however as of Dec 2024, it has not been updated to
  contain the xTB functionality used to produce QCore-xTB data.
   * An updated conda version is pending 

**Data Analysis**

* Data analysis is performed on QCore-xTB, GFN1-xTB(s) (TBLite) and Quantum Espresso data
* Archived data is stored in `tb_results`, which is a cloneable submodule:
```shell
git submodule update --remote
```
* All data analysis is performed in the notebooks, located in the [jupyter](jupyter) folder. Once the results submodule
 is cloned, each notebook should be run to completion.


## Installation

Install the project as an editable python package:

```shell
pip install -e .
```

### Docker Installation for DFTB+ and TBLite

Refer to [Dockerfile](Dockerfile) for instructions.


### Installation of Quantum Espresso

venv with dependencies required to build espresso:

```bash
conda create --name qe7 python=3.7
conda activate qe7
pip install cmake
conda install git=2.32.0
```

How espresso was built for performing these calculations:

```bash
# Ensure GCC, an MPI flavour and a BLAS flavour are available
mkdir ./build
cd ./build
cmake -DCMAKE_Fortran_COMPILER=mpif90 -DCMAKE_C_COMPILER=mpicc  -DCMAKE_INSTALL_PREFIX=/path/to/espresso7 .
make -j 4
make install
```

## <ins> Running TBLite Calculations </ins>

Initialise a container from the Dockerfile, ensuring this repository `tb_benchmarking` is mounted `/tb_benchmarking`:

```bash
# Assumes correct mounting of tis repo in the container
cd /tb_benchmarking 
source docker_venv/bin/activate
pip install --upgrade setuptools pip
cd /dftbplus/tools/dptools && pip install .
```

From within the mounted volume in the container, run:

**SCF Convergence**

```bash
python3 tb_lite/scc_convergence/converge_charges.py
```

**Energy vs Volume**

```bash
# Bulk systems
python3 tb_lite/energy_volume/energy_volume.py
# X23
python3 tb_lite/molecular_crystals/run_molecular_evsv.py
```

## <ins> Running Espresso Calculations </ins>

Check the Conda env is active, then from the package root run:

**SCF Convergence**

```shell
nohup python3 espresso/gs_convergence.py > outputs/espresso_scf/terminal_scf.out &
```

If a JSON file is not present, the script will generate one with all systems and run all systems. If a JSON file is 
already present, the script will only run systems listed as `Converged: False` or `Converged: 'Not Run'`,
whilst preserving all other entries in the file.

**Energy vs Volume**

```bash
python3 espresso/energy_volume.py
# or
nohup python3 espresso/energy_volume.py > outputs/espresso_e_vs_v/terminal.out &
```
