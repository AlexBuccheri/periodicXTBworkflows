# Dockerfile for DFTB+ with TB lite library.
#
# * Build:
#   docker build -t dftbp:22.1-omp path/2/this/repo/.
#
# docker run -it --name tb_benchmarking_container -v /Users/alexanderbuccheri/Codes/tb_benchmarking:/tb_benchmarking dftbp:22.1-omp

# * Run:
#   docker run --entrypoint bash -it dftbp:22.1-omp
#    source /tb_benchmarking/docker_venv/bin/activate
#    cd /tb_benchmarking && pip install -e .
#
# Note, because this Dockerfile does not mount the volume or COPY the project
# one cannot run `pip install -e .` in the Dockerfile, and installing from
# recommendations.txt fails

FROM ubuntu:focal

# Stop python querying geographic area
ENV DEBIAN_FRONTEND=noninteractive

# GCC 10, python 3.8.10
RUN \
apt-get update && \
apt-get install -y software-properties-common build-essential git cmake pkg-config gfortran-10 \
python3 python3-distutils python3-pip python3-apt python-is-python3 \
libopenblas-base libopenblas-dev libopenblas-openmp-dev libopenblas0-openmp libopenblas-pthread-dev libopenblas0-pthread

# Install DFTB+ with TB Lite
RUN cd / && git clone --depth 1 --branch 22.1 https://github.com/dftbplus/dftbplus.git

# Compile threaded version (WITH_OMP On is default)
RUN \
cd dftbplus && mkdir build && cd build && \
FC=gfortran-10 CC=gcc CMAKE_PREFIX_PATH=/usr/lib/tblite cmake -DWITH_TBLITE=true -DCMAKE_INSTALL_PREFIX=/usr/lib/dftb+ ../ && \
make -j && \
make install

# Set ENV vars
ENV PATH="/usr/lib/dftb+/bin:$PATH"
ENV LIBRARY_PATH="/usr/lib/dftb+/lib/:$LIBRARY_PATH"
ENV LD_LIBRARY_PATH="/usr/lib/dftb+/lib/:$LD_LIBRARY_PATH"

# Install python packages in venv
RUN apt-get update && apt install python3.8-venv
RUN python -m venv /tb_benchmarking/docker_venv
RUN . /tb_benchmarking/docker_venv/bin/activate
RUN pip install --upgrade setuptools pip
# Generated with pip freeze > requirements.txt
# This runs in the container, but when I try to build the image
# pycairo fails (even with pymatgen removed)
#COPY requirements.txt requirements.txt
##Install all required packages
#RUN pip install -r requirements.txt
RUN cd /dftbplus/tools/dptools && pip install .
