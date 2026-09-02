# Clariden (CSCS Alps, GH200 = Grace ARM64 CPU + Hopper GPU) image for
# sdate and the sibling tomography/reconstruction projects it shares an
# environment with (astra-torch, smartt).
#
# Unlike the old renku/renkulab-cuda-based Dockerfile (docker/renku in
# chip-project), this targets arm64, not amd64 -- GH200's CPU is Grace, not
# x86_64, so the previous image cannot run here at all. Build this natively
# ON a Clariden login node (arm64 host, no cross-compilation/emulation
# needed) with podman, per https://docs.cscs.ch/software/ml/pytorch/:
#
#   cd docker/clariden
#   podman build -t sdate-clariden:0.1 .
#   enroot import -o $SCRATCH/ce-images/sdate-clariden+0.1.sqsh podman://sdate-clariden:0.1
#
# then reference the .sqsh in an EDF toml and `srun --environment=...`.
#
# Base: CSCS's own "Alps Extended" NGC PyTorch build -- arm64, CUDA 12.x,
# and already carries the Slingshot/NCCL (aws_ofi_nccl) tuning Alps needs.
# jfrog is the internal (fast-on-Alps) registry; swap to the ghcr.io mirror
# below if building from off-cluster.
#   ghcr.io/eth-cscs/alps-extended-images/ngc-pytorch:26.02-py3-alps6
FROM jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:26.02-py3-alps6

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
        vim git screen ncdu htop libgl1 \
        openssh-server \
        build-essential autoconf automake libtool pkg-config \
    && rm -rf /var/lib/apt/lists/*
# NOTE: libboost-all-dev was here initially (astra-toolbox's configure.ac
# probes for it) but it's only used for astra-toolbox's own optional unit
# tests, not the library/CUDA/Python build. Installing it pulled in a huge
# chain of Ubuntu openmpi/ucx-adjacent packages that shadowed the base
# image's /opt/hpcx HPC-X stack, breaking `import torch` at runtime with
# "libucc.so.1: undefined symbol: ucs_config_doc_nop" -- confirmed by
# removing it and re-testing.

# astra-toolbox has no aarch64 wheel on PyPI and is normally installed via
# conda-forge -- but the base image has no conda, and adding one just for
# this would mean two competing CUDA/Python stacks. Build from source
# against the base image's own CUDA toolkit + system Python instead.
# GH200's Hopper GPU is compute capability 9.0 (sm_90).
#
# Source is vendored as a tarball (astra-toolbox-v2.3.0.tar.gz, fetched from
# https://github.com/astra-toolbox/astra-toolbox/archive/refs/tags/v2.3.0.tar.gz)
# rather than `git clone`d at build time: a build-time clone of this exact
# repo failed on a Clariden compute node twice out of two attempts
# ("could not read Username for 'https://github.com'") -- intermittent
# network flakiness reaching github.com from compute nodes, not a real
# dependency on astra-toolbox itself. Vendoring removes that flakiness from
# the build entirely; re-run the curl command above to bump the version.
ARG CUDA_ARCH=90
COPY astra-toolbox-v2.3.0.tar.gz /opt/astra-toolbox.tar.gz
RUN mkdir /opt/astra-toolbox && \
    tar -xzf /opt/astra-toolbox.tar.gz -C /opt/astra-toolbox --strip-components=1 && \
    rm /opt/astra-toolbox.tar.gz && \
    cd /opt/astra-toolbox/build/linux && \
    ./autogen.sh && \
    ./configure --with-cuda=/usr/local/cuda --with-python --with-install-type=module \
        NVCCFLAGS="-arch=sm_${CUDA_ARCH}" && \
    make -j"$(nproc)" && \
    make install && \
    rm -rf /opt/astra-toolbox

# astra-torch: this project's thin PyTorch wrapper around astra-toolbox.
# Pure Python (no compiled extension of its own) -- installs on any arch
# as long as astra-toolbox above is importable. Vendored from the local
# checkout (see astra-torch/) rather than `git clone`d: a build-time clone
# from github.com failed on a Clariden compute node ("could not read
# Username for 'https://github.com'") even though the astra-toolbox clone
# a few steps earlier in the same build succeeded -- looked like transient
# network flakiness rather than a hard block, but vendoring sidesteps it
# entirely and also pins the exact source used, uncommitted local edits
# included.
COPY astra-torch /opt/astra-torch
RUN pip install --no-cache-dir -e /opt/astra-torch --no-deps

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

RUN cd /myhome/sdate 2>/dev/null || true

# SSH for interactive dev, matching the RunAI-image workflow. On Clariden
# this may be unnecessary/redundant with Slurm's own node access (srun --pty
# / ssh to the allocated compute node directly) -- keep for parity, drop if
# it turns out to just add noise.
RUN mkdir -p /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#*PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd && \
    sed -i 's/#*PasswordAuthentication .*/PasswordAuthentication yes/' /etc/ssh/sshd_config
ENV NOTVISIBLE="in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
EXPOSE 22

ENV XDG_CACHE_HOME=/myhome/.cache
ENV HOME=/myhome

CMD ["/usr/sbin/sshd", "-D"]
