# Run the following commands in order:
#
# TFTEXT_DIR="/tmp/tftext"  # (change to the cloned tftext directory, e.g. "$HOME/tftext")
# TFTEXT_DEVICE="gpu"  # (Leave empty to build and run CPU only docker)
# docker build --tag tensorflow:tftext $(test "$TFTEXT_DEVICE" = "gpu" && echo "--build-arg base_image=nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04") - < tftext/docker/dev.dockerfile
# docker run --rm $(test "$TFTEXT_DEVICE" = "gpu" && echo "--runtime=nvidia") -it -v ${TFTEXT_DIR}:/tmp/tftext -v ${HOME}/.gitconfig:/home/${USER}/.gitconfig:ro -p 6006:6006 -p 8888:8888 --name tftext tensorflow:tftext bash

# TODO(drpng): upgrade to latest (18.04)
ARG cpu_base_image="ubuntu:16.04"
ARG base_image=$cpu_base_image
FROM $base_image

LABEL maintainer="Feng Wang <wangfelix87@gmail.com>"

# Re-declare args because the args declared before FROM can't be used in any
# instruction after a FROM.
ARG cpu_base_image="ubuntu:16.04"
ARG base_image=$cpu_base_image


## Update system
RUN apt-get update

# Install extras needed by most models
RUN apt-get install -y --no-install-recommends \
      git \
      build-essential \
      software-properties-common \
      gcc-4.8 g++-4.8 gcc-4.8-base \
      aria2 \
      wget \
      curl \
      htop \
      zip \
      vim \
      unzip

# Install / update Python and Python3
RUN apt-get install -y --no-install-recommends \
      python3 \
      python3-dev \
      python3-pip \
      python3-setuptools \
      python3-venv

RUN apt-get clean && \
      rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 100 &&\
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 100


# Setup Python3 environment
RUN pip3 install --upgrade pip==19.0.3
# setuptools upgraded to fix install requirements from model garden.
RUN pip3 install --upgrade setuptools
RUN pip3 install wheel absl-py

RUN pip3 --no-cache-dir install \
        Pillow \
        h5py \
        ipykernel \
        jupyter \
        jupyter_http_over_ws \
        matplotlib \
        numpy \
        pandas \
        recommonmark \
        scipy \
        sklearn \
        sphinx \
        sphinx_rtd_theme \
        && \
    python3 -m ipykernel.kernelspec

RUN jupyter serverextension enable --py jupyter_http_over_ws

# The latest tf-nightly-gpu requires CUDA 10 compatible nvidia drivers (410.xx).
# If you are unable to update your drivers, an alternative is to compile
# TensorFlow from source instead of installing from pip.
RUN pip3 install tf-nightly$(test "$base_image" != "$cpu_base_image" && echo "-gpu")


ARG bazel_version=0.17.2
# This is to install bazel, for development purposes.
ENV BAZEL_VERSION ${bazel_version}
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh


# TensorBoard
EXPOSE 6006

# Jupyter
EXPOSE 8888

WORKDIR "/tmp/tftext"

CMD ["/bin/bash"]