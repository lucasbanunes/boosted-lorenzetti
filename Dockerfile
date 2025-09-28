FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get clean
# It takes a while to install this, implemented separately for caching
RUN apt-get install -y dvipng texlive-latex-extra texlive-fonts-recommended git && apt-get clean
# ADD https://install.duckdb.org /duckdb-installer.sh
# RUN sh /duckdb-installer.sh
# ENV PATH="/root/.duckdb/cli/latest:$PATH"


WORKDIR /root
COPY requirements.txt requirements.txt
# Root only supports conda or binary installation
# We install it with conda and the rest with pip
# to have the most recent packages
RUN conda init && \
    conda config --set channel_priority strict && \
    conda create -c conda-forge --name dev root==6.34.04 python==3.11.13 && \
    conda run -n dev --live-stream pip install -r requirements.txt && \
    rm requirements.txt


# FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04
# # ENV DEBIAN_FRONTEND=noninteractive

# RUN apt-get update && apt-get upgrade -y && apt-get clean
# # It takes a while to install this, implemented separately for caching
# RUN apt-get install -y dvipng texlive-latex-extra texlive-fonts-recommended && apt-get clean

# # Install uv
# # Download the latest installer
# ADD https://astral.sh/uv/install.sh /uv-installer.sh
# # Run the installer then remove it
# RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
#     sh /uv-installer.sh && rm /uv-installer.sh
# # Ensure the installed binary is on the `PATH`
# ENV PATH="/root/.local/bin/:$PATH"