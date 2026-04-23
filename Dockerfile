FROM ubuntu:22.04

WORKDIR /app


# Install base utilities
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
/bin/bash ~/miniconda.sh -b -p /opt/conda && \
rm ~/miniconda.sh

# Put conda in path
ENV PATH=/opt/conda/bin:$PATH

# Copy only the env file first (better layer caching)
COPY environment.yml .

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
# Create the environment
RUN conda env create -f environment.yml && conda clean -afy

# Make the conda env the default for all subsequent commands
SHELL ["conda", "run", "-n", "auto_avsr", "/bin/bash", "-c"]
ENV PATH=/opt/conda/envs/auto_avsr/bin:$PATH

# Now copy the rest of the code
COPY . .

EXPOSE 8000

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "auto_avsr", "python", "train.py"]
