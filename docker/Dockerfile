FROM nvidia/cuda:9.0-cudnn7-runtime

# Install some utilities
RUN apt-get update && \
    apt-get install -y -q wget git libxrender1 libsm6 bzip2 && \
    apt-get clean

# Install miniconda
RUN MINICONDA="Miniconda3-4.2.12-Linux-x86_64.sh" && \
    wget --quiet https://repo.continuum.io/miniconda/$MINICONDA && \
    bash $MINICONDA -b -p /miniconda && \
    rm -f $MINICONDA
ENV PATH /miniconda/bin:$PATH

RUN conda install -y -q -c rdkit boost rdkit
RUN conda install -y -q -c omnia fftw3f mdtraj openmm pdbfixer 
RUN conda install -y -q -c deepchem mdtraj simdna
RUN conda install -y -q -c conda-forge nose-timer xgboost
RUN conda install -y -q joblib numpy networkx flaky zlib requests pbr biopython

RUN pip install --upgrade pip
RUN pip install tensorflow-gpu==1.14.0

RUN git clone https://github.com/deepchem/deepchem.git

RUN cd deepchem && python setup.py develop && git clean -fX
