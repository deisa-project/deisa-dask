FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential openmpi-bin libopenmpi-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

ENV UCX_TLS=tcp,self
ENV FI_PROVIDER=tcp

RUN pip install --upgrade pip \
 && pip install --no-cache-dir mpi4py \
 && pip install dask==2026.3.0 distributed==2026.3.0 \
 && pip install . \
 && pip install .[benchmark]
