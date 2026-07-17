# BUILD: docker build -t calm-env .
# RUN: docker run --gpus all -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints calm-env
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3-pip python3-dev && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip && pip3 install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && pip3 install flax optax matplotlib scipy
WORKDIR /app
COPY . .
RUN mkdir -p /app/dashboard_static
CMD ["sh", "-c", "python3 meta.py & python3 model.py & python3 -m http.server 8000 --directory /app/dashboard_static"]
