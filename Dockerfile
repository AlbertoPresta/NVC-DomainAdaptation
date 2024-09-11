# Usa l'immagine base con CUDA e cudNN preinstallati
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04


# Copy source files and make it owned by the group eidoslab
# and give write permission to the group
COPY src /src
RUN chmod 775 /src
RUN chown -R :1337 /src

# Installa i pacchetti di sistema necessari
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Aggiorna pip
RUN python3 -m pip install --upgrade pip
RUN pip install pytorch_msssim
# Installa PyTorch con supporto CUDA
RUN pip install torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113

# Installa gli altri pacchetti Python
RUN pip install numpy>=1.20.0 scipy matplotlib tensorboard tqdm bd-metric ptflops






WORKDIR /src


ENTRYPOINT ["python3"]
