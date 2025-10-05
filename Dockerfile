
# Utiliser une image Python avec support CUDA
FROM nvcr.io/nvidia/pytorch:24.08-py3

WORKDIR /app

# Copier le fichier requirements.txt
COPY requirements.txt .

# Installer les dépendances (sauf PyTorch + TorchVision + TorchAudio)
RUN python -m venv /app/venv && \
    /app/venv/bin/pip install --no-cache-dir -r requirements.txt

# Installer PyTorch + CUDA 12.1 depuis le canal officiel
# RUN pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Copier le code source
COPY . .

# Exposer le port
EXPOSE 9200

# Lancer l'application
CMD ["/app/venv/bin/uvicorn", "STT_output.py:app", "--host", "0.0.0.0", "--port", "9200"]