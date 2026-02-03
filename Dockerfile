FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

# 1. Installer system-avhengigheter
RUN apt-get update && apt-get install -y \
    git \
    libsndfile1 \
    espeak-ng \
    build-essential \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 2. Sett arbeidsmappe
WORKDIR /app

# 3. Klon StyleTTS2
RUN git clone https://github.com/yl4579/StyleTTS2.git .

# 4. Installer Python-biblioteker fra requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- RETTET STEG ---
# 5. Installer monotonic_align direkte fra GitHub
# Dette fikser "ModuleNotFoundError" og "can't cd to monotonic_align"
RUN pip install git+https://github.com/resemble-ai/monotonic_align.git

# 6. Kopier vår kildekode inn i containeren
COPY src/ ./src/
COPY configs/ ./configs/

# 7. Lag mapper for data og output
RUN mkdir -p Data/wavs Models

# Standard kommando
CMD ["python", "src/prepare_dataset.py"]
