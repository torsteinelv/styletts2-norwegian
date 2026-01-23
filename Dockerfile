FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 1. Installer system-avhengigheter (Viktig: espeak-ng for norsk fonetikk)
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

# 3. Klon StyleTTS2 (Vi bruker den som "motor")
RUN git clone https://github.com/yl4579/StyleTTS2.git .

# 4. Installer Python-biblioteker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- VIKTIG NYTT STEG ---
# 4b. Kompiler monotonic_align (Fix for ModuleNotFoundError)
# Dette må gjøres etter pip install, da det krever Cython.
RUN cd monotonic_align && python setup.py build_ext --inplace

# 5. Kopier vår kildekode inn i containeren
COPY src/ ./src/
COPY configs/ ./configs/

# 6. Lag mapper for data og output
RUN mkdir -p Data/wavs Models

# Standard kommando (kan overstyres)
CMD ["python", "src/prepare_dataset.py"]
