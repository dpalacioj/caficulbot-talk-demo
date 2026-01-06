FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 as base

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/models \
    && mkdir -p /app/databases/cosecha \
    && mkdir -p /app/databases/gastos \
    && mkdir -p /app/databases/ingresos \
    && mkdir -p /app/databases/inventario

EXPOSE 8000 8001 8002 8003 8004 8501

# Script de inicio por defecto (puede ser sobrescrito)
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
