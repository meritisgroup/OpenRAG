FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./scripts/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install accelerate
COPY ./scripts .

EXPOSE 8000

# Définir la commande de démarrage
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]