FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libgl1 libglib2.0-0 \
    texlive-latex-base texlive-latex-extra texlive-pictures \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app/src

COPY backend/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN python3 -m nltk.downloader punkt_tab

COPY backend/ .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
