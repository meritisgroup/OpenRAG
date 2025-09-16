FROM python:3.11

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl bash libgl1 libglib2.0-0 \
    texlive-latex-base texlive-fonts-recommended texlive-fonts-extra \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt 

RUN apt-get update && apt-get install -y \
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-latex-extra \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m nltk.downloader punkt_tab

COPY ./.streamlit  ./.streamlit 
COPY ./backend ./backend
COPY ./data ./data
COPY ./storage ./storage
COPY ./streamlit_ ./streamlit_
COPY front.py .
COPY ./docker/.env .env

CMD ["streamlit", "run", "front.py"]
