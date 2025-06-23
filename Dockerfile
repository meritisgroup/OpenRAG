FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*
    RUN apt-get update && \
    apt-get install -y \
      curl \
      bash

WORKDIR /app

RUN apt-get update && apt install -y libgl1 libglib2.0-0
RUN apt-get update && \
    apt-get install -y texlive-latex-base \
                       texlive-fonts-recommended \
                       texlive-fonts-extra
COPY requirements.txt .
RUN pip3 install -r requirements.txt --break-system-packages

RUN python3 -c "import nltk; nltk.download('punkt_tab')"
COPY ./.streamlit  ./.streamlit 
COPY ./backend ./backend
COPY ./data ./data
COPY ./storage ./storage
COPY ./streamlit_ ./streamlit_
COPY front.py .
COPY ./docker/.env .env

CMD ["streamlit", "run", "front.py"]