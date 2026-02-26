FROM python:3.11-slim

WORKDIR /app

COPY frontend/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY frontend/streamlit_ ./streamlit_
COPY frontend/front.py .
COPY frontend/.streamlit ./.streamlit

EXPOSE 8502

CMD ["streamlit", "run", "front.py", "--server.address", "0.0.0.0"]
