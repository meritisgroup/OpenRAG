FROM ollama/ollama:0.6.2

COPY ./entrypoint.sh /entrypoint.sh
COPY ./gemma9b_Modelfile /gemma9b_Modelfile
RUN apt-get update && apt-get install -y dos2unix
RUN dos2unix entrypoint.sh
RUN chmod +x /entrypoint.sh && ls -la /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]