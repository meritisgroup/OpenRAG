#!/bin/bash

echo "Starting Ollama server..."
ollama serve & SERVE_PID=$!


echo "Waiting for Ollama server to be active..."
while [ "$(ollama list | grep 'NAME')" == "" ]; do
  sleep 1
done

ollama create -f gemma9b_Modelfile gemma2:9b
ollama pull gemma3:1b
ollama pull mxbai-embed-large:latest
ollama pull bge-m3:latest

wait $SERVE_PID