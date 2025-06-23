#Démarrage des dockers
cd ..
cd infrastructure

# Démarrer le service Elasticsearch
echo "Démarrage du service Elasticsearch..."
cd elasticsearch
chown -R 1000:1000 ../../volumes/dev-elasticsearch
docker compose up -d
cd ..

# Démarrer le service Ollama
echo "Démarrage du service Ollama..."
cd ollama
docker compose up -d --build
cd ..
cd ..
