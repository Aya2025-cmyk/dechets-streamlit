#!/bin/bash
# Script de configuration pour Streamlit Cloud

echo "🚀 Configuration de l'environnement Streamlit..."

# Installation des dépendances système
apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

# Installation des dépendances Python
pip install -r requirements-streamlit.txt

# Création des dossiers nécessaires
mkdir -p models/pretrained
mkdir -p data/samples/images
mkdir -p data/outputs

# Téléchargement du modèle pré-entraîné si nécessaire
python -c "
from ultralytics import YOLO
print('📥 Téléchargement du modèle YOLOv8...')
model = YOLO('yolov8n.pt')
model.save('models/pretrained/yolov8n.pt')
print('✅ Modèle téléchargé avec succès!')
"

echo "✅ Configuration terminée!"