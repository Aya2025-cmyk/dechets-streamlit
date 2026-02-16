#!/bin/bash

echo "🚀 Installation des dépendances système..."
apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1

echo "✅ Dépendances système installées"

# Vérifier l'installation
python -c "import ultralytics; print('✅ Ultralytics installé')" || echo "❌ Ultralytics non installé"
python -c "import cv2; print('✅ OpenCV installé')" || echo "❌ OpenCV non installé"
