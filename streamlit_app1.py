"""
Application Streamlit pour la détection de déchets
Version simplifiée et robuste pour le déploiement
"""

import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os
import sys
from pathlib import Path
import time

# Configuration de la page
st.set_page_config(
    page_title="♻️ Détecteur de Déchets",
    page_icon="♻️",
    layout="centered"
)

# Titre
st.title("♻️ Détecteur Intelligent de Déchets")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Seuil de confiance
    confidence = st.slider(
        "Seuil de confiance",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    st.markdown("---")
    st.markdown("### 📋 Classes")
    st.markdown("• ♻️ **Recyclable** (plastique, verre, métal, papier)")
    st.markdown("• 🌱 **Organique** (restes alimentaires, végétaux)")
    st.markdown("• 🚯 **Non recyclable** (déchets mixtes, textiles sales)")
    
    st.markdown("---")
    st.markdown("### ℹ️ Info")
    st.markdown("Projet Computer Vision - L3 Big Data")
    st.markdown("Modèle: YOLOv8n")

# Chargement du modèle (avec cache et gestion d'erreurs)
@st.cache_resource
def load_model():
    """Charge le modèle YOLO avec gestion d'erreurs"""
    try:
        #from ultralytics import YOLO
        
        # Essayer de charger un modèle local, sinon utiliser le modèle par défaut
        model_paths = [
            "models/pretrained/yolov8n.pt",
            "yolov8n.pt"
        ]
        
        for path in model_paths:
            try:
                if os.path.exists(path):
                    model = YOLO(path)
                    st.sidebar.success(f"✅ Modèle chargé: {path}")
                    return model
            except:
                continue
        
        # Dernier recours : télécharger
        st.sidebar.info("📥 Téléchargement du modèle YOLOv8n...")
        model = YOLO("yolov8n.pt")
        return model
        
    except Exception as e:
        st.sidebar.error(f"❌ Erreur chargement modèle: {e}")
        return None

# Charger le modèle
with st.spinner("Chargement du modèle..."):
    model = load_model()

if model is None:
    st.error("""
    ❌ Impossible de charger le modèle.
    
    Vérifiez que les dépendances sont installées :
    ```
    pip install ultralytics opencv-python-headless
    ```
    """)
    st.stop()

# Interface principale
st.header("📸 Analyse d'image")

# Upload fichier
uploaded_file = st.file_uploader(
    "Choisissez une image...",
    type=['jpg', 'jpeg', 'png', 'webp']
)

# Image exemple (optionnel)
use_example = st.checkbox("Utiliser une image de test")

if uploaded_file is not None or use_example:
    
    # Charger l'image
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    else:
        # Créer une image de test
        image = Image.new('RGB', (640, 480), color='white')
        st.info("Image de test (aucune détection réelle)")
    
    # Afficher l'image originale
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Image originale")
        st.image(image, use_column_width=True)
    
    # Bouton de détection
    if st.button("🚀 Lancer la détection", type="primary"):
        
        with st.spinner("Analyse en cours..."):
            try:
                # Convertir PIL en numpy array
                img_array = np.array(image)
                
                # Prédiction
                results = model(img_array, conf=confidence)[0]
                
                # Compter les détections
                n_detections = len(results.boxes)
                
                # Afficher résultat
                with col2:
                    st.subheader("🎯 Résultat")
                    
                    if n_detections > 0:
                        # Afficher l'image avec détections
                        img_with_boxes = results.plot()
                        st.image(img_with_boxes, use_column_width=True)
                        
                        # Afficher les détails
                        st.success(f"✅ {n_detections} objet(s) détecté(s)")
                        
                        # Tableau des détections
                        detections_data = []
                        for box in results.boxes:
                            class_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            
                            classes = ['♻️ Recyclable', '🌱 Organique', '🚯 Non recyclable']
                            conseils = ['Poubelle jaune', 'Compost', 'Poubelle ordinaire']
                            
                            detections_data.append({
                                "Classe": classes[class_id] if class_id < len(classes) else "Inconnu",
                                "Confiance": f"{conf:.1%}",
                                "Conseil": conseils[class_id] if class_id < len(conseils) else "-"
                            })
                        
                        st.table(detections_data)
                        
                    else:
                        st.warning("⚠️ Aucun objet détecté")
                        with col2:
                            st.image(image, use_column_width=True)
                            
            except Exception as e:
                st.error(f"❌ Erreur lors de la détection: {e}")
                import traceback
                st.code(traceback.format_exc())

# Pied de page
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
    Projet réalisé dans le cadre du cours d'Initiation à la Computer Vision - L3 Big Data<br>
    © 2026
    </div>
    """,
    unsafe_allow_html=True

)
