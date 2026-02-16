"""
Application Streamlit pour la détection de déchets
Version avec modèle personnalisé entraîné
"""

import streamlit as st
import numpy as np
from PIL import Image
import os
from pathlib import Path

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
    st.markdown("Modèle: **Personnalisé - Entraîné sur dataset de déchets**")

# Chargement du modèle (avec cache)
@st.cache_resource
def load_model():
    """Charge le modèle YOLO personnalisé"""
    try:
        from ultralytics import YOLO
        
        # PRIORITÉ À VOTRE MODÈLE ENTRAÎNÉ
        model_paths = [
            "models/custom/waste_detector.pt",      # VOTRE MODÈLE ENTRAÎNÉ
            "runs/detect/waste_detection/weights/last.pt",
            "runs/detect/waste_detection/weights/best.pt",
            "models/pretrained/yolov8n.pt",
            "yolov8n.pt"
        ]
        
        for path in model_paths:
            try:
                if os.path.exists(path):
                    model = YOLO(path)
                    st.sidebar.success(f"✅ Modèle chargé: {os.path.basename(path)}")
                    
                    # Afficher le nombre de classes
                    if hasattr(model, 'names'):
                        st.sidebar.info(f"📊 {len(model.names)} classes détectables")
                    return model
            except Exception as e:
                st.sidebar.warning(f"⚠️ Impossible de charger {path}: {e}")
                continue
        
        # Dernier recours
        st.sidebar.info("📥 Téléchargement du modèle par défaut...")
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
    
    Vérifiez que :
    1. Les dépendances sont installées
    2. Le fichier de modèle existe dans `models/custom/`
    
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

if uploaded_file is not None:
    
    # Charger l'image
    image = Image.open(uploaded_file)
    
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
                        
                        # Récupérer les noms des classes du modèle
                        class_names = model.names if hasattr(model, 'names') else {0: 'recyclable', 1: 'organic', 2: 'non_recyclable'}
                        
                        # Tableau des détections
                        detections_data = []
                        for i, box in enumerate(results.boxes):
                            class_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            
                            # Mapping des classes avec emojis et conseils
                            class_mapping = {
                                0: {"name": "♻️ Recyclable", "conseil": "Poubelle jaune"},
                                1: {"name": "🌱 Organique", "conseil": "Compost"},
                                2: {"name": "🚯 Non recyclable", "conseil": "Poubelle ordinaire"}
                            }
                            
                            info = class_mapping.get(class_id, {"name": f"Classe {class_id}", "conseil": "À vérifier"})
                            
                            detections_data.append({
                                "Objet": f"{i+1}",
                                "Classe": info["name"],
                                "Confiance": f"{conf:.1%}",
                                "Conseil": info["conseil"]
                            })
                        
                        st.table(detections_data)
                        
                        # Statistiques rapides
                        recyclable = sum(1 for box in results.boxes if int(box.cls[0]) == 0)
                        organic = sum(1 for box in results.boxes if int(box.cls[0]) == 1)
                        non_recyclable = sum(1 for box in results.boxes if int(box.cls[0]) == 2)
                        
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("♻️ Recyclable", recyclable)
                        col_b.metric("🌱 Organique", organic)
                        col_c.metric("🚯 Non recyclable", non_recyclable)
                        
                    else:
                        st.warning("⚠️ Aucun objet détecté")
                        with col2:
                            st.image(image, use_column_width=True)
                            
            except Exception as e:
                st.error(f"❌ Erreur lors de la détection: {e}")
                import traceback
                with st.expander("Détails de l'erreur"):
                    st.code(traceback.format_exc())

# Pied de page
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
    Projet réalisé dans le cadre du cours d'Initiation à la Computer Vision - L3 Big Data<br>
    © 2026 - Modèle entraîné sur dataset personnalisé
    </div>
    """,
    unsafe_allow_html=True
)
