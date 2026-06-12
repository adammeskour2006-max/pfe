import streamlit as st
import librosa
import numpy as np
import joblib

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Diagnostic Acoustique Moteur", page_icon="⚙️", layout="centered")

st.title("⚙️ Outil de Maintenance Prédictive Acoustique")
st.markdown("""
Cet outil analyse la signature acoustique d'un moteur pour détecter d'éventuelles anomalies mécaniques.
""")

# --- FONCTION D'EXTRACTION (Cachée en arrière-plan) ---
@st.cache_data
def extraire_mfcc(chemin_audio):
    # Chargement du fichier audio uploadé
    y, sr = librosa.load(chemin_audio, sr=22050)
    
    # Prétraitement : Coupe les silences, accentue les aigus, normalise
    y_trim, _ = librosa.effects.trim(y)
    y_preemph = librosa.effects.preemphasis(y_trim)
    y_norm = librosa.util.normalize(y_preemph)
    
    # Extraction MFCC (13 coefficients)
    mfccs = librosa.feature.mfcc(y=y_norm, sr=sr, n_mfcc=13)
    signature = np.mean(mfccs.T, axis=0)
    
    return signature

# --- INTERFACE UTILISATEUR ---
st.subheader("1. Importation du Modèle IA")
fichier_modele = st.file_uploader("Uploadez le fichier du modèle (.joblib)", type=["joblib"])

st.subheader("2. Analyse de l'Enregistrement")
fichier_audio = st.file_uploader("Uploadez l'enregistrement du moteur (.wav)", type=["wav"])

# --- LOGIQUE DE DIAGNOSTIC ---
if fichier_modele is not None and fichier_audio is not None:
    
    # Bouton d'action
    if st.button("Lancer le Diagnostic Acoustique", type="primary"):
        with st.spinner("Analyse des harmoniques en cours..."):
            
            try:
                # 1. Chargement du modèle
                modele = joblib.load(fichier_modele)
                
                # 2. Traitement du son
                # Streamlit traite les fichiers uploadés comme des objets en mémoire, 
                # librosa peut les lire directement.
                signature = extraire_mfcc(fichier_audio)
                signature_reshape = signature.reshape(1, -1)
                
                # 3. Prédiction et calcul du score
                prediction = modele.predict(signature_reshape)
                score = modele.decision_function(signature_reshape)[0]
                
                # 4. Affichage des résultats
                st.markdown("---")
                st.subheader("Résultat du Diagnostic")
                
                # Mise en page en colonnes
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(label="Score d'Anomalie", value=f"{score:.3f}")
                    st.caption("Un score négatif indique une déviation de la norme.")
                
                with col2:
                    if prediction[0] == 1:
                        st.success("🟢 STATUT : MOTEUR SAIN")
                        st.write("La signature acoustique correspond à la norme statistique apprise.")
                    else:
                        st.error("🔴 STATUT : ANOMALIE DÉTECTÉE")
                        st.write("Une déviation anormale a été repérée (possible défaut d'injection, jeu de soupape, etc.).")
                        
                # Optionnel : Afficher un lecteur audio pour réécouter le fichier sur le tableau de bord
                st.audio(fichier_audio, format='audio/wav')
                
            except Exception as e:
                st.error(f"Une erreur s'est produite lors de l'analyse : {e}")

else:
    st.info("Veuillez uploader le modèle et un fichier audio pour commencer.")
