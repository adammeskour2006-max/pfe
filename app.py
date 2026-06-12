import streamlit as st
import librosa
import librosa.display
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION DE L'APPLICATION
# ==========================================
st.set_page_config(
    page_title="DiagMoteur Pro | Maintenance Predictive",
    page_icon="⚙️",
    layout="wide", # Passage en mode large pour les tableaux de bord professionnels
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. FONCTIONS MÉTIER (BACKEND LOGIC)
# ==========================================
@st.cache_resource(show_spinner=False)
def charger_modele(chemin_fichier):
    """Charge le modèle ML en cache pour optimiser les performances."""
    return joblib.load(chemin_fichier)

@st.cache_data(show_spinner=False)
def extraire_mfcc_et_signal(chemin_audio):
    """Extrait la signature MFCC et retourne aussi le signal pour la visualisation."""
    y, sr = librosa.load(chemin_audio, sr=22050)
    
    y_trim, _ = librosa.effects.trim(y)
    y_preemph = librosa.effects.preemphasis(y_trim)
    y_norm = librosa.util.normalize(y_preemph)
    
    mfccs = librosa.feature.mfcc(y=y_norm, sr=sr, n_mfcc=13)
    signature = np.mean(mfccs.T, axis=0)
    
    return signature, y_norm, sr

def tracer_waveform(y, sr):
    """Génère un graphique professionnel du signal audio."""
    fig, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax, color="#1f77b4", alpha=0.8)
    ax.set_title("Empreinte Acoustique du Moteur", fontsize=10, loc='left', color='#555555')
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Amplitude")
    ax.axis('off') # Style épuré sans bordures lourdes
    plt.tight_layout()
    return fig

# ==========================================
# 3. INTERFACE UTILISATEUR (FRONTEND)
# ==========================================
def main():
    # --- EN-TÊTE PRINCIPAL ---
    st.title("⚙️ Centre de Diagnostic Acoustique")
    st.markdown("Système d'analyse par **Isolation Forest** pour la détection non-invasive d'anomalies mécaniques.")
    st.divider()

    # --- BARRE LATÉRALE (PARAMÈTRES & UPLOADS) ---
    with st.sidebar:
        st.header("🛠️ Configuration")
        st.markdown("Veuillez charger les fichiers requis pour lancer l'analyse.")
        
        fichier_modele = st.file_uploader("1. Modèle d'IA (.joblib)", type=["joblib"], help="Chargez le cerveau pré-entraîné.")
        fichier_audio = st.file_uploader("2. Enregistrement Moteur (.wav)", type=["wav"], help="Chargez l'échantillon capturé sur le terrain.")
        
        st.markdown("---")
        st.caption("Mode : Maintenance Prédictive v1.2")
        st.caption("Département : Ingénierie Mécanique & Production")

    # --- ZONE PRINCIPALE DE DIAGNOSTIC ---
    if fichier_modele and fichier_audio:
        st.info("Fichiers chargés avec succès. Prêt pour l'analyse.", icon="✅")
        
        # Bouton d'action principal bien en évidence
        if st.button("🚀 Lancer le Diagnostic Mécanique", use_container_width=True, type="primary"):
            
            with st.spinner("Analyse spectrale et inférence IA en cours..."):
                try:
                    # Traitement backend
                    modele = charger_modele(fichier_modele)
                    signature, y_norm, sr = extraire_mfcc_et_signal(fichier_audio)
                    signature_reshape = signature.reshape(1, -1)
                    
                    prediction = modele.predict(signature_reshape)
                    score = modele.decision_function(signature_reshape)[0]
                    
                    # --- AFFICHAGE DES RÉSULTATS (TABS) ---
                    tab1, tab2 = st.tabs(["📊 Tableau de Bord Principal", "🔬 Détails Techniques"])
                    
                    # ONGLET 1 : Vue pour le technicien (Décision rapide)
                    with tab1:
                        st.subheader("Statut du Diagnostic")
                        
                        col1, col2, col3 = st.columns([1, 2, 1])
                        
                        with col1:
                            st.metric(label="Score d'Anomalie (IF)", value=f"{score:.4f}", delta="Tolérance: 0.00" if score > 0 else "Alerte Franche", delta_color="normal" if score > 0 else "inverse")
                            
                        with col2:
                            if prediction[0] == 1:
                                st.success("### 🟢 MOTEUR SAIN\nLa signature acoustique est conforme aux tolérances constructeur.")
                            else:
                                st.error("### 🔴 ANOMALIE DÉTECTÉE\nDéviation acoustique majeure identifiée. Inspection mécanique requise.")
                                
                        with col3:
                            st.audio(fichier_audio, format='audio/wav')
                            st.caption("Réécouter l'échantillon")
                            
                        st.markdown("---")
                        # Affichage du graphique généré
                        fig = tracer_waveform(y_norm, sr)
                        st.pyplot(fig)

                    # ONGLET 2 : Vue pour l'ingénieur (Data Brute)
                    with tab2:
                        st.subheader("Vecteur MFCC (13 Coefficients)")
                        st.markdown("Valeurs extraites pour l'inférence de l'algorithme :")
                        # Affichage élégant des données brutes
                        st.dataframe(np.round(signature.reshape(1, 13), 4), use_container_width=True)
                        
                        with st.expander("Voir l'explication du fonctionnement"):
                            st.write("""
                            L'algorithme **Isolation Forest** a analysé le vecteur de coefficients mel-fréquentiels (MFCC) ci-dessus. 
                            Un score d'anomalie positif indique que la donnée se situe au cœur de la distribution des moteurs sains (profil normal). 
                            Un score négatif indique une observation rare ou aberrante (profil en panne ou bruité).
                            """)

                except Exception as e:
                    st.error(f"Une erreur technique est survenue lors de l'exécution : {e}")
                    st.stop()
    else:
        # Écran d'accueil quand rien n'est chargé
        st.warning("👈 En attente des données. Veuillez charger le modèle et un fichier audio via le panneau latéral.")
        
        # Placeholder visuel professionnel
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Étape 1 :** Chargez votre modèle entraîné `.joblib`.")
        with col2:
            st.info("**Étape 2 :** Chargez le fichier `.wav` capturé sur le terrain.")

# Point d'entrée standard en Python
if __name__ == "__main__":
    main()
