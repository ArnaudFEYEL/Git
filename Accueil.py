import streamlit as st

st.set_page_config(
    page_title="🏠 Accueil",
    page_icon="🏠",
)


# Defines title of the main page 
st.markdown("<h1 style='text-align: center; color: white;'>Neural ODEs, Normalizing Flows et Latent ODEs</h1>", unsafe_allow_html=True)

# Defines introduction to the website
st.write("Vous vous trouvez sur la page 🏠 Accueil de notre projet de recherche en Deep Learning.")
st.write("Le site est découpé en plusieurs pages décrites ci-dessous.")
    
st.markdown("""
    - 🏠 Accueil
    - 📕 Rapport du Projet : vous y trouverez le rapport écrit de notre projet de recherche. Disponible en visualisation sur le site et en pdf.
    - 🖼 Diapositives : vous trouverez les diapositives qui accompagnent le Rapport.
    - 🔬 Piste de réflexions : vous y trouverez des tentatives d'approfondissement de notre projets, soulignant les limites et améliorations des modèles
    """)

st.markdown("## Contact")
st.markdown("- Arnaud FEYEL: arnaud.feyel@universite-paris-saclay.fr")
st.markdown("- Malek BOUZIDI: malek.bouzidi@universite-paris-saclay.fr")
st.markdown("- Noor SEMAAN: noor.semaan@universite-paris-saclay.fr")

