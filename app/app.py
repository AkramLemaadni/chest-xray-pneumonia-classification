import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==============================
# CONFIGURATION DE LA PAGE
# ==============================
st.set_page_config(
    page_title="Détection de la Pneumonie",
    page_icon="🫁",
    layout="centered"
)

# ==============================
# STYLE CSS
# ==============================
st.markdown("""
<style>
body {
    background-color: #0f172a;
}

.main {
    background-color: #0f172a;
    color: white;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}

.success {
    background-color: #064e3b;
    color: #a7f3d0;
}

.error {
    background-color: #7f1d1d;
    color: #fecaca;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# CHARGEMENT DU MODÈLE
# ==============================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("../model/mobilenetv2_pneumonia_model.keras")

model = load_model()

# ==============================
# TITRE
# ==============================
st.markdown("<div class='title'>🫁 Détection automatique de la Pneumonie</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Analyse intelligente des radiographies pulmonaires grâce à MobileNetV2</div>",
    unsafe_allow_html=True
)

st.divider()

# ==============================
# UPLOAD IMAGE
# ==============================
uploaded_file = st.file_uploader(
    "📤 Téléverser une radiographie pulmonaire",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # ==============================
    # AFFICHAGE IMAGE
    # ==============================
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Radiographie chargée", use_container_width=True)

    # ==============================
    # PRÉTRAITEMENT
    # ==============================
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # ==============================
    # PRÉDICTION
    # ==============================
    prediction = model.predict(img_array)[0][0]

    # ==============================
    # INTERPRÉTATION CORRECTE
    # ==============================
    if prediction >= 0.5:
        label = "Pneumonie détectée"
        confidence = prediction
        css_class = "error"
        icon = "❌"
    else:
        label = "Poumon sain"
        confidence = 1 - prediction
        css_class = "success"
        icon = "✅"

    st.divider()

    # ==============================
    # AFFICHAGE RÉSULTAT
    # ==============================
    st.markdown(
        f"""
        <div class='result-box {css_class}'>
            {icon} {label}<br>
            Confiance : {confidence*100:.2f}%
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==============================
    # BARRE DE CONFIANCE
    # ==============================
    st.progress(float(confidence))

    st.caption("⚠️ Cette application est un outil d'aide à la décision et ne remplace pas un avis médical.")
