import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# --- Label dan Khasiat Herbal ---
class_info = {
    "Belimbing Wuluh": "Menurunkan tekanan darah dan mengatasi batuk.",
    "Jambu Biji": "Mengandung antioksidan tinggi, bagus untuk kekebalan tubuh.",
    "Jeruk Nipis": "Membersihkan racun dalam tubuh dan menyegarkan tenggorokan.",
    "Kemangi": "Mengurangi stres dan menyehatkan pencernaan.",
    "Lidah Buaya": "Menyembuhkan luka dan mempercepat regenerasi kulit.",
    "Nangka": "Kaya serat dan vitamin C, baik untuk pencernaan.",
    "Pandan": "Mengharumkan makanan dan menenangkan saraf.",
    "Pepaya": "Melancarkan pencernaan dan memperbaiki kesehatan kulit.",
    "Seledri": "Menurunkan tekanan darah dan kolesterol.",
    "Sirih": "Antiseptik alami, baik untuk mulut dan tenggorokan."
}

class_names = list(class_info.keys())

st.set_page_config(page_title="Klasifikasi Daun Herbal", layout="centered")

# --- Load model hanya sekali ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/model_8_ft.h5")
    return model

model = load_model()

# --- UI Header ---
st.markdown("<h1 style='text-align: center; color: green;'>🌿 Klasifikasi Daun Herbal</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload gambar daun dan temukan jenis serta khasiatnya!</p>", unsafe_allow_html=True)

# --- Upload Gambar ---
uploaded_file = st.file_uploader("📤 Upload gambar daun...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Gambar yang diunggah", use_container_width=True)

    # --- Preprocessing untuk prediksi ---
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    # --- Prediksi ---
    predictions = model.predict(img_preprocessed)[0]
    top3_idx = predictions.argsort()[-3:][::-1]

    st.markdown("---")
    st.subheader("🔍 Hasil Prediksi:")

    for i, idx in enumerate(top3_idx):
        nama_daun = class_names[idx]
        confidence = predictions[idx]
        khasiat = class_info[nama_daun]

        st.markdown(f"""
        <div style='padding:10px; border-radius:10px; background-color:#f0f8f0; margin-bottom:10px'>
            <b>{i+1}. {nama_daun}</b> — <i>{confidence*100:.2f}%</i><br>
            <span style='font-size: 0.9em; color: #555;'>{khasiat}</span>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("Silakan unggah gambar daun untuk memulai klasifikasi.")
