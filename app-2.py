import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from rembg import remove
import io
import cv2

# Label dan Khasiat Herbal
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
st.markdown("<h1 style='text-align: center; color: green;'>🌿 Klasifikasi Daun Herbal</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Gunakan kamera atau upload gambar daun untuk klasifikasi.</p>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/model_8_ft.h5")

model = load_model()

def remove_bg(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_no_bg_bytes = remove(img_byte_arr.getvalue())
    return Image.open(io.BytesIO(img_no_bg_bytes))

# Pilih metode input
method = st.radio("Pilih metode input gambar:", ["📷 Kamera", "📁 Upload Gambar"])

image = None

if method == "📷 Kamera":
    run = st.checkbox('Nyalakan Kamera')

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        st.error("❌ Kamera tidak tersedia. Silakan gunakan opsi upload gambar.")
        run = False

    while run:
        ret, frame = camera.read()
        if not ret:
            st.warning("Gagal membaca kamera.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

        capture = st.button("📸 Ambil Gambar")
        if capture:
            image = Image.fromarray(frame)
            run = False  # stop camera loop

    camera.release()

elif method == "📁 Upload Gambar":
    uploaded_file = st.file_uploader("Upload gambar daun", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

# Proses jika ada gambar yang tersedia
if image:
    image_no_bg = remove_bg(image).convert("RGB")
    st.image(image_no_bg, caption="🖼️ Gambar tanpa latar belakang", use_container_width=True)

    # Preprocessing
    img = image_no_bg.resize((224, 224))
    img_array = np.array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)

    # Prediksi
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
    st.info("Silakan nyalakan kamera dan ambil gambar, atau upload gambar daun.")
