import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from rembg import remove
from PIL import Image
import io
import cv2


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

# --- Load model sekali ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/model_8_ft.h5")  # .h5 atau .keras
    return model

model = load_model()

# --- UI Header ---
# st.set_page_config(page_title="Klasifikasi Daun Herbal", layout="centered")
st.markdown("<h1 style='text-align: center; color: green;'>🌿 Klasifikasi Daun Herbal</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload gambar daun dan temukan jenis serta khasiatnya!</p>", unsafe_allow_html=True)

# --- File Upload ---
uploaded_file = st.file_uploader("📤 Upload gambar daun...", type=["jpg", "png", "jpeg"])

def remove_bg(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_no_bg_bytes = remove(img_byte_arr.getvalue())  # returns bytes
    return Image.open(io.BytesIO(img_no_bg_bytes))

run = st.checkbox('Nyalakan Kamera')

# Menampilkan frame video
FRAME_WINDOW = st.image([])

# Mengakses webcam (0 = webcam default)
camera = cv2.VideoCapture(0)
captured_image = None

while run:
    ret, frame = camera.read()
    if not ret:
        st.warning("Gagal mengakses kamera")
        break

    # Konversi BGR ke RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Tampilkan frame real-time
    FRAME_WINDOW.image(frame_rgb)

    # Tombol untuk ambil gambar
    if st.button("📸 Jepret"):
        captured_image = Image.fromarray(frame_rgb)
        st.success("✅ Gambar berhasil diambil.")
        break  # Keluar dari loop agar tidak update terus

# Stop video saat checkbox dimatikan
camera.release()

if captured_image is not None:
    st.image(captured_image, caption="🖼️ Gambar Hasil Jepretan", use_container_width=True)

if uploaded_file is not None or captured_image is not None:
    # Tampilkan gambar
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
    elif captured_image is not None:
        image = captured_image
    else:
        image = None
        
    st.image(image, caption="Gambar yang diproses", use_container_width=True)

    # image = Image.open(uploaded_file).convert("RGB")
    image_no_bg = remove_bg(image).convert("RGB")
    st.image(image, caption="Gambar tanpa latar belakang", use_container_width=True)

    # st.image(image, caption="🖼️ Gambar yang diunggah", use_container_width=True)

    # Preprocessing
    img = image.resize((224, 224))
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
    st.info("Silakan unggah gambar daun untuk memulai klasifikasi.")