import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf

app = Flask(__name__)

# Path ke model dan direktori unggahan
MODEL_PATH = 'model/Final_CekTandur.h5'
UPLOAD_FOLDER = 'uploads'

# Pastikan direktori untuk menyimpan unggahan ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Mapping kelas langsung dari class_indices
class_indices = {
    0: 'Anggur__Bercak_daun_isariopsis',
    1: 'Anggur__Esca(campak_hitam)',
    2: 'Anggur__Hitam_busuk',
    3: 'Anggur__Sehat',
    4: 'Apel__Busuk_hitam',
    5: 'Apel__Karat_apel_cedar',
    6: 'Apel__Keropeng_apel',
    7: 'Apel__Sehat',
    8: 'Jagung__Bercak_daun_abu-abu',
    9: 'Jagung__Busuk_daun',
    10: 'Jagung__Karat_umum',
    11: 'Jagung__Sehat',
    12: 'Kentang__Busuk_daun_dini',
    13: 'Kentang__Busuk_daun_telat',
    14: 'Kentang__Sehat',
    15: 'Tomat__Bercak_bakteri',
    16: 'Tomat__Bercak_daun',
    17: 'Tomat__Bercak_target',
    18: 'Tomat__Busuk_daun_dini',
    19: 'Tomat__Busuk_daun_telat',
    20: 'Tomat__Daun_keriting_kuning',
    21: 'Tomat__Jamur_septoria_lycopersici',
    22: 'Tomat__Sehat',
    23: 'Tomat__Tungau_laba-laba_Berbintik',
    24: 'Tomat__Virus_mosaik_tomat'
}

# Fungsi untuk preprocessing gambar
def preprocess_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'images' not in request.files:
        return redirect(request.url)

    files = request.files.getlist('images')
    results = []

    for file in files:
        if file.filename == '':
            continue

        # Simpan gambar yang diunggah
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Preprocess dan prediksi
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0] if prediction.ndim > 1 else np.argmax(prediction)
        predicted_label = class_indices[predicted_class]
        probability = prediction[0][predicted_class]

        # Simpan hasil prediksi
        results.append((file.filename, predicted_label, probability * 100.0))

    return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
