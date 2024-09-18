import os
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from PIL import Image


# app = Flask(__name__)

# Folder untuk menyimpan file yang diunggah sementara
app = Flask(__name__, template_folder='templates', static_folder='assets')
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Memastikan folder upload ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Fungsi untuk menambahkan saluran alfa ke gambar
def add_alpha_channel(image):
    b, g, r = cv2.split(image)
    alpha = np.ones(b.shape, dtype=b.dtype) * 255  # Saluran alfa penuh (tidak transparan)
    return cv2.merge((b, g, r, alpha))

# Fungsi untuk mengubah gambar menjadi grayscale dan menambahkan saluran alfa
def convert_to_grayscale_with_alpha(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    alpha = np.ones(gray.shape, dtype=gray.dtype) * 255  # Saluran alfa penuh
    return cv2.merge((gray, gray, gray, alpha))

# Halaman utama
@app.route('/')
def index():
    return render_template('prediksi.html')

@app.route('/reset')
def reset():
    """Clears uploaded files and prediction results."""
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    return render_template('prediksi.html')

@app.route('/download_sample/<filename>')
def download_sample(filename):
    sample_folder = 'static/sample/'
    return send_from_directory(sample_folder, filename)


@app.route('/upload', methods=['POST'])
def upload():
    fundus_file = request.files.get('fundus')
    oct_file = request.files.get('oct')
    
    fundus_sample = request.form.get('fundus_sample')
    oct_sample = request.form.get('oct_sample')

    if fundus_sample:
        fundus_file_path = os.path.join('assets/img', fundus_sample)
    elif fundus_file and fundus_file.filename != '':
        fundus_filename = secure_filename(fundus_file.filename)
        fundus_file_path = os.path.join(app.config['UPLOAD_FOLDER'], fundus_filename)
        fundus_file.save(fundus_file_path)
    else:
        return redirect(request.url)

    # Handle OCT sample or file upload
    if oct_sample:
        oct_file_path = os.path.join('assets', 'img', oct_sample)
    elif oct_file and oct_file.filename != '':
        oct_filename = secure_filename(oct_file.filename)
        oct_file_path = os.path.join(app.config['UPLOAD_FOLDER'], oct_filename)
        oct_file.save(oct_file_path)
    else:
        return redirect(request.url)

    # Baca gambar dengan OpenCV (sample atau unggahan)
    fundus_img = cv2.imread(fundus_file_path)
    oct_img = cv2.imread(oct_file_path)

    # Resize gambar ke 224x224
    fundus_resized = cv2.resize(fundus_img, (224, 224))
    oct_resized = cv2.resize(oct_img, (224, 224))

    # Konversi gambar ke grayscale dengan saluran alfa
    fundus_gray_with_alpha = convert_to_grayscale_with_alpha(fundus_resized)
    oct_gray_with_alpha = convert_to_grayscale_with_alpha(oct_resized)

    # Pisahkan gambar menjadi bagian kiri dan kanan
    width_to_overlay = 224
    fundus_left_with_alpha = fundus_gray_with_alpha[:, :width_to_overlay]
    fundus_right_with_alpha = fundus_gray_with_alpha[:, width_to_overlay:]

    oct_overlay_with_alpha = oct_gray_with_alpha[:, :width_to_overlay]

    # Blending gambar dengan transparansi
    alpha_fundus = 0.7  # 70% transparansi
    alpha_oct = 0.3  # 30% transparansi
    blended_overlay = cv2.addWeighted(fundus_left_with_alpha, alpha_fundus, oct_overlay_with_alpha, alpha_oct, 0)

    # Gabungkan gambar dengan concatenate
    concatenate = np.concatenate((blended_overlay, fundus_right_with_alpha), axis=1)

    # Simpan hasil penggabungan sementara
    combined_filename = 'combined_image.png'
    combined_path = os.path.join(app.config['UPLOAD_FOLDER'], combined_filename)
    cv2.imwrite(combined_path, concatenate)

    # Prediksi dengan model
    model = load_model('model/CNN_model.h5')
    concatenate_preprocessed = concatenate[:, :, :3] / 255.0  # Normalisasi
    predictions = model.predict(np.expand_dims(concatenate_preprocessed, axis=0))
    class_names = ['Retinopati Diabetik', 'Normal']
    predicted_class = class_names[np.argmax(predictions)]

    fundus_file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(fundus_file.filename))
    oct_file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(oct_file.filename))

    # Render template dengan gambar dan prediksi
    return render_template('prediksi.html', fundus_image=fundus_filename, oct_image=oct_filename, combined_image=combined_filename, prediction=predicted_class)


    # Tentukan filename untuk render di template
    # fundus_filename = fundus_sample if fundus_sample else fundus_file.filename
    # oct_filename = oct_sample if oct_sample else oct_file.filename

# Menampilkan file yang diunggah
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=False)
