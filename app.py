# app.py

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from modules.reconstruction import superresolution

app = Flask(__name__)

# Set the upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Main route
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return render_template('index.html', filename=filename)

    return redirect(request.url)

# Route to handle image reconstruction
@app.route('/reconstruct/<filename>')
def reconstruct_file(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Perform image reconstruction using OpenCV (example: invert colors)
    img_lq = cv2.imread(filepath, cv2.IMREAD_COLOR).astype(np.float32) / 255.

    img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
    reconstructed_img = superresolution(img_lq)
    # img = cv2.imread(filepath)
    # reconstructed_img = cv2.bitwise_not(img)

    # Save the reconstructed image
    reconstructed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'reconstructed_' + filename)
    cv2.imwrite(reconstructed_filepath, reconstructed_img)

    return render_template('reconstructed.html', original=filename, reconstructed='reconstructed_' + filename)


# ...

# Route to handle the download of the reconstructed image
@app.route('/download/<filename>')
def download_reconstructed(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

# ...

if __name__ == '__main__':
    app.run(debug=True)
