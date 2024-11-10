from flask import Flask, render_template, request, redirect, url_for, send_file
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'

# Crea las carpetas si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def imagenes_procesar(image, uploaded_file):
    mask_sizes = [(30, 30), (37, 37), (40, 40)]
    results = {}

    for size in mask_sizes:
        kernel = np.ones(size, np.uint8)
        suffix = f"{size[0]}x{size[1]}"

        # Aplicar operaciones morfológicas con el tamaño de máscara actual
        erosion = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel)
        dilatacion = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)
        top_hat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        black_hat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        top_black_combined = cv2.add(image, cv2.subtract(top_hat, black_hat))

        # Guardar cada resultado con el sufijo del tamaño de máscara, sin espacios en los nombres
        results[f"Original_{suffix}"] = image
        results[f"Erosion_{suffix}"] = erosion
        results[f"Dilatacion_{suffix}"] = dilatacion
        results[f"Top_hat_{suffix}"] = top_hat
        results[f"Black_hat_{suffix}"] = black_hat
        results[f"Imagen_original_{suffix}"] = top_black_combined

    return results



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_files = request.files.getlist('images')
        filenames = []

        for uploaded_file in uploaded_files:
            if uploaded_file.filename != '':
                image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
                uploaded_file.save(image_path)
                
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                # Pasamos el uploaded_file como parámetro
                results = imagenes_procesar(image, uploaded_file)
                
                for name, result in results.items():
                    processed_filename = f"{name}_{os.path.basename(uploaded_file.filename)}"
                    output_path = os.path.join(PROCESSED_FOLDER, processed_filename)
                    cv2.imwrite(output_path, result)
                
                filenames.append(os.path.basename(uploaded_file.filename))
        
        return redirect(url_for('results', filenames=','.join(filenames)))
    return render_template('index.html')


@app.route('/results/<filenames>')
def results(filenames):
    processed_images = {}
    for filename in filenames.split(','):
        processed_images[filename] = {}
        for size in ['30x30', '37x37', '40x40']:
            processed_images[filename][size] = {
                "Original": f"Original_{size}_{filename}",
                "Erosión": f"Erosion_{size}_{filename}",
                "Dilatación": f"Dilatacion_{size}_{filename}",
                "Top_hat": f"Top_hat_{size}_{filename}",
                "Black_hat": f"Black_hat_{size}_{filename}",
                "Imagen_original": f"Imagen_original_{size}_{filename}"
            }
    return render_template('results.html', processed_images=processed_images)

@app.route('/get_image/<filename>')
def get_image(filename):
    img_path = os.path.join(PROCESSED_FOLDER, filename)
    if os.path.exists(img_path):
        return send_file(img_path, mimetype='image/jpeg')
    else:
        return "Imagen no encontrada", 404

if __name__ == '__main__':
    app.run(debug=True)
