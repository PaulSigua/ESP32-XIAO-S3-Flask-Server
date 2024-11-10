# Author: vlarobbyk
# Version: 1.0
# Date: 2024-10-20
# Description: A simple example to process video captured by the ESP32-XIAO-S3 or ESP32-CAM-MB in Flask.

from flask import Flask, render_template, Response, request
from io import BytesIO
import cv2
import numpy as np
import requests
import time

app = Flask(__name__)

_URL = 'http://192.168.18.143'
_PORT = '81'
_ST = '/stream'
SEP = ':'

stream_url = ''.join([_URL,SEP,_PORT,_ST])

background_subtractor_knn = cv2.createBackgroundSubtractorKNN()
prev_frame_time = 0
new_frame_time = 0

# Variables globales para la resta de fondo (filtro investigado)
background = None
MAX_FRAMES = 1000
THRESH = 60
ASSIGN_VALUE = 255

# Valores de ruido iniciales
salt_noise = 5  # Porcentaje de ruido de sal
pepper_noise = 5  # Porcentaje de ruido de pimienta

selected_filter = 0  # Filtro seleccionado en el trackbar

def update_filter(value):
    global selected_filter
    selected_filter = value

cv2.namedWindow("Filtro Aplicado")
cv2.createTrackbar("Filtro", "Filtro Aplicado", 0, 8, update_filter)

def video_capture():
    res = requests.get(stream_url, stream=True)
    for chunk in res.iter_content(chunk_size=100000):
        if len(chunk) > 100:
            try:
                img_data = BytesIO(chunk)
                cv_img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)

                # Detectar movimiento y mostrar FPS
                fg_mask = detect_motion(cv_img)
                cv_img = show_fps(cv_img)

                # Aplicar mejoras de iluminación
                equalized = apply_histogram_equalization(cv_img)
                clahe_applied = apply_CLAHE(cv_img)
                background_subtracted = background_subtraction_knn(cv_img)

                # Aplicar ruido de sal y pimienta con trackbars
                noisy_frame = add_salt_pepper_noise(cv_img, salt_noise, pepper_noise)

                # Aplicar filtros de suavizado para el ruido
                median_filtered = cv2.medianBlur(noisy_frame, 5)
                blur_filtered = cv2.blur(noisy_frame, (5, 5))

                # Detección de bordes
                edges_canny = cv2.Canny(median_filtered, 100, 200)
                edges_sobel = cv2.Sobel(median_filtered, cv2.CV_64F, 1, 0, ksize=5)

                # Convertir solo imágenes en escala de grises a color
                def ensure_color(image):
                    if len(image.shape) == 2 or image.shape[2] == 1:
                        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    return image

                images = [
                    cv_img,
                    background_subtracted,
                    ensure_color(equalized),
                    ensure_color(clahe_applied),
                    ensure_color(noisy_frame),
                    ensure_color(median_filtered),
                    ensure_color(blur_filtered),
                    ensure_color(edges_canny),
                    ensure_color(edges_sobel),
                ]
                
                # Crear imagen de salida en negro
                output_image = np.zeros_like(cv_img)

                # Copiar la imagen seleccionada en la posición de salida usando copyTo
                images[selected_filter].copyTo(output_image)

                # Agregar el nombre del filtro
                filter_names = [
                    'Imagen Original', 'Resta de Fondo con KNN', 'Ecualización de Histograma', 'CLAHE',
                    'Ruido Sal y Pimienta', 'Mediana Filtrada', 'Gaussiano Filtrado', 'Canny', 'Sobel'
                ]
                cv2.putText(output_image, filter_names[selected_filter], (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Codificar imagen combinada
                (flag, encodedImage) = cv2.imencode(".jpg", output_image)
                if not flag:
                    continue

                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                      bytearray(encodedImage) + b'\r\n')

            except Exception as e:
                print(e)
                continue

def detect_motion(frame):
    fg_mask = background_subtractor_knn.apply(frame)
    return fg_mask

def show_fps(frame):
    global prev_frame_time, new_frame_time
    new_frame_time = time.time()
    fps = int(1 / (new_frame_time - prev_frame_time))
    prev_frame_time = new_frame_time
    cv2.putText(frame, f'FPS: {fps}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

def apply_histogram_equalization(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized

def apply_CLAHE(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_applied = clahe.apply(gray)
    return clahe_applied

def add_salt_pepper_noise(image, salt_prob, pepper_prob):
    noisy = image.copy()
    h, w = noisy.shape[:2]
    # Aplicar ruido de sal
    num_salt = np.ceil(salt_prob * h * w / 100)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255
    
    # Aplicar ruido de pimienta
    num_pepper = np.ceil(pepper_prob * h * w / 100)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0
    return noisy

def background_subtraction_knn(frame):
    fg_mask = background_subtractor_knn.apply(frame)
    return fg_mask

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_stream")
def video_stream():
    return Response(video_capture(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/set_noise", methods=["POST"])
def set_noise():
    global salt_noise, pepper_noise
    salt_noise = int(request.form.get("salt", 5))
    pepper_noise = int(request.form.get("pepper", 5))
    return ("", 204)

if __name__ == "__main__":
    app.run(debug=True)
