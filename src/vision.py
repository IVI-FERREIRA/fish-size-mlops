# src/vision.py

import cv2
import numpy as np


def extract_features_from_image(image_bytes: bytes) -> dict:
    """
    Extrai medidas simples do peixe a partir da imagem.

    Premissas:
    - peixe único na imagem
    - fundo claro
    - medidas retornadas em PIXELS
    """

    # Converte bytes em imagem
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    # Binarização (separa peixe do fundo)
    _, thresh = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Encontra contornos
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Usa o maior contorno (assumindo que é o peixe)
    contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(contour)

    return {
        "Length1": float(w),
        "Length2": float(w),
        "Length3": float(w),
        "Height": float(h),
        "Width": float(h * 0.3),
    }
