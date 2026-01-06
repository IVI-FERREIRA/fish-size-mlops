# src/vision.py

import cv2
import numpy as np


def extract_features_from_image(
    image_bytes: bytes,
    ruler_cm: float = 10.0  # quantos cm reais vamos usar como referência
) -> dict:
    """
    Extrai medidas do peixe e converte de pixels para centímetros
    usando uma régua visível na imagem.
    """

    # Decodifica imagem
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

    # Binarização
    _, thresh = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Assume que o maior contorno é o peixe
    fish_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(fish_contour)

    # --- REGUA (simples) ---
    # Assumimos que a régua está na parte inferior da imagem
    height_img = image.shape[0]
    ruler_region = thresh[int(height_img * 0.85):, :]  # 15% inferior

    ruler_contours, _ = cv2.findContours(
        ruler_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    ruler_contour = max(ruler_contours, key=cv2.contourArea)
    _, _, ruler_width_px, _ = cv2.boundingRect(ruler_contour)

    # Pixels por centímetro
    px_per_cm = ruler_width_px / ruler_cm

    # Conversão para cm
    length_cm = w / px_per_cm
    height_cm = h / px_per_cm

    return {
        "Length1": float(length_cm),
        "Length2": float(length_cm),
        "Length3": float(length_cm),
        "Height": float(height_cm),
        "Width": float(height_cm * 0.3),
    }
