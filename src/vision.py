# src/vision.py
import cv2
import numpy as np


def extract_features_from_image(image_bytes: bytes, ruler_cm: float = 15.0) -> dict:
    """
    Extrai medidas do peixe convertendo pixels -> cm
    usando a régua AMARELA visível na imagem.
    """

    # Decode
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # --- DETECÇÃO DA RÉGUA (AMARELA) ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])

    mask_ruler = cv2.inRange(hsv, lower_yellow, upper_yellow)

    contours, _ = cv2.findContours(
        mask_ruler, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("Régua não detectada")

    ruler_contour = max(contours, key=cv2.contourArea)
    _, _, ruler_width_px, _ = cv2.boundingRect(ruler_contour)

    px_per_cm = ruler_width_px / ruler_cm

    # --- DETECÇÃO DO PEIXE ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    fish_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(fish_contour)

    length_cm = w / px_per_cm
    height_cm = h / px_per_cm

    return {
    "Length1": round(length_cm, 2),
    "Length2": round(length_cm, 2),
    "Length3": round(length_cm, 2),
    "Height": round(height_cm, 2),
    "Width": round(height_cm * 0.3, 2),
     }

