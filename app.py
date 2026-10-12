import gradio as gr
import os
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import cv2

def process_passport(image):
    if image is None:
        return None

    # Convert to OpenCV
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # --- 1. Gentle glare / contrast improvement using LAB ---
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Mild CLAHE (very important: low clipLimit)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # --- 2. Light edge-preserving smoothing (reduces noise without blurring text) ---
    img = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)

    # Convert back to PIL
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # --- 3. Controlled sharpening for text ---
    enhancer = ImageEnhance.Sharpness(img_pil)
    img_pil = enhancer.enhance(1.45)

    # --- 4. Slight contrast boost ---
    enhancer = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer.enhance(1.15)

    return np.array(img_pil)

demo = gr.Interface(
    fn=process_passport,
    inputs=gr.Image(type="numpy", label="Upload Passport Photo"),
    outputs=gr.Image(type="numpy", label="Cleaned Result", format="png"),
    title="Passport Cleaner",
    description="Improved version – better glare reduction and clearer text (gentle processing)."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
