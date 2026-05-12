"""
FastAPI-приложение для классификации изображений Fashion MNIST.
Загружает лучшую модель и предоставляет эндпоинт /predict для классификации.
"""

import io
import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf

app = FastAPI(
    title="Fashion MNIST Classifier API",
    description="API для классификации изображений одежды (Fashion MNIST)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "best_classification_model.keras"),
)

model = None


def load_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model file not found: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
    return model


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Преобразует загруженное изображение в формат 28x28 grayscale, нормализует.

    Fashion MNIST: белый объект на чёрном фоне.
    Если входное изображение имеет светлый фон (>50% ярких пикселей),
    инвертируем его, чтобы привести к формату датасета.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0

    # Автоматическая инверсия: если фон светлый, инвертируем
    if np.mean(arr) > 0.5:
        arr = 1.0 - arr

    arr = arr.reshape(1, 28, 28, 1)
    return arr


@app.on_event("startup")
async def startup_event():
    try:
        load_model()
    except Exception as e:
        print(f"Warning: could not load model on startup: {e}")


@app.get("/")
async def root():
    return {
        "message": "Fashion MNIST Classifier API",
        "endpoints": {"/predict": "POST an image to classify"},
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Принимает изображение, возвращает предсказанный класс и вероятности
    по всем 10 классам Fashion MNIST.
    """
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        img_array = preprocess_image(contents)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not process image")

    mdl = load_model()
    predictions = mdl.predict(img_array, verbose=0)
    probs = predictions[0].tolist()

    predicted_class_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = float(probs[predicted_class_idx])

    return {
        "predicted_class": predicted_class,
        "predicted_class_index": predicted_class_idx,
        "confidence": round(confidence, 4),
        "probabilities": {
            CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(probs)
        },
    }
