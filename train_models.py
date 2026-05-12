"""
Скрипт для обучения 4 моделей на Fashion MNIST и сохранения лучшей модели.
Модели:
  1) Dense NN (Работа 2)
  2) CNN без BN/Dropout (Работа 3)
  3) CNN с BN и Dropout (Работа 4)
  4) Transfer Learning — MobileNetV2 (Работа 5)
"""

import os
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Input,
    BatchNormalization, Dropout, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

os.makedirs("models", exist_ok=True)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# ---------- Загрузка данных ----------
print("Загрузка Fashion MNIST...")
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()

x_train = trainX.astype("float32") / 255.0
x_test  = testX.astype("float32")  / 255.0

x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn  = x_test.reshape(-1, 28, 28, 1)

num_classes = 10
y_train = to_categorical(trainy, num_classes)
y_test  = to_categorical(testy, num_classes)

print(f"Train: {x_train_cnn.shape}, Test: {x_test_cnn.shape}")

EPOCHS = 10
BATCH_SIZE = 128

results = {}


def evaluate_model(model, x, y_true_cat, y_true_labels, model_name):
    """Оценивает модель и возвращает метрики."""
    # Время инференса
    times = []
    for _ in range(3):
        start = time.time()
        preds = model.predict(x, verbose=0)
        elapsed = time.time() - start
        times.append(elapsed)
    avg_inference_time = np.mean(times)
    avg_per_image_ms = (avg_inference_time / len(x)) * 1000

    y_pred = np.argmax(preds, axis=1)

    acc = accuracy_score(y_true_labels, y_pred)
    prec = precision_score(y_true_labels, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true_labels, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true_labels, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true_labels, y_pred)

    result = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "inference_time_total_s": round(avg_inference_time, 4),
        "inference_time_per_image_ms": round(avg_per_image_ms, 4),
        "confusion_matrix": cm.tolist(),
    }
    print(f"\n{model_name}: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, "
          f"f1={f1:.4f}, infer={avg_per_image_ms:.4f} ms/img")
    return result


# ==================== Модель 1: Dense NN (Работа 2) ====================
print("\n" + "=" * 60)
print("Модель 1: Dense NN (Работа 2)")
print("=" * 60)

model_1 = Sequential([
    Input(shape=(28, 28, 1)),
    Flatten(),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"),
    Dense(num_classes, activation="softmax"),
], name="dense_nn")

model_1.compile(optimizer=Adam(learning_rate=1e-3),
                loss="categorical_crossentropy", metrics=["accuracy"])
model_1.summary()

model_1.fit(x_train_cnn, y_train, validation_data=(x_test_cnn, y_test),
            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

results["Dense NN (Работа 2)"] = evaluate_model(
    model_1, x_test_cnn, y_test, testy, "Dense NN (Работа 2)")


# ==================== Модель 2: CNN без BN/Dropout (Работа 3) ====================
print("\n" + "=" * 60)
print("Модель 2: CNN без BN/Dropout (Работа 3)")
print("=" * 60)

model_2 = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(num_classes, activation="softmax"),
], name="cnn_no_bn_dropout")

model_2.compile(optimizer=Adam(learning_rate=1e-3),
                loss="categorical_crossentropy", metrics=["accuracy"])
model_2.summary()

model_2.fit(x_train_cnn, y_train, validation_data=(x_test_cnn, y_test),
            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

results["CNN без BN/Dropout (Работа 3)"] = evaluate_model(
    model_2, x_test_cnn, y_test, testy, "CNN без BN/Dropout (Работа 3)")


# ==================== Модель 3: CNN с BN и Dropout (Работа 4) ====================
print("\n" + "=" * 60)
print("Модель 3: CNN с BN и Dropout (Работа 4)")
print("=" * 60)

model_3 = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    Conv2D(32, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    Conv2D(64, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation="softmax"),
], name="cnn_bn_dropout")

model_3.compile(optimizer=Adam(learning_rate=1e-3),
                loss="categorical_crossentropy", metrics=["accuracy"])
model_3.summary()

model_3.fit(x_train_cnn, y_train, validation_data=(x_test_cnn, y_test),
            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

results["CNN с BN и Dropout (Работа 4)"] = evaluate_model(
    model_3, x_test_cnn, y_test, testy, "CNN с BN и Dropout (Работа 4)")


# ==================== Модель 4: Transfer Learning MobileNetV2 (Работа 5) ====================
print("\n" + "=" * 60)
print("Модель 4: Transfer Learning MobileNetV2 (Работа 5)")
print("=" * 60)

# Fashion MNIST 28x28 -> resize to 32x32 RGB for MobileNetV2
x_train_rgb = np.repeat(x_train_cnn, 3, axis=-1)  # (60000, 28, 28, 3)
x_test_rgb  = np.repeat(x_test_cnn, 3, axis=-1)

x_train_rgb = tf.image.resize(x_train_rgb, (32, 32)).numpy()
x_test_rgb  = tf.image.resize(x_test_rgb, (32, 32)).numpy()

base_model = MobileNetV2(
    weights="imagenet", include_top=False,
    input_shape=(32, 32, 3),
)
base_model.trainable = False

inputs = Input(shape=(32, 32, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation="softmax")(x)
model_4 = Model(inputs, outputs, name="mobilenetv2_transfer")

model_4.compile(optimizer=Adam(learning_rate=1e-3),
                loss="categorical_crossentropy", metrics=["accuracy"])
model_4.summary()

model_4.fit(x_train_rgb, y_train, validation_data=(x_test_rgb, y_test),
            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

results["MobileNetV2 Transfer (Работа 5)"] = evaluate_model(
    model_4, x_test_rgb, y_test, testy, "MobileNetV2 Transfer (Работа 5)")


# ---------- Выбор лучшей модели по F1-мере ----------
print("\n" + "=" * 60)
print("Сравнение моделей")
print("=" * 60)

best_name = max(results, key=lambda k: results[k]["f1_score"])
print(f"\nЛучшая модель по F1-мере: {best_name} (F1 = {results[best_name]['f1_score']})")

# Сохранение лучшей модели
model_map = {
    "Dense NN (Работа 2)": model_1,
    "CNN без BN/Dropout (Работа 3)": model_2,
    "CNN с BN и Dropout (Работа 4)": model_3,
    "MobileNetV2 Transfer (Работа 5)": model_4,
}

best_model = model_map[best_name]
best_model.save("models/best_classification_model.keras")
print(f"Лучшая модель сохранена: models/best_classification_model.keras")

# Сохранение результатов в JSON
with open("models/comparison_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("Результаты сравнения сохранены: models/comparison_results.json")

# Сохраняем также все модели для воспроизводимости
model_1.save("models/model_dense_nn.keras")
model_2.save("models/model_cnn_no_bn_dropout.keras")
model_3.save("models/model_cnn_bn_dropout.keras")
model_4.save("models/model_mobilenetv2_transfer.keras")
print("Все модели сохранены в папку models/")
