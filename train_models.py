"""
Скрипт для обучения 4 моделей на Fashion MNIST и сохранения лучшей модели.
"""

import os, time, json
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
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import MobileNetV2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

os.makedirs("models", exist_ok=True)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

print("Загрузка Fashion MNIST...")
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()

x_train = trainX.astype("float32") / 255.0
x_test  = testX.astype("float32") / 255.0
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn  = x_test.reshape(-1, 28, 28, 1)

num_classes = 10
y_train = to_categorical(trainy, num_classes)
y_test  = to_categorical(testy, num_classes)
print(f"Train: {x_train_cnn.shape}, Test: {x_test_cnn.shape}")

BATCH = 128
results = {}
histories = {}

cb = lambda: [
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
]

def evaluate(model, x, y_true, name):
    times = []
    for _ in range(5):
        t0 = time.time(); preds = model.predict(x, verbose=0); times.append(time.time()-t0)
    ms = (np.mean(times)/len(x))*1000
    yp = np.argmax(preds, axis=1)
    r = {
        "model_name": name,
        "accuracy": round(accuracy_score(y_true, yp), 4),
        "precision": round(precision_score(y_true, yp, average="macro", zero_division=0), 4),
        "recall": round(recall_score(y_true, yp, average="macro", zero_division=0), 4),
        "f1_score": round(f1_score(y_true, yp, average="macro", zero_division=0), 4),
        "inference_time_per_image_ms": round(ms, 4),
        "confusion_matrix": confusion_matrix(y_true, yp).tolist(),
    }
    print(f"\n{name}: acc={r['accuracy']}, f1={r['f1_score']}, ms/img={r['inference_time_per_image_ms']}")
    return r

# === Модель 1: Dense NN (Работа 2) ===
print("\n" + "="*50 + "\nМодель 1: Dense NN\n" + "="*50)
m1 = Sequential([
    Input(shape=(28,28,1)), Flatten(),
    Dense(512, activation="relu"), Dense(256, activation="relu"),
    Dense(128, activation="relu"), Dense(num_classes, activation="softmax"),
], name="dense_nn")
m1.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
h1 = m1.fit(x_train_cnn, y_train, validation_data=(x_test_cnn, y_test),
            epochs=20, batch_size=BATCH, callbacks=cb(), verbose=2)
histories["Dense NN (Работа 2)"] = h1.history
results["Dense NN (Работа 2)"] = evaluate(m1, x_test_cnn, testy, "Dense NN (Работа 2)")

# === Модель 2: CNN без BN/Dropout (Работа 3) ===
print("\n" + "="*50 + "\nМодель 2: CNN без BN/Dropout\n" + "="*50)
m2 = Sequential([
    Input(shape=(28,28,1)),
    Conv2D(32, (3,3), padding="same", activation="relu"),
    Conv2D(32, (3,3), padding="same", activation="relu"),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), padding="same", activation="relu"),
    Conv2D(64, (3,3), padding="same", activation="relu"),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), padding="same", activation="relu"),
    MaxPooling2D((2,2)),
    Flatten(), Dense(256, activation="relu"), Dense(num_classes, activation="softmax"),
], name="cnn_simple")
m2.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
h2 = m2.fit(x_train_cnn, y_train, validation_data=(x_test_cnn, y_test),
            epochs=20, batch_size=BATCH, callbacks=cb(), verbose=2)
histories["CNN без BN/Dropout (Работа 3)"] = h2.history
results["CNN без BN/Dropout (Работа 3)"] = evaluate(m2, x_test_cnn, testy, "CNN без BN/Dropout (Работа 3)")

# === Модель 3: CNN с BN и Dropout (Работа 4) ===
print("\n" + "="*50 + "\nМодель 3: CNN с BN и Dropout\n" + "="*50)
m3 = Sequential([
    Input(shape=(28,28,1)),
    Conv2D(32, (3,3), padding="same", activation="relu"), BatchNormalization(),
    Conv2D(32, (3,3), padding="same", activation="relu"), BatchNormalization(),
    MaxPooling2D((2,2)), Dropout(0.25),
    Conv2D(64, (3,3), padding="same", activation="relu"), BatchNormalization(),
    Conv2D(64, (3,3), padding="same", activation="relu"), BatchNormalization(),
    MaxPooling2D((2,2)), Dropout(0.25),
    Conv2D(128, (3,3), padding="same", activation="relu"), BatchNormalization(),
    Conv2D(128, (3,3), padding="same", activation="relu"), BatchNormalization(),
    MaxPooling2D((2,2)), Dropout(0.25),
    Flatten(), Dense(512, activation="relu"), BatchNormalization(), Dropout(0.5),
    Dense(num_classes, activation="softmax"),
], name="cnn_bn_dropout")
m3.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
h3 = m3.fit(x_train_cnn, y_train, validation_data=(x_test_cnn, y_test),
            epochs=30, batch_size=BATCH, callbacks=[
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
                EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
            ], verbose=2)
histories["CNN с BN и Dropout (Работа 4)"] = h3.history
results["CNN с BN и Dropout (Работа 4)"] = evaluate(m3, x_test_cnn, testy, "CNN с BN и Dropout (Работа 4)")

# === Модель 4: MobileNetV2 Transfer Learning (Работа 5) ===
print("\n" + "="*50 + "\nМодель 4: MobileNetV2 Transfer\n" + "="*50)
SZ = 96
print(f"Resize to {SZ}x{SZ} RGB...")
x_tr_rgb = tf.image.resize(np.repeat(x_train_cnn, 3, axis=-1), (SZ, SZ)).numpy()
x_te_rgb = tf.image.resize(np.repeat(x_test_cnn,  3, axis=-1), (SZ, SZ)).numpy()

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(SZ, SZ, 3))
base.trainable = False
inp = Input(shape=(SZ, SZ, 3))
x = base(inp, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
out = Dense(num_classes, activation="softmax")(x)
m4 = Model(inp, out, name="mobilenetv2_transfer")
m4.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

print("Этап 1: обучение головы...")
h4a = m4.fit(x_tr_rgb, y_train, validation_data=(x_te_rgb, y_test),
             epochs=8, batch_size=64, verbose=2)

print("Этап 2: fine-tuning...")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
m4.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
h4b = m4.fit(x_tr_rgb, y_train, validation_data=(x_te_rgb, y_test),
             epochs=10, batch_size=64, callbacks=[
                 ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
                 EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
             ], verbose=2)

combined = {k: h4a.history[k]+h4b.history[k] for k in h4a.history}
histories["MobileNetV2 Transfer (Работа 5)"] = combined
results["MobileNetV2 Transfer (Работа 5)"] = evaluate(m4, x_te_rgb, testy, "MobileNetV2 Transfer (Работа 5)")

# === Итоги ===
print("\n" + "="*50 + "\nСравнение\n" + "="*50)
for n, r in results.items():
    print(f"  {n}: F1={r['f1_score']}, Acc={r['accuracy']}")

best = max(results, key=lambda k: results[k]["f1_score"])
print(f"\nЛучшая: {best} (F1={results[best]['f1_score']})")

models = {"Dense NN (Работа 2)": m1, "CNN без BN/Dropout (Работа 3)": m2,
          "CNN с BN и Dropout (Работа 4)": m3, "MobileNetV2 Transfer (Работа 5)": m4}
models[best].save("models/best_classification_model.keras")

with open("models/comparison_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

ser_hist = {n: {k: [float(v) for v in vs] for k,vs in h.items()} for n,h in histories.items()}
with open("models/training_histories.json", "w", encoding="utf-8") as f:
    json.dump(ser_hist, f, ensure_ascii=False, indent=2)

for n, mdl in models.items():
    fname = n.split("(")[0].strip().lower().replace(" ", "_").replace("/", "_")
    mdl.save(f"models/model_{fname}.keras")

print("Готово!")
