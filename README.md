# Fashion MNIST Classification — Практическая работа №10

Сравнение моделей классификации изображений на датасете Fashion MNIST, развертывание лучшей модели в виде API (FastAPI) и создание интерфейса (Streamlit).

## Датасет

**Fashion MNIST** — 70 000 изображений 28×28 в оттенках серого, 10 классов предметов одежды:

| Индекс | Класс        |
|--------|-------------|
| 0      | T-shirt/top |
| 1      | Trouser     |
| 2      | Pullover    |
| 3      | Dress       |
| 4      | Coat        |
| 5      | Sandal      |
| 6      | Shirt       |
| 7      | Sneaker     |
| 8      | Bag         |
| 9      | Ankle boot  |

## Сравниваемые модели

| # | Модель | Описание |
|---|--------|----------|
| 1 | **Dense NN** (Работа 2) | Полносвязная нейронная сеть: Flatten → Dense(256) → Dense(128) → Dense(10) |
| 2 | **CNN без BN/Dropout** (Работа 3) | Свёрточная сеть: 3 блока Conv2D+MaxPool → Dense(128) → Dense(10) |
| 3 | **CNN с BN и Dropout** (Работа 4) | CNN с BatchNormalization и Dropout для регуляризации |
| 4 | **MobileNetV2 Transfer** (Работа 5) | Трансферное обучение на базе MobileNetV2 (ImageNet) |

## Результаты сравнения

| Модель | Accuracy | Precision | Recall | F1-мера | Время инференса (мс/изобр.) |
|--------|----------|-----------|--------|---------|----------------------------|
| Dense NN (Работа 2) | 0.8880 | 0.8888 | 0.8880 | 0.8867 | 0.0272 |
| **CNN без BN/Dropout (Работа 3)** | **0.9143** | **0.9174** | **0.9143** | **0.9149** | 0.0934 |
| CNN с BN и Dropout (Работа 4) | 0.9058 | 0.9161 | 0.9058 | 0.9050 | 0.1977 |
| MobileNetV2 Transfer (Работа 5) | 0.6716 | 0.6930 | 0.6716 | 0.6674 | 0.4100 |

**Лучшая модель по F1-мере:** CNN без BN/Dropout (Работа 3) — F1 = 0.9149

## Структура проекта

```
fashion-mnist-classification/
├── api/
│   ├── main.py                          # FastAPI-приложение
│   └── requirements.txt                 # Зависимости API
├── streamlit_app/
│   ├── app.py                           # Streamlit-приложение
│   └── requirements.txt                 # Зависимости Streamlit
├── models/
│   └── best_classification_model.keras  # Лучшая модель
├── practical_work_10.ipynb              # Ноутбук с анализом
├── train_models.py                      # Скрипт обучения моделей
├── README.md                            # Документация
└── .gitignore
```

## Локальный запуск

### 1. Обучение моделей (опционально)

```bash
pip install tensorflow scikit-learn numpy pandas matplotlib seaborn
python train_models.py
```

### 2. Запуск API

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. Запуск Streamlit

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

В боковой панели Streamlit укажите URL вашего API (например `http://localhost:8000`).

## Примеры использования API

### Классификация изображения

```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@image.png"
```

### Пример ответа

```json
{
    "predicted_class": "Ankle boot",
    "predicted_class_index": 9,
    "confidence": 0.9871,
    "probabilities": {
        "T-shirt/top": 0.0001,
        "Trouser": 0.0002,
        "Pullover": 0.0005,
        "Dress": 0.0003,
        "Coat": 0.0004,
        "Sandal": 0.0012,
        "Shirt": 0.0001,
        "Sneaker": 0.0089,
        "Bag": 0.0012,
        "Ankle boot": 0.9871
    }
}
```

## Развертывание

### API на Render.com

1. Создайте Web Service на [render.com](https://render.com)
2. Укажите команду запуска: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
3. В Environment Variables добавьте: `PYTHON_VERSION=3.12.1`
4. Root Directory: оставьте пустым (корень репозитория)

### Streamlit на Streamlit Cloud

1. Перейдите на [share.streamlit.io](https://share.streamlit.io)
2. Укажите репозиторий и файл `streamlit_app/app.py`
3. В боковой панели приложения введите URL вашего API на Render.com

## Ссылки

- **GitHub-репозиторий:** *ваша ссылка*
- **API (Render.com):** *ваша ссылка*
- **Streamlit-интерфейс:** *ваша ссылка*
