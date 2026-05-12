"""
Streamlit-приложение для классификации изображений Fashion MNIST.
Позволяет загружать изображение или рисовать на холсте, отправляет на API
и отображает результаты классификации.
"""

import io
import requests
import streamlit as st
import numpy as np
from PIL import Image
import plotly.express as px
import pandas as pd
from streamlit_drawable_canvas import st_canvas

API_URL = st.sidebar.text_input(
    "API URL",
    value="https://your-api.onrender.com",
    help="Адрес развернутого FastAPI-бэкенда (например, https://your-app.onrender.com)",
)

st.title("Fashion MNIST Classifier")
st.markdown(
    """
Классификация изображений одежды с помощью нейронной сети.  
**Классы:** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
"""
)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

tab_upload, tab_draw = st.tabs(["Загрузить изображение"])

image_to_send = None

with tab_upload:
    uploaded_file = st.file_uploader(
        "Выберите изображение", type=["png", "jpg", "jpeg", "bmp", "webp"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение", width=200)
        image_to_send = uploaded_file.getvalue()

with tab_draw:
    st.markdown("Нарисуйте предмет одежды на холсте (белым по чёрному фону):")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas_result.image_data is not None:
        drawn = canvas_result.image_data[:, :, :3]
        if np.any(drawn > 0):
            img_pil = Image.fromarray(drawn.astype(np.uint8)).convert("L")
            st.image(img_pil, caption="Ваш рисунок (grayscale)", width=150)
            buf = io.BytesIO()
            img_pil.save(buf, format="PNG")
            image_to_send = buf.getvalue()

if st.button("Классифицировать", type="primary", disabled=image_to_send is None):
    if image_to_send is None:
        st.warning("Сначала загрузите или нарисуйте изображение.")
    else:
        predict_url = API_URL.rstrip("/") + "/predict"
        with st.spinner("Отправка на сервер..."):
            try:
                resp = requests.post(
                    predict_url,
                    files={"file": ("image.png", image_to_send, "image/png")},
                    timeout=30,
                )
                resp.raise_for_status()
                result = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("Не удалось подключиться к API. Проверьте URL в боковой панели.")
                st.stop()
            except requests.exceptions.Timeout:
                st.error("Превышено время ожидания ответа от API.")
                st.stop()
            except Exception as e:
                st.error(f"Ошибка: {e}")
                st.stop()

        st.success(
            f"**Предсказание:** {result['predicted_class']}  "
            f"(уверенность: {result['confidence'] * 100:.1f}%)"
        )

        probs = result["probabilities"]
        df = pd.DataFrame(
            {"Класс": list(probs.keys()), "Вероятность": list(probs.values())}
        )
        df = df.sort_values("Вероятность", ascending=True)

        fig = px.bar(
            df,
            x="Вероятность",
            y="Класс",
            orientation="h",
            title="Распределение вероятностей по классам",
            color="Вероятность",
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            xaxis_title="Вероятность",
            yaxis_title="",
            showlegend=False,
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Подробные вероятности"):
            for cls, prob in sorted(probs.items(), key=lambda x: -x[1]):
                st.write(f"- **{cls}**: {prob * 100:.2f}%")

st.markdown("---")
st.markdown(
    "Разработано в рамках практической работы №10. "
    "[GitHub](https://github.com/)"
)
