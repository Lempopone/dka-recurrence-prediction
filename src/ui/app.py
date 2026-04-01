import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://127.0.0.1:8000"

st.set_page_config(layout="wide")

st.title("Система поддержки принятия решений при ДКА")

patient_id = st.text_input("Введите номер истории болезни")

if st.button("Получить прогноз"):

    response = requests.get(f"{API_URL}/predict/{patient_id}").json()

    if "error" in response:
        st.error("Пациент не найден")
    else:
        clinical = response["clinical"]

        ### уровень риска
        st.header(f"Уровень риска: {clinical['risk_level']}")
        st.write(clinical["summary"])

        col1, col2 = st.columns(2)
        col1.metric("Прогноз", "Рецидив" if response["prediction"] else "Без рецидива")
        col2.metric("Надежность", response["confidence"])

        ### инфо о выборочной группе
        if response["dataset"] == "train":
            st.info("Пациент был использован в обучающей выборке")
        elif response["dataset"] == "test":
            st.warning("Пациент находится в тестовой выборке")
        else:
            st.write("Пациент не был в обучающих или тестовых данных")

        ### ключевые факторы
        st.subheader("Ключевые факторы риска")

        for d in clinical["drivers"]:
            st.write(f"{d['feature']} — {d['impact']}")

        ### клинические отклонения
        st.subheader("Клинически значимые отклонения")

        if clinical["alerts"]:
            for alert in clinical["alerts"]:
                st.warning(alert)
        else:
            st.success("Значимых отклонений не выявлено")

        ### визуализация факторов
        shap_data = pd.DataFrame({
            "feature": list(response["shap"].keys()),
            "value": list(response["shap"].values())
        }).sort_values("value", key=abs, ascending=False).head(10)

        fig = px.bar(
            shap_data,
            x="value",
            y="feature",
            orientation="h",
            title="Факторы, влияющие на риск"
        )

        st.plotly_chart(fig, use_container_width=True)

        ### вероятность модели
        st.subheader("Вероятность рецидива (predict_proba)")
        st.write(f"{response['proba']:.3f}")
