import streamlit as st
import joblib

# Configuracao da pagina
st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="🤖",
    layout="centered"
)

# Carregar o modelo
@st.cache_resource
def load_model():
    return joblib.load("spam_model.pkl")

model = load_model()

# Cabecalho
st.title("🤖 AI Spam Detector")

st.markdown(
    "### Inteligencia Artificial para deteccao de SPAM"
)

st.write(
    "Introduz uma mensagem e deixa a nossa IA "
    "analisar o seu conteudo."
)

st.divider()

# Entrada do utilizador
message = st.text_area(
    "Escreve a tua mensagem:",
    placeholder="Exemplo: Congratulations! You won a prize!",
    height=150
)

# Botao de analise
if st.button("🔍 Analisar mensagem", use_container_width=True):

    if message.strip():

        # Previsao
        prediction = model.predict([message])[0]

        # Probabilidades
        probabilities = model.predict_proba([message])[0]

        # Obter as classes
        classes = model.classes_

        # Encontrar a probabilidade da previsao
        confidence = probabilities[
            list(classes).index(prediction)
        ]

        # Apresentar resultado
        st.divider()

        st.subheader("Resultado da analise")

        if prediction == "spam":
            st.error("⚠️ SPAM DETETADO!")
        else:
            st.success("✅ MENSAGEM NORMAL (HAM)")

        # Mostrar confianca estimada
        st.write(
            f"Confiança estimada: {confidence * 100:.2f}%"
        )

        st.progress(float(confidence))

        # Mostrar probabilidades das categorias
        st.subheader("Probabilidades por categoria")

        for category, probability in zip(classes, probabilities):
            st.write(
                f"{category.upper()}: {probability * 100:.2f}%"
            )

    else:
        st.warning("Por favor, escreve uma mensagem.")

st.divider()

st.caption(
    "AI Spam Detector | Desenvolvido com Python, "
    "Scikit-learn e Streamlit."
)