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

# Inicializar historico
if "history" not in st.session_state:
    st.session_state.history = []

# Titulo
st.title("🤖 AI Spam Detector")

st.markdown(
    "Classifica mensagens utilizando Machine Learning."
)

st.divider()

# Entrada da mensagem
message = st.text_area(
    "Escreve uma mensagem:",
    placeholder="Exemplo: Congratulations! You won a prize!"
)

# Botao de analise
if st.button("🔍 Analisar mensagem", use_container_width=True):

    if not message.strip():
        st.warning("Escreve uma mensagem primeiro.")

    else:
        prediction = model.predict([message])[0]

        probabilities = model.predict_proba([message])[0]

        classes = model.classes_

        confidence = probabilities[
            list(classes).index(prediction)
        ]

        # Guardar no historico
        st.session_state.history.append({
            "message": message,
            "prediction": prediction,
            "confidence": float(confidence)
        })

        st.subheader("Resultado")

        if prediction == "spam":
            st.error("⚠️ SPAM DETETADO!")
        else:
            st.success("✅ MENSAGEM NORMAL (HAM)")

        st.write(
            f"Confianca estimada: {confidence * 100:.2f}%"
        )

        st.progress(float(confidence))

# Estatisticas
st.divider()
st.subheader("📊 Estatisticas")

total = len(st.session_state.history)

spam_count = sum(
    item["prediction"] == "spam"
    for item in st.session_state.history
)

ham_count = total - spam_count

col1, col2, col3 = st.columns(3)

col1.metric("Total", total)
col2.metric("SPAM", spam_count)
col3.metric("HAM", ham_count)

# Historico
st.divider()
st.subheader("📜 Historico de analises")

if st.session_state.history:

    for item in reversed(st.session_state.history):

        st.write("Mensagem:", item["message"])
        st.write("Classificacao:", item["prediction"].upper())

        st.write(
            f"Confianca: {item['confidence'] * 100:.2f}%"
        )

        st.divider()

    if st.button("Limpar historico"):
        st.session_state.history = []
        st.rerun()

else:
    st.info("Ainda nao existem mensagens analisadas.")

st.caption(
    "AI Spam Detector | Python + Scikit-learn + Streamlit"
)