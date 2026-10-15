import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Teclado Predictivo AI", layout="centered")

st.title("📱 Teclado Visual Predictivo")
st.write("Este teclado predictivo utiliza un modelo GRU entrenado en conversaciones de WhatsApp para adivinar tu siguiente palabra.")

@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model('predictive_text_model.h5')
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

try:
    model, tokenizer = load_model_and_tokenizer()
    st.success("¡Modelo y Tokenizador cargados con éxito!")
except Exception as e:
    st.error(f"Error al cargar los archivos. Asegúrate de tener 'predictive_text_model.h5' y 'tokenizer.pkl' en el mismo directorio. {e}")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Estado persistente del texto escrito
if 'texto' not in st.session_state:
    st.session_state.texto = ""

# Entrada de texto del usuario
input_text = st.text_input("Escribe tu mensaje aquí:", value=st.session_state.texto)
st.session_state.texto = input_text

# Lógica de predicción de la siguiente palabra
sugerencias = []
if st.session_state.texto:
    cleaned = clean_text(st.session_state.texto)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=14, padding='pre')  # max_sequence_len - 1 = 14

    preds = model.predict(padded, verbose=0)[0]
    top_indices = np.argsort(preds)[-3:][::-1]

    for index in top_indices:
        for word, idx in tokenizer.word_index.items():
            if idx == index:
                sugerencias.append(word)
                break

# Teclado Visual Predictivo
st.write("### Sugerencias del Teclado:")
if sugerencias:
    cols = st.columns(3)
    for idx, sug in enumerate(sugerencias):
        if cols[idx].button(f"✨ {sug}", key=f"btn_{sug}_{idx}"):
            st.session_state.texto += f" {sug}"
            st.rerun()
else:
    st.info("Escribe algo para ver predicciones de la IA.")

st.write("**Mensaje actual:** ", st.session_state.texto)