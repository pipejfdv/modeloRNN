# Teclado Visual Predictivo AI

Aplicación de teclado predictivo que utiliza un modelo GRU entrenado con conversaciones de WhatsApp para sugerir la siguiente palabra. La interfaz está construida con Streamlit.

## Requisitos

- Python 3.10 o superior
- `predictive_text_model.h5` (modelo entrenado)
- `tokenizer.pkl` (tokenizador)

Ambos archivos deben estar en el mismo directorio que `app.py`.

## Instalación

```bash
# Crear y activar un entorno virtual (opcional pero recomendado)
python -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Ejecución

```bash
streamlit run app.py
```

Luego abre en tu navegador la URL que muestra la terminal (por defecto `http://localhost:8501`).

## Uso

1. Escribe un mensaje en el campo de texto.
2. La aplicación mostrará hasta 3 sugerencias de la siguiente palabra.
3. Haz clic en una sugerencia (✨) para agregarla al mensaje.

## Estructura del proyecto

```
.
├── app.py                       # Aplicación Streamlit
├── predictive_text_model.h5     # Modelo GRU entrenado
├── tokenizer.pkl                # Tokenizador entrenado
├── requirements.txt             # Dependencias
└── README.md
```
