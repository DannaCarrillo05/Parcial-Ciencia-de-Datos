from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_flores_cnn.keras"
IMG_SIZE = (64, 64)
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]


@st.cache_resource
def load_model(path: Path, num_classes: int):
	# Se reconstruye la arquitectura y se cargan pesos para evitar
	# incompatibilidades de deserializacion entre versiones de Keras.
	input_layer = tf.keras.layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
	x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(input_layer)
	x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
	x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(x)
	x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
	x = tf.keras.layers.Flatten()(x)
	x = tf.keras.layers.Dense(64, activation="relu")(x)
	output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

	model = tf.keras.Model(inputs=input_layer, outputs=output)
	model.load_weights(path)
	return model


def preprocess_image(image: Image.Image, target_size=(64, 64)) -> np.ndarray:
	image = image.convert("RGB")
	image = image.resize(target_size)
	image_array = np.array(image, dtype=np.float32) / 255.0
	return np.expand_dims(image_array, axis=0)


def main() -> None:
	st.set_page_config(page_title="Clasificador de Flores", layout="wide")

	st.title("Clasificador de Flores con CNN")
	st.write(
		"Carga una imagen de flor y el modelo mostrara la clase predicha y la "
		"distribucion de probabilidades."
	)

	st.subheader("Catalogo de clases")
	st.info(", ".join(CLASS_NAMES))

	try:
		model = load_model(MODEL_PATH, num_classes=len(CLASS_NAMES))
		st.success(f"Modelo cargado correctamente desde: {MODEL_PATH} (modo compatible)")
	except Exception as error:
		st.error(
			"No fue posible cargar el modelo. Asegurate de que exista el archivo "
			f"{MODEL_PATH}."
		)
		st.exception(error)
		return

	uploaded_file = st.file_uploader(
		"Sube una imagen de prueba", type=["jpg", "jpeg", "png", "webp"]
	)

	if not uploaded_file:
		st.warning("Selecciona una imagen para realizar una prediccion.")
		return

	image = Image.open(uploaded_file)
	image_batch = preprocess_image(image, target_size=IMG_SIZE)

	probabilities = model.predict(image_batch, verbose=0)[0]
	best_idx = int(np.argmax(probabilities))
	best_class = CLASS_NAMES[best_idx]
	best_prob = float(probabilities[best_idx])

	result_df = pd.DataFrame(
		{
			"Clase": CLASS_NAMES,
			"Probabilidad": probabilities,
		}
	).sort_values("Probabilidad", ascending=False)

	result_df["Probabilidad (%)"] = (result_df["Probabilidad"] * 100).round(2)
	result_df["Mas probable"] = result_df["Clase"].apply(
		lambda cls: "SI" if cls == best_class else ""
	)

	col1, col2 = st.columns([1, 1])

	with col1:
		st.subheader("Imagen cargada")
		st.image(image, use_container_width=True)
		st.success(
			f"Clase mas probable: {best_class} | Confianza: {best_prob * 100:.2f}%"
		)

	with col2:
		st.subheader("Distribucion de probabilidades")
		st.bar_chart(
			result_df.set_index("Clase")["Probabilidad"],
			use_container_width=True,
		)
		st.dataframe(
			result_df[["Clase", "Probabilidad (%)", "Mas probable"]],
			use_container_width=True,
			hide_index=True,
		)


if __name__ == "__main__":
	main()
