#  CLASIFICADOR DE FRUTAS

import os
import zipfile
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import numpy as np
from tensorflow.keras.preprocessing import image


# Descomprimir

zip_path = "./data/fruits.zip"
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("./data")
    print("✅ Dataset descomprimido correctamente.")
else:
    print(" No se encontró el archivo fruits.zip")


# CARGA  DE DATOS

print("Preparando generadores de datos...")

train_dir = r"C:\Users\adrin\Desktop\Proyecto Machine Learning\Frutas\data\fruits-360_original-size\fruits-360-original-size\Training"
val_dir   = r"C:\Users\adrin\Desktop\Proyecto Machine Learning\Frutas\data\fruits-360_original-size\fruits-360-original-size\Validation"

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)

val_data = datagen.flow_from_directory(
    val_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)

num_classes = len(train_data.class_indices)
print(f"✅ Clases detectadas: {num_classes}")


# MODELO

print("Construyendo modelo con MobileNetV2...")

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# ENTRENAMIENTO

print("Entrenando modelo...")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)


# EVALUACIÓN Y VISUALIZACIÓN

print("Evaluando resultados...")

plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)
plt.show()


# GUARDAR EL MODELO

os.makedirs("models", exist_ok=True)
model.save("models/fruit_classifier_mobilenetv2.h5")
print("✅ Modelo guardado en models/fruit_classifier_mobilenetv2.h5")


# PREDICCIÓN DE EJEMPLO

example_path = os.path.join(val_dir, os.listdir(val_dir)[0], os.listdir(os.path.join(val_dir, os.listdir(val_dir)[0]))[0])
print(f"Ejemplo de predicción: {example_path}")

img = image.load_img(example_path, target_size=(100,100))
img_array = np.expand_dims(np.array(img)/255.0, axis=0)
pred = model.predict(img_array)

pred_class = list(train_data.class_indices.keys())[np.argmax(pred)]
print(f"🔮 Predicción del modelo: {pred_class}")
