import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

# Cargar modelo
model_path = "models/fruit_classifier_mobilenetv2.h5"
model = load_model(model_path)
print("✅ Modelo cargado correctamente.")

# Detectar automáticamente las clases según el dataset original
train_dir = r"C:\Users\adrin\Desktop\Proyecto Machine Learning\Frutas\data\fruits-360_original-size\fruits-360-original-size\Training"
class_names = sorted(os.listdir(train_dir))
print(f"✅ Clases detectadas: {len(class_names)} clases")

# Imagen de prueba
img_path = r"C:\Users\adrin\Downloads\-1262x640.jpg"

if not os.path.exists(img_path):
    print("❌ No se encontró la imagen.")
    exit()

img = image.load_img(img_path, target_size=(100, 100))
img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

pred = model.predict(img_array)
print("Forma de salida del modelo:", pred.shape)

predicted_class = class_names[np.argmax(pred)]
confidence = np.max(pred)

print(f"🔮 Predicción: {predicted_class} (confianza: {confidence:.2f})")
