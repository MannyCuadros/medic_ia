import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from joblib import load

# Ruta relativa desde 'Preduccion_dest.py' hacia el archivo del modelo
ruta_modelo = os.path.join(os.path.dirname(__file__), 'modelo', 'modelo_entrenado.h5')
ruta_escalador = os.path.join(os.path.dirname(__file__), 'modelo', 'escalador.pkl')


# Cargar el modelo y el escalador
model = load_model(ruta_modelo)
scaler = load(ruta_escalador)

# Ingresar los datos manuales (no escalados)
nombre_datos = ['__temperatura', '__pulso', '__pas', '__pad', '__sat02']
nuevos_datos = np.array([[36.5, 88.0, 102, 49, 98.0]])
datos_df = pd.DataFrame(nuevos_datos, columns=nombre_datos)
print("Datos ingresados:")
print(datos_df)

# Escalar los datos
nuevos_datos_escalados = scaler.transform(datos_df)
print("Datos escalados:")
print(nuevos_datos_escalados)

# Realizar la predicción
prediccion = model.predict(nuevos_datos_escalados)

# Interpretar el resultado
categoria = (prediccion > 0.5).astype(int)  # Redondea a 0 o 1

print(f'Predicción: {prediccion}')
print(f'Categoría asignada: {categoria[0][0]}')  # 0 o 1

print(categoria)