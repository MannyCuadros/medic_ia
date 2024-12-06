import numpy as np
from tensorflow.keras.models import load_model
from joblib import load


# Cargar el modelo y el escalador
model = load_model('/home/manny/workspaces/medic_ia/src/modelo/modelo_entrenado.h5')
scaler = load('/home/manny/workspaces/medic_ia/src/modelo/escalador.pkl')

# Ingresar los datos manuales (no escalados)
nuevos_datos = np.array([[36.5, 88.0, 102, 49, 98.0]])
print("Datos ingresados:")
print(nuevos_datos)

# Escalar los datos
nuevos_datos_escalados = scaler.transform(nuevos_datos)
print("Datos escalados:")
print(nuevos_datos_escalados)

# Realizar la predicción
prediccion = model.predict(nuevos_datos_escalados)

# Interpretar el resultado
categoria = (prediccion > 0.5).astype(int)  # Redondea a 0 o 1

print(f'Predicción: {prediccion}')
print(f'Categoría asignada: {categoria[0][0]}')  # 0 o 1

print(categoria)