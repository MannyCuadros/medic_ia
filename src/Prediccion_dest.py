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
#nuevos_datos = np.array([[36.5, 88.0, 102, 49, 98.0]])
nuevos_datos = np.array([[36.59, 80, 128, 57, 99]])
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

if (categoria[0][0] == 1):
    
    print("Grupo 3 -> Domicilio")
else:
    print("Sub grupo")
    ruta_modelo_sub = os.path.join(os.path.dirname(__file__), 'modelo', 'modelo_entrenado_sub.h5')
    ruta_escalador_sub = os.path.join(os.path.dirname(__file__), 'modelo', 'escalador_sub.pkl')


    # Cargar el modelo y el escalador
    model_sub = load_model(ruta_modelo_sub)
    scaler_sub = load(ruta_escalador_sub)

    # Ingresar los datos manuales (no escalados)
    nombre_datos_sub = ['__temperatura', '__pulso', '__pas', '__pad', '__sat02']
    nuevos_datos_sub = nuevos_datos
    datos_df_sub = pd.DataFrame(nuevos_datos_sub, columns=nombre_datos)
    print("Datos ingresados:")
    print(datos_df_sub)

    # Escalar los datos
    nuevos_datos_escalados_sub = scaler.transform(datos_df_sub)
    print("Datos escalados:")
    print(nuevos_datos_escalados_sub)

    # Realizar la predicción
    prediccion_sub = model.predict(nuevos_datos_escalados_sub)

    # Interpretar el resultado
    categoria_sub = np.argmax(prediccion_sub, axis=1) + 1

    print(f'Predicción: {prediccion_sub}')
    print(f'Categoría asignada: {categoria_sub}')  # 1, 2, 4, 5, 6

    print(categoria_sub[0])

    if (categoria_sub[0] == 1):
    
        print("Grupo 1 -> Carabineros o PDI")

    elif (categoria_sub[0] == 2):
    
        print("Grupo 2 -> Derivación")

    elif (categoria_sub[0] == 4):
    
        print("Grupo 4 -> Hospitalización")

    elif (categoria_sub[0] == 5):
    
        print("Grupo 5 -> Hospitalización domiciliaria")

    elif (categoria_sub[0] == 6):
    
        print("Grupo 5 -> Otros")

print("fin predicción")