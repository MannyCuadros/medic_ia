import os
import json
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
#nuevos_datos = np.array([[36.59, 80, 128, 57, 99]])
datos_df = pd.DataFrame(nuevos_datos, columns=nombre_datos)
print("Datos ingresados:")
print(datos_df)

# Escalar los datos
nuevos_datos_escalados = scaler.transform(datos_df)

# Realizar la predicción
prediccion = model.predict(nuevos_datos_escalados)

# Interpretar el resultado
categoria = (prediccion > 0.5).astype(int)  # Redondea a 0 o 1
categoria_valor = categoria[0][0]

print(f'Predicción: {prediccion}')
#print(f'Categoría asignada: {categoria_valor}')  # 0 o 1


lista_datos = nuevos_datos.flatten().tolist()
# Crear el diccionario con los valores
datos = {nombre: valor for nombre, valor in zip(nombre_datos, lista_datos)}  # Asociar nombres y valores


if (categoria_valor == 1):
    
    print("Grupo 3 -> Domicilio")
    datos["__destino"] = "Domicilio"
else:
    print("Sub grupo")
    ruta_modelo_sub = os.path.join(os.path.dirname(__file__), 'modelo', 'modelo_entrenado_sub.h5')
    ruta_escalador_sub = os.path.join(os.path.dirname(__file__), 'modelo', 'escalador_sub.pkl')


    # Cargar el modelo y el escalador
    model_sub = load_model(ruta_modelo_sub)
    scaler_sub = load(ruta_escalador_sub)

    # Escalar los datos
    nuevos_datos_escalados_sub = scaler_sub.transform(datos_df)

    # Realizar la predicción
    prediccion_sub = model_sub.predict(nuevos_datos_escalados_sub)

    # Interpretar el resultado
    categoria_sub = np.argmax(prediccion_sub, axis=1) + 1
    categoria_valor_sub = categoria_sub[0]

    print(f'Predicción: {prediccion_sub}')
    print(f'Categoría asignada: {categoria_valor_sub}')  # 1, 2, 4, 5, 6

    if (categoria_valor_sub == 1):
    
        print("Grupo 1 -> Carabineros o PDI")
        datos["__destino"] = "Carabineros o PDI"

    elif (categoria_valor_sub == 2):
    
        print("Grupo 2 -> Derivación")
        datos["__destino"] = "Derivación"

    elif (categoria_valor_sub == 4):
    
        print("Grupo 4 -> Hospitalización")
        datos["__destino"] = "Hospitalización"

    elif (categoria_valor_sub == 5):
    
        print("Grupo 5 -> Hospitalización domiciliaria")
        datos["__destino"] = "Hospitalización domiciliaria"

    elif (categoria_valor_sub == 6):
    
        print("Grupo 5 -> Otros")
        datos["__destino"] = "Otros"


carpeta_salida = "files"
os.makedirs(carpeta_salida, exist_ok=True)

# Ruta completa del archivo
archivo_salida = os.path.join(os.path.dirname(__file__), 'files', "datos.json")
print(datos)

# Guardar en un archivo JSON
with open(archivo_salida, "w") as archivo:
    json.dump(datos, archivo, indent=4)  # Guardar con formato legible

print(f"Datos guardados en {archivo_salida} con éxito.")

print("---Fin predicción---")