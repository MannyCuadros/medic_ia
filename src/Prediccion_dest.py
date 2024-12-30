import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from joblib import load
import argparse

# Ruta relativa desde 'Preduccion_dest.py' hacia el archivo del modelo
ruta_modelo = os.path.join(os.path.dirname(__file__), 'modelo', 'modelo_entrenado.h5')
ruta_escalador = os.path.join(os.path.dirname(__file__), 'modelo', 'escalador.pkl')

# Cargar el modelo y el escalador
model = load_model(ruta_modelo)
scaler = load(ruta_escalador)

# predecir 1 lista de valores
# los parámetros de entrada deben ser en forma de diccionario 
# el diccionario tiene esta forma:
# {'ID': 'U0003670130', '__temperatura': 36.5, '__pulso': 88.0, '__pas': 102.0, '__pad': 49.0, '__sat02': 98.0}
def predecir(lista_datos): 
    
    print("-------------------------------------------------------------------------------")
    # Extraer el ID y convierte los valores en un dataframe
    id_valor = lista_datos.pop('ID')
    datos_df = pd.DataFrame([lista_datos])

    # Escalar los datos
    datos_escalados = scaler.transform(datos_df)

    # Realizar la predicción
    prediccion = model.predict(datos_escalados)

    # Interpretar el resultado
    categoria = (prediccion > 0.5).astype(int)  # Redondea a 0 o 1
    categoria_valor = categoria[0][0]

    print(f'Predicción: {prediccion}')
    #print(f'Categoría asignada: {categoria_valor}')  # 0 o 1

    if (categoria_valor == 1):
        
        print("Grupo 3 -> Domicilio")
        lista_datos["__destino"] = "Domicilio"
    else:
        print("Sub grupo")
        ruta_modelo_sub = os.path.join(os.path.dirname(__file__), 'modelo', 'modelo_entrenado_sub.h5')
        ruta_escalador_sub = os.path.join(os.path.dirname(__file__), 'modelo', 'escalador_sub.pkl')

        # Cargar el modelo y el escalador
        model_sub = load_model(ruta_modelo_sub)
        scaler_sub = load(ruta_escalador_sub)

        # Escalar los datos
        datos_escalados_sub = scaler_sub.transform(datos_df)

        # Realizar la predicción
        prediccion_sub = model_sub.predict(datos_escalados_sub)

        # Interpretar el resultado
        categoria_sub = np.argmax(prediccion_sub, axis=1) + 1
        categoria_valor_sub = categoria_sub[0]

        print(f'Predicción: {prediccion_sub}')
        print(f'Categoría asignada: {categoria_valor_sub}')  # 1, 2, 4, 5, 6

        if (categoria_valor_sub == 1):
        
            print("Grupo 1 -> Carabineros o PDI")
            lista_datos["__destino"] = "Carabineros o PDI"

        elif (categoria_valor_sub == 2):
        
            print("Grupo 2 -> Derivación")
            lista_datos["__destino"] = "Derivación"

        elif (categoria_valor_sub == 4):
        
            print("Grupo 4 -> Hospitalización")
            lista_datos["__destino"] = "Hospitalización"

        elif (categoria_valor_sub == 5):
        
            print("Grupo 5 -> Hospitalización domiciliaria")
            lista_datos["__destino"] = "Hospitalización domiciliaria"

        elif (categoria_valor_sub >= 6):
        
            print("Grupo 6 -> Otros")
            lista_datos["__destino"] = "Otros"

    lista_datos = {'ID': id_valor, **lista_datos}

    return(lista_datos)


def predecir_varios(listas_datos, nombre):

    predicciones = {nombre: []}

    for fila in range(len(listas_datos)):

        lista = predecir(listas_datos[fila])

        predicciones[nombre].append(lista)

    return(predicciones)


def main():

    # Uso de argumentos para la predicción de un elemento o varios
    parser = argparse.ArgumentParser(description="Predicción con uno o varios pacientes")
    parser.add_argument(
        "--action", 
        choices=["u", "v"],  # Opciones válidas
        default="u",         # Valor por defecto si no se pasa ningún argumento
        help="'u' para un paciente y 'v' para varios."
    )
    args = parser.parse_args()

    # Leer el archivo con los datos a predecir
    ruta_input = os.path.join(os.path.dirname(__file__), 'files', 'input.json')
    with open(ruta_input, 'r') as file:
        data = json.load(file)
    
    nombre_data = list(data.keys())[0]

    # Extraer los datos de pacientes
    pacientes = data['pacientes']

    if args.action == "u":

        print("Predecir destino paciente")
        salida = predecir(pacientes[0])
        print(salida)
        print("---Fin predicción---")

        #carpeta_salida = "files"
        #os.makedirs(carpeta_salida, exist_ok=True)

        # Ruta completa del archivo
        archivo_salida = os.path.join(os.path.dirname(__file__), 'files', "paciente.json")

        # Guardar en un archivo JSON
        with open(archivo_salida, "w") as archivo:
            json.dump(salida, archivo, indent=4)  # Guardar con formato legible

        print(f"Datos guardados en {archivo_salida} con éxito.")

    elif args.action == "v":
        
        print("Predecir destino varios pacientes")
        salida_varios = predecir_varios(pacientes, nombre_data)
        print(salida_varios)
        print("---Fin predicción---")

        # Ruta completa del archivo
        archivo_salida = os.path.join(os.path.dirname(__file__), 'files', "output.json")

        # Guardar en un archivo JSON
        with open(archivo_salida, "w") as archivo:
            json.dump(salida_varios, archivo, indent=4)  # Guardar con formato legible

        print(f"Datos guardados en {archivo_salida} con éxito.")

if __name__ == "__main__":
    main()






