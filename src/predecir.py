import joblib
import json
import os
import pandas as pd
import distribucion_data as dd
import funciones_distribucion as fd
#import categorizar as cat
import archivos as files
from sklearn.preprocessing import FunctionTransformer
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def procesar_json(archivo):
    """
    Procesa un archivo JSON y devuelve un DataFrame de pandas.
    """
    columnas_ordenadas = ['ID', 'Sexo', 'Temperatura', 'Pulso', 'PAS', 'PAD', 'SatO2', 'Edad', 'Triaje']
    try:
        df = fd.leer_json(archivo, 'files')

        if df is None:
            print(f"Error: Archivo {archivo} vacío")
            return None
        
        else:
            
            # Reordenar las columnas
            columnas_existentes = [col for col in columnas_ordenadas if col in df.columns]
            df = df[columnas_existentes]
                
            return df
            
    except FileNotFoundError:
        print(f"Error: El archivo '{archivo}' no se encontró.")
        return None
    except json.JSONDecodeError:
        print(f"Error: El archivo '{archivo}' no es un JSON válido.")
        return None
        
def procesar_csv(archivo):
    """
    Procesa un archivo CSV y devuelve un DataFrame de pandas.
    """
    columnas_ordenadas = ['ID', 'Sexo', 'Temperatura', 'Pulso', 'PAS', 'PAD', 'SatO2', 'Edad', 'Triaje']
    try:
        df = fd.leer_csv(archivo, 'files', ";")
        
        if df is None:
            print(f"Error: Archivo {archivo} vacío")
            return None
        
        else:
            # Reordenar las columnas
            columnas_existentes = [col for col in columnas_ordenadas if col in df.columns]
            df = df[columnas_existentes]
            
            return df
        
    except FileNotFoundError:
        print(f"Error: El archivo '{archivo}' no se encontró.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: El archivo '{archivo}' está vacío.")
        return None

def leer_entrada(nombre_archivo):
    """
    Lee un archivo (JSON o CSV) desde la carpeta 'files' y devuelve un DataFrame de pandas.

    Parámetros:
    nombre_archivo (str): El nombre del archivo con su extensión.

    Devuelve:
    pd.DataFrame: Un DataFrame con los datos, o None si hay un error.
    """
    # Construir la ruta completa del archivo
    ruta_base = 'files'
    ruta_archivo = os.path.join(ruta_base, nombre_archivo)
    print(f"Leyendo archivo: '{ruta_archivo}'")
    
    # Verificar si el archivo existe y su extensión
    if not os.path.exists(ruta_archivo):
        print(f"Error: No se encontró el archivo '{nombre_archivo}' en la carpeta '{ruta_base}'.")
        return None
    
    nombre, extension = os.path.splitext(nombre_archivo)
    
    if extension.lower() == '.json':
        data_pacientes = procesar_json(nombre_archivo)
    elif extension.lower() == '.csv':
        data_pacientes = procesar_csv(nombre_archivo)
    else:
        print(f"Error: El tipo de archivo '{extension}' no es compatible. Solo se aceptan .json y .csv.")
        return None
    
    if data_pacientes is not None:
        return data_pacientes
    else:
        return None

def cargar_modelo(version = 1):
    """
    Carga el modelo y el label encoder
    Si version es 0, carga últimos generados en modelos_files.txt y encoders_files.txt
    Si version es 1, carga el modelo y enconder seleccionado en modelo.txt y encoder.txt
    Retorna: (modelo, label_encoder, mensaje_error)
    """

    if version == 0:
        nombre_modelo = files.get_first_file("modelo")
        nombre_encoder = files.get_first_file("encoder")
        print("Último generado:")
    if version == 1:
        nombre_modelo = files.get_modelo()
        nombre_encoder = files.get_encoder()
        print("Seleccionado:")

    ruta = 'modelos'
    ruta_modelo = os.path.join(ruta, nombre_modelo)
    ruta_encoder = os.path.join(ruta, nombre_encoder)
    
    try:
        modelo = joblib.load(ruta_modelo)
        label_encoder = joblib.load(ruta_encoder)
        print(f"Usando \033[1m{nombre_modelo}\033[0m y \033[1m{nombre_encoder}\033[0m para predecir")
        return modelo, label_encoder, None
    except FileNotFoundError:
        return None, None, "Archivo de modelo o encoder no encontrado"
    except Exception as e:
        return None, None, f"Error cargando modelo: {str(e)}"

def agregar_categoricos(df):
    df_nuevo = df.copy(deep=True)
    # Diccionario de mapeo: {columna_cat: (columna_origen, función)}
    categoricos = {
        'Temperatura_cat': ('Temperatura', dd.discretizar_temperatura),
        'Pulso_cat': ('Pulso', dd.discretizar_pulso),
        'PAS_cat': ('PAS', dd.discretizar_pas),
        'PAD_cat': ('PAD', dd.discretizar_pad),
        'SatO2_cat': ('SatO2', dd.discretizar_sat02),
        'Edad_cat': ('Edad', dd.discretizar_edad)
    }
    
    for col_cat, (col_origen, funcion) in categoricos.items():
        if col_origen in df_nuevo.columns:
            df_nuevo[col_cat] = df_nuevo[col_origen].apply(funcion)
        else:
            print(f"Advertencia: Columna {col_origen} no encontrada")
    
    return df_nuevo

def predecir_uno(paciente, modelo, label_encoder):
    id = paciente["ID"]
    datos = paciente.drop(columns=["ID"])

    columnas_modelo = ['Temperatura', 'Pulso', 'PAS', 'PAD', 'SatO2', 'Edad', 'Sexo', 'Triaje']
    datos_completos = datos[columnas_modelo]
    datos_completos = agregar_categoricos(datos)

    try:
        predicciones_encoded = modelo.predict(datos_completos)
        predicciones = label_encoder.inverse_transform(predicciones_encoded)
        respuesta = datos_completos.drop(
            columns=['Temperatura_cat', 'Pulso_cat', 'PAS_cat', 'PAD_cat', 'SatO2_cat', 'Edad_cat']
            ).copy()
        respuesta.insert(0, "ID", id)
        respuesta["Destino"] = predicciones
        return respuesta
    except Exception as e:
        print(f"Error: {e}")
        return None, f"Error en predicción: {str(e)}"

# Función para hacer predicciones
def predecir(nombre_archivo, version = 1):
    """
    Realiza predicciones con los datos de pacientes dentro
    del archivo paciente (para 1 paciente) o 
    pacientes (para varios pacientes en bloque)
    Retorna: (predicciones, mensaje_error)
    Solo se puede predecir cuando los campos en la interfaz
    están todos llenos y dentro de los límites válidos
    caso contrario debe emitirse una advertencia que no es posible
    """
    print("------------------------------------Predicción-----------------------------------")


    pacientes = leer_entrada(nombre_archivo)
    if pacientes is None:
        print("Error: No se pudo realizar la predicción")
        return None
    elif not pacientes.empty:
        print("Datos de pacientes:")
        print(pacientes)
        modelo, label_encoder, error = cargar_modelo(version)
        if error:
            print(f"Error: {error}")
            return None, "Error en predicción"
        cant_pacientes = len(pacientes)
        print(f"\033[0mSe hará la predicción de \033[96m\033[1m{cant_pacientes} \033[0mpaciente(s)")
        
        try:
            if cant_pacientes == 0:
                error = "No existen pacientes que predecir"
                return None, error
            
            elif cant_pacientes == 1: 
                prediccion_df = predecir_uno(pacientes, modelo, label_encoder)
                print(prediccion_df)

                #guardar resultado en el archivo destino.json
                fd.guardar_json(prediccion_df, 'destino.json', 'files')           
        
                return prediccion_df, None
            
            else:
                resultados = []  # Lista para guardar cada fila con predicción

                for i in range(cant_pacientes):
                    paciente = pacientes.iloc[i:i+1]  # Extrae la fila como series
                    pred_df = predecir_uno(paciente, modelo, label_encoder)  # Llama a tu función
                    if isinstance(pred_df, tuple):  # Si devuelve (None, error)
                        print(f"Error en fila {i}: {pred_df[1]}")
                        continue
                    resultados.append(pred_df)

                # Concatenar todas las filas procesadas
                df_resultados = pd.concat(resultados, ignore_index=True)

                fd.guardar_json(df_resultados, 'destinos.json', 'files')
                print(f"\033[0mSe hizo la predicción de \033[96m\033[1m{len(df_resultados)} \033[0mpaciente(s)")

                return df_resultados, None

        except Exception as e:
            return None, f"Error en predicción: {str(e)}"
    
# Ejecutar archivo para hacer uso del programa:
if __name__ == "__main__":
    
    resultado = predecir('paciente.json', 0)
    