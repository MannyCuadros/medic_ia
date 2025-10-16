import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import archivos as files
from datetime import datetime


# Leer archivo CSV y guardarlo en dataframe
def leer_csv(nombre_archivo, carpeta="databases", separator=","):
    
    ruta_archivo = os.path.join(carpeta, nombre_archivo)
    tipos_datos = {'ID': str}
    
    try:
        # Intenta leer el archivo
        df = pd.read_csv(
            ruta_archivo,
            sep=separator,
            low_memory=False,
            dtype=tipos_datos)
        
        if 'ID' in df.columns:
        # Si la columna existe, la convertimos explícitamente a string (dtype 'object')
            df['ID'] = df['ID'].astype(str)
        
        # Mensaje de éxito con formato
        print(f"\033[92mÉxito:\033[0m Archivo '\033[1m{nombre_archivo}\033[0m' leído correctamente")
        return df
        
    except Exception as e:
        # Mensaje de error detallado
        print(f"\n\033[91mError:\033[0m No se pudo leer el archivo '\033[1m{nombre_archivo}\033[0m'")
        print(f"\033[94mRuta intentada:\033[0m {ruta_archivo}. \033[94mTipo de error:\033[0m {type(e).__name__}. \033[94mDetalles:\033[0m {str(e)}\n")
        return None
    
# Guardar dataframe en archivo CSV
def guardar_csv(df, nombre_archivo, carpeta="databases", separator=",", index=False, encoding='utf-8'):
    
    ruta_archivo = os.path.join(carpeta, nombre_archivo)
    
    try:
        # Crear directorio si no existe
        os.makedirs(carpeta, exist_ok=True)
        
        # Guardar el DataFrame
        df.to_csv(
            ruta_archivo,
            sep=separator,
            index=index,
            encoding=encoding
        )
        
        # Mensaje de éxito
        print(f"\033[92mÉxito:\033[0m DataFrame guardado en '\033[1m{nombre_archivo}\033[0m' en la carpeta \033[94m{carpeta}\033[0m")
        return True
        
    except Exception as e:
        # Mensaje de error detallado
        print(f"\n\033[91mError:\033[0m No se pudo guardar el archivo '\033[1m{nombre_archivo}\033[0m'")
        print(f"\033[94mRuta intentada:\033[0m {ruta_archivo}")
        print(f"\033[94mTipo de error:\033[0m {type(e).__name__}")
        print(f"\033[94mDetalles:\033[0m {str(e)}\n")
        return False

#Leer archivo json y guardarlo en dataframe
def leer_json(nombre_archivo, carpeta="databases"):
    
    ruta_archivo = os.path.join(carpeta, nombre_archivo)
    
    try:
        # Intenta leer el archivo
        with open(ruta_archivo, 'r') as f:
            datos_json = json.load(f)
            # Asegurar que los datos sean una lista de diccionarios para consistencia
            if isinstance(datos_json, dict):
                datos_json = [datos_json]
            df = pd.DataFrame(datos_json)
        
        # Mensaje de éxito con formato
        print(f"\033[92mÉxito:\033[0m Archivo '\033[1m{nombre_archivo}\033[0m' leído correctamente")
        return df
        
    except Exception as e:
        # Mensaje de error detallado
        print(f"\n\033[91mError:\033[0m No se pudo leer el archivo '\033[1m{nombre_archivo}\033[0m'")
        print(f"\033[94mRuta intentada:\033[0m {ruta_archivo}. \033[94mTipo de error:\033[0m {type(e).__name__}. \033[94mDetalles:\033[0m {str(e)}\n")
        return None
    
# Guardar dataframe en archivo json
def guardar_json(data, nombre_archivo, carpeta="databases", index=False, orient='records'):
    
    ruta_archivo = os.path.join(carpeta, nombre_archivo)
    
    try:
        # Crear directorio si no existe
        os.makedirs(carpeta, exist_ok=True)
        
        # Guardar el DataFrame
        if isinstance(data, pd.DataFrame):
            data.to_json(
                path_or_buf=ruta_archivo,
                orient=orient, # 'records' es un formato común (lista de diccionarios)
                index=index,   # Controla si el índice se incluye o no
                indent=4,       # Para una salida JSON formateada y legible
                force_ascii=False
            )
        # Guardar el diccionario
        elif isinstance(data, (dict,list)):
            with open(ruta_archivo, 'w') as f:
                json.dump(
                    data, 
                    f, 
                    indent=4, 
                    ensure_ascii=False # Equivalente a force_ascii=False
                )
        else:
            raise TypeError("El parámetro 'data' debe ser un pandas DataFrame, diccionario o lista.")
        
        # Mensaje de éxito
        print(f"\033[92mÉxito:\033[0m DataFrame guardado en '\033[1m{nombre_archivo}\033[0m' en la carpeta \033[94m{carpeta}\033[0m")
        return True
        
    except Exception as e:
        # Mensaje de error detallado
        print(f"\n\033[91mError:\033[0m No se pudo guardar el archivo '\033[1m{nombre_archivo}\033[0m'")
        print(f"\033[94mRuta intentada:\033[0m {ruta_archivo}")
        print(f"\033[94mTipo de error:\033[0m {type(e).__name__}")
        print(f"\033[94mDetalles:\033[0m {str(e)}\n")
        return False
    

def guardar_basedatos(df, nombre_archivo = "basedatos", carpeta = "databases", version = None):
    
    print("-------------------------------Guardar Base de Datos-------------------------------")

    version = version or datetime.now().strftime("%Y%m%d%H%M")
    nombre_archivo = nombre_archivo + '_' + version + '.csv'
    try:
        # Crear directorio si no existe
        ruta_archivo = os.path.join(carpeta, nombre_archivo)
        os.makedirs(carpeta, exist_ok=True)
        
        # Guardar el DataFrame
        df.to_csv(ruta_archivo, sep= ",", index=False, encoding='utf-8')
        
        # Mensaje de éxito
        print(f"\033[92mÉxito:\033[0m Base de datos guardada en '\033[94m{ruta_archivo}\033[0m'")
        
        files.update_base_datos_list()
        files.save_basedatos(nombre_archivo)  
    except Exception as e:
        # Mensaje de error detallado
        print(f"\n\033[91mError:\033[0m No se pudo guardar el archivo '\033[1m{nombre_archivo}\033[0m'")
        print(f"\033[94mRuta intentada:\033[0m {ruta_archivo}")
        print(f"\033[94mTipo de error:\033[0m {type(e).__name__}")
        print(f"\033[94mDetalles:\033[0m {str(e)}\n")
        return False


# Obtener los nombres de las columnas
def obtener_nombres_columnas(dataframe):
    return dataframe.columns.tolist()

# Crear un DataFrame filtrado por columnas
def filtrar_por_columnas(dataframe, nombres_columnas):
    columnas_validas = []
    
    for col in nombres_columnas.keys():
        if col in dataframe.columns:
            columnas_validas.append(col)
    
    df_selecto = dataframe[columnas_validas].copy()

    df_selecto.rename(columns=nombres_columnas, inplace=True)
    
    return df_selecto


def determinar_tipo_columna(columna):
    """Determina si una columna es categórica o numérica y su subtipo"""
    tipo = 'Categórico'
    
    # Verificar si es numérico
    if pd.api.types.is_numeric_dtype(columna):
              
        # Verificar si es entero o decimal
        if pd.api.types.is_integer_dtype(columna):
            tipo = 'Entero'
        else:
            # Verificar si todos los valores son enteros (aunque sea float)
            if (columna.dropna() % 1 == 0).all():
                tipo = 'Entero'
            else:
                tipo = 'Decimal'
                
    # Verificar si es categórico
    else:

        if pd.api.types.is_string_dtype(columna):
            tipo = 'Categórico' 
        
    return tipo

def calcular_datos_faltantes(columna):
    """Calcula métricas de completitud de la columna"""
    total = len(columna)
    no_nulos = columna.count()
    no_nulos_per = round((no_nulos / total) * 100, 2)
    nulos = total - no_nulos
    nulos_per = round((nulos / total) * 100, 2)

    #return [total, no_nulos, no_nulos_per, nulos, nulos_per]
    return {
        'total_registros': total,
        'no_nulos': no_nulos,
        'porcentaje_no_nulos': round((no_nulos / total) * 100, 2) if total > 0 else 0,
        'nulos': nulos,
        'porcentaje_nulos': round((nulos / total) * 100, 2) if total > 0 else 0
    }

def analizar_valores_categoricos(columna):
    """Analiza valores únicos y sus frecuencias para columnas categóricas"""
    if pd.api.types.is_numeric_dtype(columna):
        return {}
        
    conteo = columna.value_counts(dropna=False)
    porcentajes = columna.value_counts(normalize=True, dropna=False) * 100
    
    return {
        'valores_unicos': len(conteo),
        'valores': {
            str(valor): {
                'conteo': int(conteo[valor]),
                'porcentaje': float(round(porcentajes[valor], 2))
            } for valor in conteo.index
        }
    }

def analizar_valores_numericos(columna):
    """Calcula estadísticas básicas para columnas numéricas"""
    if not pd.api.types.is_numeric_dtype(columna):
        return {}
    
    return {
        'minimo': float(round(columna.min(), 2)) if not columna.empty else None,
        'maximo': float(round(columna.max(), 2)) if not columna.empty else None,
        'media': float(round(columna.mean(), 2)) if not columna.empty else None,
        'mediana': float(round(columna.median(), 2)) if not columna.empty else None,
        'moda': float(round(columna.mode().iloc[0],2)) if not columna.empty else None
    }

def analizar_dataframe(df):
    """Función principal que integra todos los análisis"""
    analisis = {}
    
    for columna in df.columns:
        # Análisis básico de la columna
        analisis[columna] = {
            'tipo': determinar_tipo_columna(df[columna]),
            'completitud': calcular_datos_faltantes(df[columna]),
        }
        
        # Análisis específico según el tipo
        if analisis[columna]['tipo'] == 'Categórico':
            analisis[columna].update(analizar_valores_categoricos(df[columna]))
        else:
            analisis[columna].update(analizar_valores_numericos(df[columna]))
    
    return analisis

def visualizar_data(df):
    resultado_analisis = analizar_dataframe(df)

    # Acceder a la información
    for columna, datos in resultado_analisis.items():
        print(f"\n=== Análisis de columna: {columna} ===")
        print(f"Tipo de dato: {datos['tipo']}")
            
        if datos['tipo'] == 'Categórico':
            print(f"Valores únicos: {datos['valores_unicos']}")
            conteos = {clave: valor['conteo'] for clave, valor in datos['valores'].items()}
            print(str(conteos))
        else:
            print(f"Rango: [{datos['minimo']} - {datos['maximo']}]")


def mostrar_estadisticas(ruta_carpeta = "estadisticas", nombre_archivo = "estadística.json", nombre_grafico = "Graficos", formato = "png"):

    # Leer el archivo JSON exportado
    dir = ruta_carpeta + "/" + nombre_archivo
    with open(dir, "r", encoding="utf-8") as f:
        data_export = json.load(f)

    # Obtener la lista de variables (mediciones)
    measurements = list(data_export.keys())
    n = len(measurements)
    # Configurar una cuadrícula de subplots (por ejemplo, 2 columnas)
    ncols = 2
    nrows = (n + 1) // 2  # redondea hacia arriba en caso de número impar


    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axs = axs.flatten()  # para iterar fácilmente sobre todos los ejes

    for i, measurement in enumerate(measurements):
        ax = axs[i]
        items = data_export[measurement]
        # Extraer categorías, porcentajes y cuentas
        categorias = [item["categoria"] for item in items]
        porcentajes = [item["porcentaje"] for item in items]
        cantidad = [item["cantidad"] for item in items]
        
        # Generar una lista de colores distintos usando el colormap 'tab20'
        cmap = plt.cm.get_cmap('tab20', len(categorias))
        colores = [cmap(j) for j in range(len(categorias))]
        
        # Crear el gráfico de barras basado en el porcentaje, asignando un color distinto a cada barra
        bars = ax.bar(categorias, porcentajes, color=colores)
        ax.set_title(f"{measurement}")
        ax.set_ylabel("Porcentaje")
        ax.set_xlabel("Categorías")
        # Ajustar límite superior del eje y para dejar espacio a las anotaciones
        ax.set_ylim(0, max(porcentajes)*1.2 if max(porcentajes) > 0 else 1)
        
        # Agregar anotación: "XX.XX% (cuenta)" sobre cada barra
        for bar, pct, cnt in zip(bars, porcentajes, cantidad):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.5,
                f"{pct:.2f}% ({cnt})",
                ha='center',
                va='bottom',
                fontsize=9
            )

    # Eliminar ejes vacíos en caso de que la cuadrícula tenga más subplots de los necesarios
    for j in range(i+1, nrows*ncols):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

    version = version or datetime.now().strftime("%Y%m%d%H%M")
    archivo_grafico = nombre_grafico + '_' + version + "." + formato
    fig.savefig(os.path.join(ruta_carpeta, archivo_grafico))
    plt.close()