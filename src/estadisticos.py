import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import seaborn as sns
import math
import funciones_distribucion as fd
import archivos as files
from datetime import datetime
from pathlib import Path



def guardar_estadisticas(df, nombre_archivo = "estadística", carpeta = "estadistica", version = None):
    """
    Calcula frecuencias y porcentajes para columnas categóricas,
    imprime los resultados y los exporta a un archivo JSON.
    """
    resultados = {}

    analisis = fd.analizar_dataframe(df)
    
    for columna, datos in analisis.items():
        tipo = datos['tipo']
        if tipo == 'Categórico':
            # Procesar datos categóricos
            valores = []
            for valor, stats in datos['valores'].items():
                valores.append({
                    'categoria': valor if valor != 'nan' else None,
                    'cantidad': int(stats['conteo']),
                    'porcentaje': round(stats['porcentaje'], 2)
                })
                
            resultados[columna] = {
                'tipo': tipo,
                'datos': valores
            }
            
        else:
            # Procesar datos numéricos
            resultados[columna] = {
                'tipo': tipo,
                'datos': [{
                    'mínimo': round(datos['minimo'], 2),
                    'máximo': round(datos['maximo'], 2),
                    'media': round(datos['media'], 2),
                    'mediana': round(datos['mediana'], 2),
                    'moda': round(datos['moda'], 2)
                }]
            }
        
    # Exportar a JSON
    version = version or datetime.now().strftime("%Y%m%d%H%M")
     
    try:
        # Crear directorio si no existe
        os.makedirs(carpeta, exist_ok=True)
        
        nombre_archivo = nombre_archivo + '_' + version + '.json'
        ruta_archivo = carpeta + '/' + nombre_archivo
        with open(ruta_archivo, 'w', encoding='utf-8') as f:
            json.dump(resultados, f, indent=4, ensure_ascii=False)
        
        # Mensaje de éxito
        print(f"\033[92mÉxito:\033[0m Estadísticas guardadas en '\033[94m{ruta_archivo}\033[0m'")
        #return {nombre_archivo + version + 'json', resultados}
        
        files.update_estadisticas_list()

        return {nombre_archivo: resultados}
        
    except Exception as e:
        # Mensaje de error detallado
        print(f"\n\033[91mError:\033[0m No se pudo guardar el archivo '\033[1m{nombre_archivo}\033[0m'")
        print(f"\033[94mRuta intentada:\033[0m {ruta_archivo}")
        print(f"\033[94mTipo de error:\033[0m {type(e).__name__}")
        print(f"\033[94mDetalles:\033[0m {str(e)}\n")
        return None


def generar_graficos(nombre_archivo, df=None, carpeta = "estadisticas", nombre_grafico = "graficos", formato = "png", version = None):
    """
    Genera gráficos a partir de un archivo JSON de estadísticas:
    - Histogramas para variables categóricas
    - Curvas de densidad con estadísticas para variables numéricas
    """
    # Leer datos del JSON
    ruta_archivo = os.path.join(carpeta, nombre_archivo)
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        datos = json.load(f)
    
    n_plots = len(datos)
    n_cols = math.ceil(math.sqrt(n_plots))
    n_rows = math.ceil(n_plots / n_cols)
    
    fig, axs = plt.subplots(5, 3, figsize=(n_cols*7, n_rows*7))
    axs = axs.flatten()  # Aplanar para acceso secuencial


    # Configuración estética
    sns.set_style("whitegrid")
    colores_estadisticas = {
        'minimo': '#FF0000',    # Claves sin acentos
        'maximo': '#00FF00',
        'media': '#0000FF',
        'moda': '#FF00FF',
        'mediana': '#FFA500'
    }
    
    for idx, (columna, info) in enumerate(datos.items()):
        ax = axs[idx]
        tipo = info['tipo']
        datos_columna = info['datos']
        
        if tipo == 'Categórico':
            # Gráfico de barras vertical
            categorias = [item['categoria'] or 'NaN' for item in datos_columna]
            porcentajes = [item["porcentaje"] for item in datos_columna]
            cantidad = [item["cantidad"] for item in datos_columna]

            # Generar una lista de colores distintos usando el colormap 'tab20'
            cmap = plt.cm.get_cmap('tab20', len(categorias))
            colores = [cmap(j) for j in range(len(categorias))]
            
            # Crear el gráfico de barras basado en el porcentaje, asignando un color distinto a cada barra
            bars = ax.bar(categorias, porcentajes, color=colores)
            ax.set_title(f"{columna}", pad=12)
            ax.set_ylabel("Porcentaje")
            ax.set_xlabel("")
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

        else:
            if df is not None and columna in df:
                serie = pd.to_numeric(df[columna], errors='coerce').dropna()
                    
                if serie.empty:
                    continue  # Saltar si no hay datos numéricos
                
                sns.histplot(df[columna], ax=ax, kde=True,
                            color='#1f77b4', stat='density', bins=30,
                            alpha=0.3, linewidth=0.5)
                
                sns.kdeplot(df[columna], ax=ax, 
                           color='darkblue', linewidth=2)
                
                # Estadísticas
                stats = datos_columna[0]
                for stat, valor in stats.items():
                # Normalizar nombre de estadística
                    stat_normalizado = (
                        stat.lower()
                        .replace('í', 'i')
                        .replace('á', 'a')
                        .replace('é', 'e')
                    )
                    
                    if stat_normalizado in colores_estadisticas and valor is not None:
                        ax.axvline(
                            valor, 
                            color=colores_estadisticas[stat_normalizado],
                            linestyle='--', 
                            linewidth=1.5,
                            label=f"{stat}: {valor:.2f}"
                        )
                
                ax.set_title(f"{columna}", fontsize=12)
                ax.legend(fontsize=8)
                ax.set_xlabel("Valores", fontsize=10)
                ax.set_ylabel("Densidad", fontsize=10)
        
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
    
    # Ocultar ejes vacíos
    for ax in axs[len(datos):]:
        ax.axis('off')
    
    # Guardar gráfico
    try:
        version = version or datetime.now().strftime("%Y%m%d%H%M")      
        nombre_grafico = nombre_grafico + '_' + version + '.' + formato
        ruta_archivo = os.path.join(carpeta, nombre_grafico)
        plt.tight_layout(pad=3.0)
        fig.savefig(ruta_archivo, dpi=300, bbox_inches='tight')
        plt.close()
        # Mensaje de éxito
        print(f"\033[92mÉxito:\033[0m Gráficos estadísticos guardados en '\033[94m{ruta_archivo}\033[0m'")
        
        files.update_graficos_list()

    except Exception as e:
        # Mensaje de error detallado
        print(f"\n\033[91mError:\033[0m No se pudo guardar el archivo '\033[1m{ruta_archivo}\033[0m'")
        print(f"\033[94mRuta intentada:\033[0m {ruta_archivo}")
        print(f"\033[94mTipo de error:\033[0m {type(e).__name__}")
        print(f"\033[94mDetalles:\033[0m {str(e)}\n")
    

def generar_estadisticas(df = None, version = None):

    print("-----------------------------------Estadísticas------------------------------------")

    if df is None:
        archivo_estadisticas = files.get_first_file("basedatos")
        #archivo_estadisticas = files.get_basedatos()
        print(f"Estadisticas generadas del archivo {archivo_estadisticas}")
    
        df = fd.leer_csv(archivo_estadisticas)
    else:
        df = df
        print("Estadisticas generadas de dataframe")
    
    resultado = guardar_estadisticas(df, nombre_archivo = "estadística", carpeta = "estadisticas", version = version)
    estadistico = list(resultado.keys())[0]
    generar_graficos(estadistico, df, version = version)

