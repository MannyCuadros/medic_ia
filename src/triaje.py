import os
import pandas as pd
import distribucion_data as dd
import funciones_distribucion as fd
import categorizar as cat
import archivos as files
from sklearn.preprocessing import FunctionTransformer
from scipy.sparse import csr_matrix

def cargar_basedatos(version):
    
    dir = 'databases'
    
    if version == 0:
        base_datos = files.get_first_file("basedatos")
    else:
        base_datos = files.get_basedatos()
    
    df = fd.leer_csv(base_datos)

    return df

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

def buscar_pacientes(df_basedatos, datos_entrada):
    try:
        # 1. Generar columnas categóricas en los nuevos datos
        print("Datos de entrada")
        print(datos_entrada)
        pacientes_df = agregar_categoricos(datos_entrada).copy(deep=True)
        print("Datos + categorías")
        print(pacientes_df)
        pacientes_df_num = pacientes_df.drop(columns=[
            'Temperatura_cat', 
            'Pulso_cat', 
            'PAS_cat', 
            'PAD_cat', 
            'SatO2_cat', 
            'Edad_cat'
            ])
        
        cols_a_convertir = [col for col in pacientes_df_num.columns if col != "Sexo"]
        pacientes_df_num[cols_a_convertir] = pacientes_df_num[cols_a_convertir].astype(float)
        
        # Definimos la precisión de redondeo para cada columna numérica:
        precision_round = {
            'Temperatura': 1,  # <--- CLAVE: Redondeo a un solo decimal (ej. 36.29 -> 36.3)
            'Pulso': 0,        # Pulso, PAS, etc., normalmente son números enteros
            'PAS': 0, 
            'PAD': 0, 
            'SatO2': 0, 
            'Edad': 0
        }
        
        # Redondear los DATOS DE ENTRADA
        for col, dec in precision_round.items():
            if col in pacientes_df_num.columns:
                pacientes_df_num[col] = pacientes_df_num[col].round(dec)
                
        # Redondear la BASE DE DATOS antes de usarla en el merge
        # Copia temporal de la base de datos para no alterarla
        df_basedatos_temp = df_basedatos.copy() 
        for col, dec in precision_round.items():
            if col in df_basedatos_temp.columns:
                # Asegurar que la columna es float antes de redondear y aplicar el redondeo
                df_basedatos_temp[col] = df_basedatos_temp[col].astype(float).round(dec)
                
        # 2. Definir columnas categóricas relevantes (excluyendo Triaje)
        
        resultados_num = pd.merge(
            df_basedatos,
            pacientes_df_num,
            on=pacientes_df_num.columns.tolist(),
            how='inner'
        )

        if len(resultados_num) != 0:
            print("----------------------------------------")
            print(f"Cantidad de similitudes = {len(resultados_num)}")
            print("----------------------------------------")
            print(resultados_num)
            return resultados_num
        else:
            print("----------------------------------------")
            print("No se encontraron similitudes numéricas. \nProcediento a buscar similitudes categóricas:")
            pacientes_df_cat = pacientes_df.drop(columns=[
                'Temperatura', 
                'Pulso', 
                'PAS', 
                'PAD', 
                'SatO2', 
                'Edad'
                ])
            
            resultados_cat = pd.merge(
            df_basedatos,
            pacientes_df_cat,
            on=pacientes_df_cat.columns.tolist(),
            how='inner'
            )

            if len(resultados_cat) != 0:
                print("----------------------------------------")
                print(f"Cantidad de similitudes = {len(resultados_cat)}")
                print("----------------------------------------")
                print(resultados_cat)
                return resultados_cat
            else:
                print("No se encontraron similitudes")
                return None
    
    except Exception as e:
        print(f"Error en búsqueda: {str(e)}")
        return pd.DataFrame()

def calcular_triaje(df_filtrado):
   
    # Definir el diccionario de mapeo
    mapping = {'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5}
    
    # Convertir los valores de 'Triaje' a números
    triaje_numerico = df_filtrado['Triaje'].map(mapping)
    # Calcular el promedio (asegurándose de ignorar valores nulos)
    promedio = triaje_numerico.mean()   
    #return math.floor(promedio)
    
    # Retornar el promedio redondeado a 0 decimales
    return round(promedio)

def agregar_triaje(df, version = 1):

    basedatos = cargar_basedatos(version)

    filas_modificadas = []
    lista_categorias = []

    for i in range(df.shape[0]):
        paciente = df.iloc[i:i+1].copy()
        indice = paciente.index[0] 

        # Busca al paciente en la base de datos
        # En caso que no se encuentre con son valores numéricos
        # Se buscará por sus categorías
        pacientes = buscar_pacientes(basedatos,paciente)

        # Calcula el valor del triaje
        # En caso que se encontraron similitudes, calcula el valor promedio
        # En caso que no exista similitudes, se emplea el método de scoring
        if pacientes is not None:
            print("----------------------------------------")
            valor_triaje = calcular_triaje(pacientes)
            cat_triaje = 'C' + str(valor_triaje)
        else:
            nuevo_df = paciente.drop(columns=['Sexo'])
            categoria = cat.categorizar(nuevo_df)
            cat_triaje = categoria
        
        print(f"Triaje correspondiente = {cat_triaje}")

        paciente.loc[indice, 'Triaje'] = cat_triaje
        print(paciente)

        filas_modificadas.append(paciente)
        lista_categorias.append(cat_triaje)
        
        print("---------------------------------------------------------------")

    if filas_modificadas:
        df_triaje = pd.concat(filas_modificadas)
        print(df_triaje)
        return df_triaje,lista_categorias
    else:
        return df, None
    
# Ejemplo de uso en otro programa:
if __name__ == "__main__":

    pacientes = pd.DataFrame([{
        'Sexo': 'Mujer',
        'Temperatura': 37.5,
        'Pulso': 120,
        'PAS': 115,
        'PAD': 80,
        'SatO2': 68,
        'Edad': 2
    }])
    """
    pacientes = pd.concat(
        [pacientes, pd.DataFrame([{

            'Sexo': 'Hombre',
            'Temperatura': 36,
            'Pulso': 110,
            'PAS': 116,
            'PAD': 90,
            'SatO2': 98,
            'Edad': 1
        }])
        ],
        ignore_index=True
    )

    pacientes = pd.concat(
        [pacientes, pd.DataFrame([{

            'Sexo': 'Hombre',
            'Temperatura': 36.4,
            'Pulso': 78,
            'PAS': 139,
            'PAD': 89,
            'SatO2': 97,
            'Edad': 31
        }])
        ],
        ignore_index=True
    )

    pacientes = pd.concat(
        [pacientes, pd.DataFrame([{

            'Sexo': 'Mujer',
            'Temperatura': 36,
            'Pulso': 97,
            'PAS': 127,
            'PAD': 59,
            'SatO2': 98,
            'Edad': 24
        }])
        ],
        ignore_index=True
    )
    
    pacientes = pd.concat(
        [pacientes, pd.DataFrame([{

            'Sexo': 'Hombre',
            'Temperatura': 36.2,
            'Pulso': 98,
            'PAS': 138,
            'PAD': 98,
            'SatO2': 97,
            'Edad': 47
        }])
        ],
        ignore_index=True
    )

    pacientes = pd.concat(
        [pacientes, pd.DataFrame([{

            'Sexo': 'Hombre',
            'Temperatura': 36.29,
            'Pulso': 68,
            'PAS': 187,
            'PAD': 77,
            'SatO2': 100,
            'Edad': 63
        }])
        ],
        ignore_index=True
    )
    """
    print(pacientes)

    pacientes,categorias = agregar_triaje(pacientes, 0)
    print(categorias)
     
