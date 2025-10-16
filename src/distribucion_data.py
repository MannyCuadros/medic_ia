import pandas as pd
import numpy as np
import funciones_distribucion as fd
import estadisticos as stats
import archivos as files
from datetime import datetime

# Filtrar columnas numéricas a un mínimo y un máximo y seleccionando categóricas necesarias
def filtrar_datos(df):
    df = df.copy()
    # Considerar solo 'Mujer' y 'Hombre' en columna 'Sexo'
    df['Sexo'] = df['Sexo'].replace(
                                    {'Femenino':'Mujer',
                                    'Masculino':'Hombre'}
                                    )
    df = df[df['Sexo'].isin(['Hombre', 'Mujer'])]
    
    # Limitar datos numéricos
    mask = (
        df.get('Edad', pd.Series(True, index=df.index)).between(0, 120) &          # Edad entre 0 y 120
        df.get('Temperatura', pd.Series(True, index=df.index)).between(30, 45) &   # Temp entre 30 y 45    
        df.get('Pulso', pd.Series(True, index=df.index)).between(25, 200) &        # Pulso entre 25 y 200
        df.get('PAS', pd.Series(True, index=df.index)).between(50, 250) &          # PAS entre 50 y 250  
        df.get('PAD', pd.Series(True, index=df.index)).between(30, 150) &          # PAD entre 30 y 150 
        df.get('SatO2', pd.Series(True, index=df.index)).between(50, 100)          # SatO2 entre 50 y 100
        )          

    df = df[mask]

    # Utilizar la primera parte de los datos de la columna Triaje
    df['Triaje'] = df['Triaje'].str.split(' - ').str[0]
    
    # Utilizar solo 3 Categorías: 'Destino', 'Derivación' y 'Hospitalización'
    df['Destino'] = df['Destino'].replace(
    {'Derivación Hospital Servicio de Salud': 'Derivación',
    'Derivación Hospital Red Nacional': 'Derivación',
    'Otro Centro o institución': 'Derivación'}
    )
    df = df[df['Destino'].isin(['Domicilio', 'Hospitalización', 'Derivación'])]
    
    return df


## Funciones para discretizar las variables numéricas
def discretizar_temperatura(temp):
    try:
        temp = float(temp)
    except:
        return None
    if temp < 36:
        return "Hipotermia"
    elif 36 <= temp <= 37:
        return "Normotermia"
    elif 37 < temp < 38:
        return "Febrícula"
    elif temp >= 38:
        return "Fiebre"
    else:
        print(f"temperatura: {temp} -> None")
        return None

def discretizar_pulso(pulso):
    try:
        pulso = float(pulso)
    except:
        return None
    if pulso < 60:
        return "Bradicardia"
    elif 60 <= pulso <= 100:
        return "Normocardia"
    elif pulso > 100:
        return "Taquicardia"
    else:
        return None

def discretizar_pas(pas):
    try:
        pas = float(pas)
    except:
        return None
    if pas < 90:
        return "Hipotensión"
    elif 90 <= pas < 120:
        return "Normal"
    elif 120 <= pas < 149:
        return "Prehipertensión"
    elif pas >= 140:
        return "Hipertensión"
    else:
        return None

def discretizar_pad(pad):
    try:
        pad = float(pad)
    except:
        return None
    if pad < 60:
        return "Hipotensión"
    elif 60 <= pad < 80:
        return "Normal"
    elif 80 <= pad < 90:
        return "Prehipertensión"
    elif pad >= 90:
        return "Hipertensión"
    else:
        return None

def discretizar_sat02(sat02):
    try:
        sat02 = float(sat02)
    except:
        return None
    """
    if sat02 < 0 or sat02 > 100:
        return None
    if 95 <= sat02 <= 100:
        return "Normal"
    elif 90 <= sat02 < 95:
        return "Hipoxemia leve"
    elif 85 <= sat02 < 90:
        return "Hipoxemia moderada"
    elif sat02 < 85:
        return "Hipoxemia severa"
    """
    if sat02 < 0 or sat02 > 100:
        return None
    if 92 <= sat02 <= 100:
        return "Normal"
    elif 88 <= sat02 < 92:
        return "Hipoxemia moderada"
    elif sat02 < 88:
        return "Hipoxemia severa"
    else:
        return None
    
def discretizar_edad(edad):
    try:
        edad = int(edad)
    except:
        return None
    if edad < 0 or edad > 120:
        return None
    """
    if 0 <= edad <= 2:
        return "Lactante"
    elif 2 < edad < 19:
        return "Pediátrico"
    elif 19 <= edad < 40:
        return "Adulto joven"
    elif 40 <= edad < 65:
        return "Adulto medio"
    elif edad >= 65:
        return "Adulto mayor"
    """
    if 0 < edad < 18:
        return "Pediátrico"
    elif 18 <= edad < 35:
        return "Adulto joven"
    elif 35 <= edad < 65:
        return "Adulto medio"
    elif edad >= 65:
        return "Adulto mayor"
    else:
        return None


def prenormalizacion(caperta = "databases" , base_datos1 = "Datos HACQ (Ampliada) - AG.csv", base_datos2 = "Base de datos para desarrollo v2.csv"):

    print("---------------------------------Pre Normalización---------------------------------")

    carpeta_BD = caperta
    nombre_bd1 = base_datos1
    nombre_bd2 = base_datos2

    df1 = fd.leer_csv(nombre_bd1, carpeta_BD, ";")
    df2 = fd.leer_csv(nombre_bd2, carpeta_BD, ",")

    # Manejar duplicados en la primera base de datos
    # Eliminar duplicados manteniendo el primer valor de EdadPaciente
    df1_unico = df1.drop_duplicates(subset=['NroEpisodio'], keep='first')

    # Crear diccionario de mapeo NroEpisodio -> EdadPaciente
    mapeo_edades = df1_unico.set_index('NroEpisodio')['EdadPaciente'].to_dict()

    # Crear la nueva columna en la segunda base de datos
    df2['__edad'] = df2['var002'].map(mapeo_edades)

    # Guardar el resultado (opcional)
    fd.guardar_csv(df2, 'Base de datos para desarrollo v2(con edades).csv', carpeta_BD, ",")
    
    columnas1 = {
        #"TRIAjE INICIAL": "Triaje",
        "Categoria de urgencia (TRIAGE FINAL)": "Triaje",
        "TEMPERATURA AXIAL": "Temperatura",
        "FRECUENCIA CARDIACA": "Pulso",
        "PRESIÓN ARTERIAL SISTÓLICA (PAS)": "PAS",
        "PRESIÓN ARTERIAL DIASTÓLICA (PAD)": "PAD",
        "Saturación O2": "SatO2",
        "EdadPaciente": "Edad",
        "Sexo": "Sexo",
        "Destino": "Destino"
    }

    columnas2 = {
        #"__categ_ini": "Triaje",
        "__categ_fin": "Triaje",
        "__temperatura": "Temperatura",
        "__pulso": "Pulso",
        "__pas": "PAS",
        "__pad": "PAD",
        "__sat02": "SatO2",
        "__edad": "Edad",
        "__sexo": "Sexo",
        "__destino": "Destino"
    }

    df_principal1 = fd.filtrar_por_columnas(df1, columnas1)
    df_principal2 = fd.filtrar_por_columnas(df2, columnas2)

    columns_to_convert = ['Temperatura', 'Pulso', 'PAS', 'PAD', 'SatO2']

    # Convertir las columnas a float64
    for column in columns_to_convert:
        df_principal1[column] = pd.to_numeric(df_principal1[column], errors='coerce').astype('float64')

    for column in columns_to_convert:
        df_principal2[column] = pd.to_numeric(df_principal2[column], errors='coerce').astype('float64')

    # Elimina filas que tienen valores vacíos o nulos
    df_filtrado1 = df_principal1.dropna()
    df_filtrado2 = df_principal2.dropna()

    # Limita los rangos de los datos numéricos a valores posibles
    df_filtrado1 = filtrar_datos(df_filtrado1)
    df_filtrado2 = filtrar_datos(df_filtrado2)


    # Filtrar df según los valores en la columna "Destino" (oversampling)
    df_selector = df_filtrado1[df_filtrado1['Destino'].isin(['Derivación', 'Hospitalización'])]

    # Concatenar df_fusion con el df selector
    df_fusion = pd.concat([df_filtrado2, df_selector], ignore_index=True)

    # Reordenar aleatoriamente las filas
    df_fusion = df_fusion.sample(frac=1, random_state=42).reset_index(drop=True)

    fd.guardar_csv(df_fusion, "Base de datos Smart Triage.csv")
    return df_fusion

def normalizacion(nombre_basedatos = "Base de datos Smart Triage.csv", df_entrada = None):
    
    print("-----------------------------------Normalización-----------------------------------")

    if df_entrada is None:
        df = fd.leer_csv(nombre_basedatos)
    else:
        df = df_entrada
    
    columnas = {
        "Triaje": "Triaje",
        "Temperatura": "Temperatura",
        "Pulso": "Pulso",
        "PAS": "PAS",
        "PAD": "PAD",
        "SatO2": "SatO2",
        "Edad": "Edad",
        "Sexo": "Sexo",
        "Destino": "Destino"
    }

    df_principal = fd.filtrar_por_columnas(df, columnas)

    columnas_a_convertir = ['Temperatura', 'Pulso', 'PAS', 'PAD', 'SatO2', 'Edad']

    # Convertir las columnas a float64
    for columna in columnas_a_convertir:
        df_principal[columna] = pd.to_numeric(df_principal[columna], errors='coerce').astype('float64')

    # Elimina filas que tienen valores vacíos o nulos
    df_filtrado = df_principal.dropna()

    # Limita los rangos de los datos numéricos a valores posibles
    df_filtrado = filtrar_datos(df_filtrado)

    # Crear nueva tabla con columnas categóricas para cada variable
    df_cat = df_filtrado.copy()
    df_cat['Edad_cat']              = df_filtrado['Edad'].apply(discretizar_edad)
    df_cat['Temperatura_cat']       = df_filtrado['Temperatura'].apply(discretizar_temperatura)
    df_cat['Pulso_cat']             = df_filtrado['Pulso'].apply(discretizar_pulso)
    df_cat['PAS_cat']               = df_filtrado['PAS'].apply(discretizar_pas)
    df_cat['PAD_cat']               = df_filtrado['PAD'].apply(discretizar_pad)
    df_cat['SatO2_cat']             = df_filtrado['SatO2'].apply(discretizar_sat02)

    #resetear índices
    df_cat.reset_index(drop=True, inplace=True)

    nuevo_orden = ['Sexo', 'Temperatura', 'Temperatura_cat', 'Pulso', 'Pulso_cat', 'PAS', 'PAS_cat', 'PAD', 'PAD_cat', 'SatO2', 'SatO2_cat', 'Edad', 'Edad_cat', 'Triaje', 'Destino']
    df_cat = df_cat[nuevo_orden]
    
    #print("----------------------------------------------------------------------------------")
    # Acceder a la información
    #fd.visualizar_data(df_cat)
    
    return df_cat

def generar_basedatos():

    df_p = prenormalizacion()
    df_n = normalizacion(df_entrada = df_p)
    
    nueva_version = datetime.now().strftime("%Y%m%d%H%M")

    #Guardar base de datos y estadísticos
    fd.guardar_basedatos(df_n,"basedatos",version = nueva_version)
    stats.generar_estadisticas(df_n, version = nueva_version)

if __name__ == "__main__":
    
    generar_basedatos()