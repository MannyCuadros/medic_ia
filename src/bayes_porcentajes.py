import pandas as pd
import json

# 1. Cargar el archivo CSV y extraer las columnas de interés
ruta = "databases/Base de datos para desarrollo v2.csv"
columnas = ['__sexo', '__temperatura', '__pulso', '__pas', '__pad', '__sat02', '__destino']
df = pd.read_csv(ruta, usecols=columnas)

print(f"Tamaño base datos: {df.shape}")
print(f"Nombre de columnas: {df.columns.values}")


# Guardar el número total de registros originales
total_registros = len(df)
print(f"Numero de filas {total_registros}")

print("---------------------------------------------")

# 2. Eliminar filas que tengan valor 0 o 1 en cualquiera de las columnas
for col in columnas:
    df = df[~df[col].isin([0, 1])]
# Eliminar filas con datos faltantes (vacíos)
df = df.dropna()

# 3. Discretización de las variables

## __sexo: Solo se permiten los valores "masculino" y "femenino"
#df_sexo = df['__sexo'].str.lower()  # Homogeneizar a minúsculas
df['__sexo'] = df['__sexo'].str.lower()  # Homogeneizar a minúsculas
df = df[df['__sexo'].isin(['masculino', 'femenino'])]
#print(df_sexo)

print(f"Nuevo Numero de filas sin vacios, ni 0 ni 1: {len(df)}")

print("---------------------------------------------")

## Funciones para discretizar las variables numéricas
def discretizar_temperatura(temp):
    try:
        temp = float(temp)
    except:
        return None
    if temp < 36:
        return "hipotermia"
    elif 36 <= temp <= 37:
        return "normotermia"
    elif 37 < temp < 38:
        return "febrícula"
    elif temp >= 38:
        return "fiebre"
    else:
        print(f"temperatura: {temp} -> None")
        return None

def discretizar_pulso(pulso):
    try:
        pulso = float(pulso)
    except:
        return None
    if pulso < 60:
        return "bradicardia"
    elif 60 <= pulso <= 100:
        return "normocardia"
    elif pulso > 100:
        return "taquicardia"
    else:
        return None

def discretizar_pas(pas):
    try:
        pas = float(pas)
    except:
        return None
    if pas < 90:
        return "hipotensión"
    elif 90 <= pas <= 120:
        return "normal"
    elif 120 < pas <= 139:
        return "prehipertensión"
    elif pas >= 140:
        return "hipertensión"
    else:
        return None

def discretizar_pad(pad):
    try:
        pad = float(pad)
    except:
        return None
    if pad < 60:
        return "hipotensión"
    elif 60 <= pad <= 80:
        return "normal"
    elif 80 < pad <= 89:
        return "prehipertensión"
    elif pad >= 90:
        return "hipertensión"
    else:
        return None

def discretizar_sat02(sat02):
    try:
        sat02 = float(sat02)
    except:
        return None
    if sat02 < 0 or sat02 > 100:
        return None
    if 95 <= sat02 <= 100:
        return "normal"
    elif 90 <= sat02 <= 94:
        return "hipoxemia leve"
    elif 85 <= sat02 <= 89:
        return "hipoxemia moderada"
    elif sat02 < 85:
        return "hipoxemia severa"
    else:
        return None
    

# Crear nuevas columnas discretizadas para cada variable numérica
df['__temperatura_disc'] = df['__temperatura'].apply(discretizar_temperatura)
df['__pulso_disc']       = df['__pulso'].apply(discretizar_pulso)
df['__pas_disc']         = df['__pas'].apply(discretizar_pas)
df['__pad_disc']         = df['__pad'].apply(discretizar_pad)
df['__sat02_disc']       = df['__sat02'].apply(discretizar_sat02)

print(f"Tamaño nuevo base datos: {df.shape}")
print("-----------------------------------------------------------------------------------")

# Número total de registros después de la limpieza y discretización
registros_limpios = len(df)

# 4. Calcular tablas de porcentaje para cada variable discretizada
def tabla_porcentajes(serie, total_original, total_limpio):
    conteos = serie.value_counts()
    porcentaje_total = (conteos / total_original * 100).round(2)
    porcentaje_limpio = (conteos / total_limpio * 100).round(2)
    return pd.DataFrame({
        "Cuenta": conteos,
        "Porcentaje (Total)": porcentaje_total,
        "Porcentaje (Limpios)": porcentaje_limpio
    })

tabla_sexo         = tabla_porcentajes(df['__sexo'], total_registros, registros_limpios)
tabla_temperatura  = tabla_porcentajes(df['__temperatura_disc'], total_registros, registros_limpios)
tabla_pulso        = tabla_porcentajes(df['__pulso_disc'], total_registros, registros_limpios)
tabla_pas          = tabla_porcentajes(df['__pas_disc'], total_registros, registros_limpios)
tabla_pad          = tabla_porcentajes(df['__pad_disc'], total_registros, registros_limpios)
tabla_sat02        = tabla_porcentajes(df['__sat02_disc'], total_registros, registros_limpios)
tabla_destino      = tabla_porcentajes(df['__destino'], total_registros, registros_limpios)

# 5. Combinar todas las estadísticas en una sola tabla general
def agregar_nombre_variable(tabla, nombre_variable):
    tabla = tabla.copy()
    tabla['Variable'] = nombre_variable
    # Asignar nombre al índice antes de hacer reset_index
    tabla.index.name = 'Categoría'
    tabla = tabla.reset_index()
    return tabla[['Variable', 'Categoría', 'Cuenta', 'Porcentaje (Total)', 'Porcentaje (Limpios)']]

tabla_sexo_df        = agregar_nombre_variable(tabla_sexo, '__sexo')
tabla_temperatura_df = agregar_nombre_variable(tabla_temperatura, '__temperatura')
tabla_pulso_df       = agregar_nombre_variable(tabla_pulso, '__pulso')
tabla_pas_df         = agregar_nombre_variable(tabla_pas, '__pas')
tabla_pad_df         = agregar_nombre_variable(tabla_pad, '__pad')
tabla_sat02_df       = agregar_nombre_variable(tabla_sat02, '__sat02')
tabla_destino_df     = agregar_nombre_variable(tabla_destino, '__destino')

tabla_general = pd.concat([tabla_sexo_df, tabla_temperatura_df, tabla_pulso_df, tabla_pas_df, tabla_pad_df, tabla_sat02_df, tabla_destino_df], ignore_index=True)

# Mostrar la tabla general
print("Tabla General de Estadísticas:")
print(tabla_general.to_string(index=False))

# Información adicional sobre cantidad de registros
print("\nNúmero total de registros originales:", total_registros)
print("Número de registros después de la limpieza:", registros_limpios)

print("---------------------------------------------------------------------------------------------------------------------------------------------------------------")

def formatear_tabla(tabla):
    """
    Formatea cada fila de la tabla en un string con el formato:
    "Categoría: Cuenta (Porcentaje%)"
    """
    lineas = []
    # Usamos el índice (Categoría) ya que la tabla la generamos con value_counts()
    for categoria, row in tabla.iterrows():
        # Por ejemplo: "femenino: 47418 (56.4%)"
        linea = f"{categoria}: {int(row['Cuenta'])} ({row['Porcentaje (Limpios)']}%)"
        lineas.append(linea)
    return lineas

# Formateamos cada tabla
stats = {}
stats['__sexo']         = formatear_tabla(tabla_sexo)
stats['__temperatura']  = formatear_tabla(tabla_temperatura)
stats['__pas']          = formatear_tabla(tabla_pas)
stats['__pad']          = formatear_tabla(tabla_pad)
stats['__pulso']        = formatear_tabla(tabla_pulso)
stats['__sat02']        = formatear_tabla(tabla_sat02)
stats['__destino']      = formatear_tabla(tabla_destino)

# Determinar el máximo número de filas entre todas las variables
max_filas = max(len(lst) for lst in stats.values())

# Rellenar cada lista con cadenas vacías si tiene menos elementos
for key in stats:
    while len(stats[key]) < max_filas:
        stats[key].append("")

# Crear el DataFrame final
tabla_general = pd.DataFrame(stats)

# Mostrar la tabla resultante
print("Tabla General de Estadísticas:")
print(tabla_general.to_string(index=False))

# Diccionario que relaciona cada variable con su tabla de estadísticas
variables = {
    '__sexo': tabla_sexo,
    '__temperatura': tabla_temperatura,
    '__pulso': tabla_pulso,
    '__pas': tabla_pas,
    '__pad': tabla_pad,
    '__sat02': tabla_sat02,
    '__destino': tabla_destino
}

# Preparar el diccionario para exportar
data_export = {}
for variable, tabla in variables.items():
    data_export[variable] = []
    for categoria, row in tabla.iterrows():
        data_export[variable].append({
            "categoria": categoria,
            "cuenta": int(row["Cuenta"]),
            "porcentaje_limpios": float(row["Porcentaje (Limpios)"])
        })

# Exportar a un archivo JSON
with open("estadisticas.json", "w", encoding="utf-8") as f:
    json.dump(data_export, f, indent=4, ensure_ascii=False)

print("Archivo 'estadisticas.json' exportado correctamente.")