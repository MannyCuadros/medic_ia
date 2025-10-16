import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import classification_report

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

print(f"Nuevo Numero de filas sin vacios, ni 0 ni 1: {len(df)}")
print("---------------------------------------------")

# 3. Discretización de las variables

## __sexo: Solo se permiten los valores "masculino" y "femenino"
#df_sexo = df['__sexo'].str.lower()  # Homogeneizar a minúsculas
df['__sexo'] = df['__sexo'].str.lower()  # Homogeneizar a minúsculas
df = df[df['__sexo'].isin(['masculino', 'femenino'])]

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
    

# Crear nueva tabla con columnas categóricas para cada variable
df_cat = pd.DataFrame()
df_cat['__sexo']        = df['__sexo']
df_cat['__temperatura'] = df['__temperatura'].apply(discretizar_temperatura)
df_cat['__pulso']       = df['__pulso'].apply(discretizar_pulso)
df_cat['__pas']         = df['__pas'].apply(discretizar_pas)
df_cat['__pad']         = df['__pad'].apply(discretizar_pad)
df_cat['__sat02']       = df['__sat02'].apply(discretizar_sat02)
df_cat['__destino']     = df['__destino']

#resetear índices
df_cat.reset_index(drop=True, inplace=True)

# Número total de registros después de la limpieza y discretización
registros_limpios = len(df_cat)

print(df_cat)
print(f"Tamaño nuevo base datos: {df_cat.shape}")
print("-----------------------------------------------------------------------------------")


# Definir las variables predictoras (X) y la variable objetivo (y)
x = df_cat[['__sexo', '__temperatura', '__pulso', '__pas', '__pad', '__sat02']]
y = df_cat['__destino']

x = x.copy()
y = y.copy()

# Convertir cada columna a datos categóricos y luego a códigos numéricos
for col in x.columns:
    x[col] = x[col].astype('category').cat.codes

y = y.astype('category').cat.codes

# Dividir los datos en conjunto de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo Naive Bayes para variables categóricas
model = CategoricalNB()
model.fit(x_train, y_train)

# Predecir para el conjunto de prueba
y_pred = model.predict(x_test)

print("-----------------------------------------------------------------------------------")

# Mostrar el reporte de clasificación
print(classification_report(y_test, y_pred, zero_division=0))

print("-----------------------------------------------------------------------------------")

# Probabilidad previa de cada destino
p_destino = df['__destino'].value_counts(normalize=True).to_dict()
print(p_destino)

print("-----------------------------------------------------------------------------------")

# Crear un diccionario anidado para guardar P(__sexo = x | __destino = d)
p_sexo_cond = {}
for destino in df_cat['__destino'].unique():
    sub_df = df_cat[df_cat['__destino'] == destino]
    total = len(sub_df)
    frec = sub_df['__sexo'].value_counts().to_dict()
    p_sexo_cond[destino] = {sexo: frec[sexo] / total for sexo in frec}

print("P(__sexo = x | __destino = d)")
print(p_sexo_cond)

print("-----------------------------------------------------------------------------------")

# Crear un diccionario anidado para guardar P(__temperatura = x | __destino = d)
p_temperatura_cond = {}
for destino in df_cat['__destino'].unique():
    sub_df = df_cat[df_cat['__destino'] == destino]
    total = len(sub_df)
    frec = sub_df['__temperatura'].value_counts().to_dict()
    p_temperatura_cond[destino] = {temperatura: frec[temperatura] / total for temperatura in frec}

print("P(__temperatura = x | __destino = d)")
print(p_temperatura_cond)

print("-----------------------------------------------------------------------------------")

# Crear un diccionario anidado para guardar P(__pulso = x | __destino = d)
p_pulso_cond = {}
for destino in df_cat['__destino'].unique():
    sub_df = df_cat[df_cat['__destino'] == destino]
    total = len(sub_df)
    frec = sub_df['__pulso'].value_counts().to_dict()
    p_pulso_cond[destino] = {pulso: frec[pulso] / total for pulso in frec}

print("P(__pulso = x | __destino = d)")
print(p_pulso_cond)

print("-----------------------------------------------------------------------------------")

# Crear un diccionario anidado para guardar P(__pas = x | __destino = d)
p_pas_cond = {}
for destino in df_cat['__destino'].unique():
    sub_df = df_cat[df_cat['__destino'] == destino]
    total = len(sub_df)
    frec = sub_df['__pas'].value_counts().to_dict()
    p_pas_cond[destino] = {pas: frec[pas] / total for pas in frec}

print("P(__pas = x | __destino = d)")
print(p_pas_cond)

print("-----------------------------------------------------------------------------------")

# Crear un diccionario anidado para guardar P(__pad = x | __destino = d)
p_pad_cond = {}
for destino in df_cat['__destino'].unique():
    sub_df = df_cat[df_cat['__destino'] == destino]
    total = len(sub_df)
    frec = sub_df['__pad'].value_counts().to_dict()
    p_pad_cond[destino] = {pad: frec[pad] / total for pad in frec}

print("P(__pad = x | __destino = d)")
print(p_pad_cond)

print("-----------------------------------------------------------------------------------")

# Crear un diccionario anidado para guardar P(__sat02 = x | __destino = d)
p_sat02_cond = {}
for destino in df_cat['__destino'].unique():
    sub_df = df_cat[df_cat['__destino'] == destino]
    total = len(sub_df)
    frec = sub_df['__sat02'].value_counts().to_dict()
    p_sat02_cond[destino] = {sat02: frec[sat02] / total for sat02 in frec}

print("P(__sat02 = x | __destino = d)")
print(p_sat02_cond)

print("-----------------------------------------------------------------------------------")

# Supongamos que la evidencia del nuevo paciente es:
nuevo = {
    '__sexo': 'femenino',
    '__temperatura': 'fiebre',
    '__pulso': 'taquicardia',
    '__pas': 'hipertensión',
    '__pad': 'hipertensión',
    '__sat02': 'hipoxemia severa'
}

# Calcula la probabilidad proporcional para cada destino
prob_destinos = {}
for d in p_destino:
    prob = p_destino[d]
    prob *= p_sexo_cond[d].get(nuevo['__sexo'], 1e-6)  # usar un valor pequeño si la categoría no aparece
    # Repite para cada variable predictora; por ejemplo:
    prob *= p_temperatura_cond[d].get(nuevo['__temperatura'], 1e-6)
    prob *= p_pulso_cond[d].get(nuevo['__pulso'], 1e-6)
    prob *= p_pas_cond[d].get(nuevo['__pas'], 1e-6)
    prob *= p_pad_cond[d].get(nuevo['__pad'], 1e-6)
    prob *= p_sat02_cond[d].get(nuevo['__sat02'], 1e-6)
    prob_destinos[d] = prob

# Normaliza para obtener probabilidades que sumen 1
suma = sum(prob_destinos.values())
prob_destinos_normalizadas = {d: prob_destinos[d] / suma for d in prob_destinos}

print("Probabilidades posteriores:")
print(prob_destinos_normalizadas)
