import os
import pandas as pd
import numpy as np
import collections as cols
from imblearn.combine import SMOTEENN
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Cargar el archivo CSV
file_path = os.path.join(os.path.dirname(__file__), 'databases', 'Base de datos para desarrollo v2.csv')
dtype_map = {30: 'str'}
df = pd.read_csv(file_path, dtype=dtype_map)

# Paso 1: Eliminar las columnas no útiles y vacíos

df_cleaned = df[['__temperatura','__pulso','__pas','__pad','__sat02','__destino']].copy()

# Eliminar filas con valores NaN en cualquier columna

# Eliminar las filas donde '__destino'
df_cleaned = df_cleaned.dropna(subset=['__destino'])

# Rellenar los datos NaN por 0
df_cleaned = df_cleaned.fillna(0)

# Paso 2: 

# Convertir las columnas que se encuentran como objeto a numéricas
# Columnas a convertir
columns_to_convert = ['__pas', '__pulso', '__sat02']

# Convertir las columnas a float64
for column in columns_to_convert:
    df_cleaned[column] = pd.to_numeric(df_cleaned[column], errors='coerce').astype('float64')

# Crear una copia del DataFrame para almacenar los mapeos
category_mappings = {}

label_encoders = {}
for column in ['__destino']:
    le = LabelEncoder()
    df_cleaned[column] = le.fit_transform(df_cleaned[column].astype(str)) + 1
    label_encoders[column] = le
    
    category_mappings[column] = pd.DataFrame({
        'Categórico': le.classes_,
        'Numérico': range(1, len(le.classes_) + 1)
    })

print("Mapeo para '__destino':")
print(category_mappings['__destino'])

sorted_df_dest = df_cleaned.sort_values('__destino')

# Separamos el conjunto de datos Destino
categoria_counts_dest = sorted_df_dest['__destino'].value_counts().sort_index()

# Crear un DataFrame con el inicio y fin de cada categoría
sorted_df_dest = sorted_df_dest.reset_index(drop=True)

first_group_dest = sorted_df_dest[sorted_df_dest['__destino'].isin([1, 2, 4, 5, 6])]
second_group_dest = sorted_df_dest[sorted_df_dest['__destino'] == 3]

# Exportamos el dataframe con el subgrupo __destino = 1, 2, 4, 5 (first_group_dest)
output_file_path_sub = os.path.join(os.path.dirname(__file__), 'databases', 'Base de datos para desarrollo v2_dest_sub(preprocesada).csv')
first_group_dest.to_csv(output_file_path_sub, index=False)

# Balanceo de datos con SMOTE
#x = nuevo_grupo_dest[['__temperatura', '__pulso', '__pas', '__pad', '__sat02']]
#y = nuevo_grupo_dest['__destino']
'''
# Aplicar SMOTE solo al conjunto de entrenamiento
smote_enn = SMOTEENN(random_state=42)
x_balanced, y_balanced = smote_enn.fit_resample(x, y)

print(x_balanced)

# Verificar los nuevos tamaños de las clases
print("Distribución después de SMOTE:")
print(y_balanced.value_counts())

nuevo_grupo_dest = pd.concat([x_balanced, y_balanced], axis=1)
'''
# Sobremuestrear la clase minoritaria
class_0_upsampled = resample(first_group_dest,
                             replace=True,      # Permitir duplicados
                             n_samples=len(second_group_dest),  # Igualar a la clase mayoritaria
                             random_state=42)

# Categorizamos al primer grupo como 0 y al segundo grupo como 1 para un mejor entrenamiento
first_group_dest_tmp = class_0_upsampled.copy()
first_group_dest_tmp['__destino'] = 0
second_group_dest_tmp = second_group_dest.copy()
second_group_dest_tmp['__destino'] = 1
nuevo_grupo_dest = pd.concat([first_group_dest_tmp, second_group_dest_tmp], ignore_index=True)
nuevo_grupo_dest = nuevo_grupo_dest.sort_values('__destino')

# Exportar el DataFrame resultante a un archivo CSV
output_file_path = os.path.join(os.path.dirname(__file__), 'databases', 'Base de datos para desarrollo v2_dest(preprocesada) 2.csv')
nuevo_grupo_dest.to_csv(output_file_path, index=False)

print(f"El archivo preprocesado se ha guardado en {output_file_path}")