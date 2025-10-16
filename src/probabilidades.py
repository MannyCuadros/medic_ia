import pandas as pd
import numpy as np
import funciones_distribucion as fd

# Cargar los datos (ajusta el nombre del archivo)
df = fd.leer_csv("basedatos_202503271745.csv")

# 1. Probabilidades marginales por variable categórica (tablas separadas)
cat_columns = [col for col in df.columns if '_cat' in col]
marginal_tables = {}

for col in cat_columns:
    prob = df[col].value_counts(normalize=True).reset_index()
    prob.columns = ['Valor', 'Probabilidad']
    prob['Probabilidad'] = prob['Probabilidad'].round(3)
    marginal_tables[col] = prob

# 2. Probabilidades condicionales respecto a Triaje (una tabla por variable)
triaje_tables = {}

for col in cat_columns:
    if col not in ['Triage', 'Destino']:
        cross_tab = pd.crosstab(
            index=df['Triage'], 
            columns=df[col], 
            normalize='index'
        ).round(3)
        triaje_tables[f"{col}_Triage"] = cross_tab

# 3. Probabilidades condicionales respecto a Destino (una tabla por variable)
destino_tables = {}

for col in cat_columns:
    if col not in ['Triage', 'Destino']:
        cross_tab = pd.crosstab(
            index=df['Destino'], 
            columns=df[col], 
            normalize='index'
        ).round(3)
        destino_tables[f"{col}_Destino"] = cross_tab

# Mostrar todas las tablas
print("\n" + "="*50 + "\nTablas de Probabilidades Marginales:\n" + "="*50)
for var, table in marginal_tables.items():
    print(f"\nVariable: {var}")
    print(table.to_string(index=False))

print("\n" + "="*50 + "\nTablas de Probabilidades vs Triaje:\n" + "="*50)
for var, table in triaje_tables.items():
    print(f"\nVariable: {var.split('_')[0]}")
    print(table.to_string())

print("\n" + "="*50 + "\nTablas de Probabilidades vs Destino:\n" + "="*50)
for var, table in destino_tables.items():
    print(f"\nVariable: {var.split('_')[0]}")
    print(table.to_string())