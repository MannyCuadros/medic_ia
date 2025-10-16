import csv
import itertools
from collections import Counter

# Definición de las variables con sus opciones y puntajes
edad = [
    ("Lactante (>=0 y <2 años)", 4),
    ("Pediátrico (>=2 y <19 años)", 4),
    ("Adulto joven (>=19 y <35 años)", 0),
    ("Adulto medio (>=35 y <65 años)", 4),
    ("Adulto mayor (>=65 años)", 6)
]

saturacion = [
    ("Normal (>=95)", 0),
    ("Hipoxemia leve (>=90 y <95)", 9),
    ("Hipoxemia moderada (>=85 y <90)", 12),
    ("Hipoxemia severa (<85)", 15)
]

frecuencia = [
    ("Taquicardia (>100 bpm)", 9),
    ("Normocardia (>=60 y <=100 bpm)", 0),
    ("Bradicardia (<60 bpm)", 9)
]

pa_sistolica = [
    ("Hipertensión (>=140 mmHg)", 5),
    ("Prehipertensión (>=120 y <140 mmHg)", 3),
    ("Normal (>=90 y <120 mmHg)", 2),
    ("Hipotensión (<90 mmHg)", 2)
]

pa_diastolica = [
    ("Hipertensión (>=90 mmHg)", 5),
    ("Prehipertensión (>=80 y <90 mmHg)", 3),
    ("Normal (>=60 y <80 mmHg)", 2),
    ("Hipotensión (<60 mmHg)", 2)
]

temperatura = [
    ("Hipotermia (<36°C)", 4),
    ("Normotermina (>=36 y <=37°C)", 0),
    ("Febrícula (>37 y <38°C)", 2),
    ("Fiebre (>=38°C y <39.4°C)", 4),
    ("Fiebre alta (>=39.4°C)", 6)
]

# Función para determinar la prioridad en base al puntaje total
def prioridad(total):
    if 0 <= total <= 10:
        return "V"
    elif 11 <= total <= 21:
        return "IV"
    elif 22 <= total <= 32:
        return "III"
    elif 33 <= total <= 44:
        return "II"
    elif total > 45:
        return "I"
    else:
        return "Sin clasificar"

# Se generan todas las combinaciones
combinaciones = list(itertools.product(edad, saturacion, frecuencia, pa_sistolica, pa_diastolica, temperatura))

# Escribimos en un archivo CSV (opcional)
with open("combinaciones_triaje.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    # Encabezado
    writer.writerow([
        "Edad", "Puntaje Edad",
        "Saturación", "Puntaje Saturación",
        "Frecuencia Cardiaca", "Puntaje FC",
        "PA Sístole", "Puntaje PA Sístole",
        "PA Diástole", "Puntaje PA Diástole",
        "Temperatura", "Puntaje Temperatura",
        "Puntaje Total", "Prioridad"
    ])
    
    for comb in combinaciones:
        (edad_text, p_edad), (sat_text, p_sat), (freq_text, p_freq), \
        (sistol_text, p_sistol), (diast_text, p_diast), (temp_text, p_temp) = comb
        
        total = p_edad + p_sat + p_freq + p_sistol + p_diast + p_temp
        pri = prioridad(total)
        
        writer.writerow([
            edad_text, p_edad,
            sat_text, p_sat,
            freq_text, p_freq,
            sistol_text, p_sistol,
            diast_text, p_diast,
            temp_text, p_temp,
            total, pri
        ])

# Ejemplo: Mostrar las primeras 5 combinaciones en consola
for comb in combinaciones[:5]:
    (edad_text, p_edad), (sat_text, p_sat), (freq_text, p_freq), \
    (sistol_text, p_sistol), (diast_text, p_diast), (temp_text, p_temp) = comb
    total = p_edad + p_sat + p_freq + p_sistol + p_diast + p_temp
    pri = prioridad(total)
    print(f"Edad: {edad_text} ({p_edad}), Saturación: {sat_text} ({p_sat}), FC: {freq_text} ({p_freq}), "
          f"PA Sistólica: {sistol_text} ({p_sistol}), PA Diastólica: {diast_text} ({p_diast}), "
          f"Temperatura: {temp_text} ({p_temp}) => Total: {total} => Prioridad: {pri}")


# Abre el archivo CSV (asegúrate de que la codificación sea la correcta, p.ej., 'utf-8')
with open("combinaciones_triaje.csv", "r", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    # Se asume que la columna que contiene la prioridad se llama "Prioridad"
    conteo_prioridades = Counter(row["Prioridad"] for row in reader)

# Mostrar el conteo por cada categoría
for categoria in ["I", "II", "III", "IV", "V"]:
    print(f"Categoría {categoria}: {conteo_prioridades.get(categoria, 0)}")