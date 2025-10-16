import pandas as pd
import json
import requests
import uuid

# Configurar la URL del servidor FHIR
FHIR_URL = "http://localhost:8080/fhir"

rutaBD = "databases/Datos HACQ (Ampliada) - AG.csv"
separador = ";"

# Leer el archivo CSV
def leer_csv(ruta_archivo, separator):
    print(f"Leyendo archivo {ruta_archivo}")
    return pd.read_csv(ruta_archivo, sep=separator, dtype=str).fillna("")

# Función para generar un ID único basado en el RUN del paciente
def generar_id_paciente(row):
    if row["RunPaciente"]:
        return f"pat-{row['RunPaciente']}"
    return f"pat-{uuid.uuid4().hex[:8]}"  # Si no tiene RUN, genera un ID aleatorio

# Función para convertir un paciente a JSON FHIR
def crear_json_paciente(row, patient_id):
    return {
        "resourceType": "Patient",
        "id": patient_id,
        "name": [{
            "use": "official",
            "family": f"{row['ApellidoPaterno']} {row['ApellidoMaterno']}".strip(),
            "given": [row["NombrePaciente"]]
        }],
        "gender": row["Sexo"].lower(),
        "extension": [{
            "url": "http://example.org/fhir/StructureDefinition/edadPaciente",
            "valueInteger": int(row["EdadPaciente"]) if row["EdadPaciente"].isdigit() else None
        }]
    }

# Función para crear observaciones FHIR
def crear_json_observacion(row, patient_id):
    observaciones = []
    valores = {
        "Categoria de urgencia (TRIAGE FINAL)": ("categoria-urgencia", "valueString"),
        "TEMPERATURA AXIAL": ("8310-5", "valueQuantity", "°C", "Cel"),
        "FRECUENCIA CARDIACA": ("8867-4", "valueQuantity", "beats/min", "/min"),
        "PRESIÓN ARTERIAL SISTÓLICA (PAS)": ("8480-6", "valueQuantity", "mmHg", "mm[Hg]"),
        "PRESIÓN ARTERIAL DIASTÓLICA (PAD)": ("8462-4", "valueQuantity", "mmHg", "mm[Hg]"),
        "Saturación O2": ("2710-2", "valueQuantity", "%", "%"),
        "Destino": ("destino", "valueString")
    }
    
    for campo, (codigo, tipo, *unidades) in valores.items():
        if row[campo]:  # Solo agregar si tiene valor
            obs = {
                "resourceType": "Observation",
                "status": "final",
                "subject": {"reference": f"Patient/{patient_id}"},
                "code": {"text": campo}
            }
            if tipo == "valueString":
                obs["valueString"] = row[campo]
            elif tipo == "valueQuantity":
                obs["valueQuantity"] = {
                    "value": float(row[campo]),
                    "unit": unidades[0],
                    "system": "http://unitsofmeasure.org",
                    "code": unidades[1]
                }
            observaciones.append(obs)

    return observaciones

# Función para guardar los JSON localmente
def guardar_json(data, filename):
    print("Guardando archivo... ")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Función para enviar datos al servidor FHIR
def enviar_fhir(data, endpoint, metodo="POST"):
    url = f"{FHIR_URL}/{endpoint}"
    headers = {"Content-Type": "application/fhir+json"}
    response = requests.request(metodo, url, headers=headers, json=data)
    print(f"{metodo} {url} -> {response.status_code}")
    return response

print("Empezando...")
df = leer_csv(rutaBD, separador)

# Procesar cada fila del CSV
for _, row in df.iterrows():
    patient_id = generar_id_paciente(row)
    
    # Crear JSON de paciente
    json_paciente = crear_json_paciente(row, patient_id)
    guardar_json(json_paciente, f"fhir_data/data/{patient_id}.json")
    
    # Subir paciente al servidor FHIR con PUT
    #enviar_fhir(json_paciente, f"Patient/{patient_id}", "PUT")

    # Crear y subir observaciones
    observaciones = crear_json_observacion(row, patient_id)
    for i, obs in enumerate(observaciones):
        obs_id = f"obs-{patient_id}-{i}"
        obs["id"] = obs_id  # Asignar un ID único
        guardar_json(obs, f"fhir_data/data/{obs_id}.json")
        #enviar_fhir(obs, "Observation", "POST")

print("✅ Todos los datos han sido procesados y enviados a FHIR.")
