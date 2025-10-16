import json
import os

def generate_patient_resource(id_value, family, given, gender, birthDate):
    """
    Crea un recurso Patient simplificado siguiendo la estructura FHIR.
    Se incluyen algunos campos obligatorios y ejemplos de otros.
    """
    patient = {
        "resourceType": "Patient",
        "id": id_value,
        "identifier": [
            {
                "system": "http://hospital.example.org/patients",
                "value": id_value
            }
        ],
        "active": True,
        "name": [
            {
                "family": family,
                "given": [given]
            }
        ],
        "telecom": [
            {
                "system": "phone",
                "value": "+1-555-0100",
                "use": "home"
            }
        ],
        "gender": gender,
        "birthDate": birthDate,
        # Se pueden agregar más campos según la estructura FHIR
        "address": [
            {
                "line": ["123 Main Street"],
                "city": "Anytown",
                "state": "AN",
                "postalCode": "12345",
                "country": "USA"
            }
        ]
    }
    return patient

def generate_data():
    # Crear directorio para los datos FHIR si no existe
    os.makedirs("fhir_data", exist_ok=True)
    
    # Generar recursos Patient (con estructura FHIR simplificada)
    patients = [
        generate_patient_resource("U0003670130", "Doe", "John", "male", "1980-01-01"),
        generate_patient_resource("U0003678475", "Smith", "Alice", "female", "1990-02-15"),
        generate_patient_resource("U0003679758", "Brown", "Bob", "male", "1975-07-30")
    ]
    with open("fhir_data/Patient.json", "w") as f:
        json.dump(patients, f, indent=4)
    
    # Definición de observaciones con los siguientes códigos LOINC:
    # 29463-7: Peso
    # 8302-2: Talla
    # 2339-0: Glucosa (se usará para este ejemplo el campo "__hgt")
    # 8310-5: Temperatura corporal
    # 8480-6: Presión arterial sistólica
    # 8462-4: Presión arterial diastólica
    # 8867-4: Pulso cardíaco
    # 2708-6: Saturación de oxígeno (SpO2)
    
    # Valores de ejemplo (según el formato de salida solicitado)
    patient_obs = {
        "U0003670130": {
            "29463-7": 75.4,   # Peso (kg)
            "8302-2": 168,     # Talla (cm)
            "2339-0": 74,      # Valor usado para glucosa (en este ejemplo se asigna a __hgt)
            "8310-5": 36.5,    # Temperatura (°C)
            "8480-6": 102,     # Presión sistólica (mmHg)
            "8462-4": 49,      # Presión diastólica (mmHg)
            "8867-4": 88,      # Pulso (lpm)
            "2708-6": 98       # Saturación O2 (%)
        },
        "U0003678475": {
            "29463-7": 63.1,
            "8302-2": 156,
            "2339-0": 72,
            "8310-5": 39.5,
            "8480-6": 102,
            "8462-4": 60,
            "8867-4": 108,
            "2708-6": 30
        },
        "U0003679758": {
            "29463-7": 87.3,
            "8302-2": 178,
            "2339-0": 82,
            "8310-5": 36.59,
            "8480-6": 128,
            "8462-4": 57,
            "8867-4": 80,
            "2708-6": 99
        }
    }
    
    # Generar recursos Observation para cada paciente y cada código LOINC
    observations = []
    obs_counter = 1
    for patient_id, obs_data in patient_obs.items():
        for loinc, value in obs_data.items():
            obs = {
                "resourceType": "Observation",
                "id": f"obs{obs_counter}",
                "status": "final",
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                "code": "vital-signs"
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": loinc
                        }
                    ],
                    "text": "Observation"
                },
                "subject": {
                    "reference": f"Patient/{patient_id}"
                },
                "valueQuantity": {
                    "value": value,
                    "unit": ""
                }
            }
            observations.append(obs)
            obs_counter += 1

    with open("fhir_data/Observation.json", "w") as f:
        json.dump(observations, f, indent=4)

if __name__ == "__main__":
    generate_data()
    print("Datos FHIR generados en la carpeta 'fhir_data'.")
