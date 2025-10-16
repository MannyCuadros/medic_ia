import json

def query_fhir_data():
    # Cargar los datos de pacientes y observaciones
    with open("fhir_data/Patient.json") as f:
        patients = json.load(f)
    with open("fhir_data/Observation.json") as f:
        observations = json.load(f)
    
    # Mapeo de códigos LOINC a campos de salida
    loinc_to_field = {
        "29463-7": "__peso",         # Peso
        "8302-2": "__talla",         # Talla
        "2339-0": "__hgt",           # Glucosa (se mapea a __hgt en este ejemplo)
        "8310-5": "__temperatura",   # Temperatura corporal
        "8480-6": "__pas",           # Presión arterial sistólica
        "8462-4": "__pad",           # Presión arterial diastólica
        "8867-4": "__pulso",         # Pulso cardíaco
        "2708-6": "__sat02"          # Saturación de oxígeno
    }
    
    # Inicializar diccionario de resultados por paciente usando el ID del recurso
    patient_results = {}
    for patient in patients:
        pid = patient["id"]
        # Se inicia con el ID; los demás campos se completarán si se encuentran observaciones
        patient_results[pid] = {"ID": pid}
    
    # Recorrer las observaciones y asignar valores a cada paciente
    for obs in observations:
        coding = obs.get("code", {}).get("coding", [])
        if not coding:
            continue
        loinc = coding[0].get("code")
        if loinc in loinc_to_field:
            patient_ref = obs.get("subject", {}).get("reference", "")
            # Se espera que el formato sea "Patient/{patient_id}"
            pid = patient_ref.split("/")[-1]
            field = loinc_to_field[loinc]
            value = obs.get("valueQuantity", {}).get("value")
            patient_results[pid][field] = value
    
    # Convertir los resultados a la estructura solicitada
    output = {"pacientes": list(patient_results.values())}
    
    # Guardar el resultado en un archivo JSON
    with open("fhir_data/aggregated_output.json", "w") as f:
        json.dump(output, f, indent=4)
    
    print("Archivo 'aggregated_output.json' generado con la siguiente información:")
    print(json.dumps(output, indent=4))

if __name__ == "__main__":
    query_fhir_data()
