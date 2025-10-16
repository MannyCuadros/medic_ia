import requests

# URL base del servidor FHIR
FHIR_BASE_URL = "https://hapi.fhir.org/baseR4"
OBSERVATION_URL = f"{FHIR_BASE_URL}/Observation"
PATIENT_URL = f"{FHIR_BASE_URL}/Patient"
LOINC_URL = "https://loinc.org/"  # Para buscar códigos desconocidos

# Función para obtener el nombre del paciente por su ID
def get_patient_name(patient_id):
    response = requests.get(f"{PATIENT_URL}/{patient_id}", headers={"Accept": "application/fhir+json"})
    if response.status_code == 200:
        patient_data = response.json()
        name_info = patient_data.get("name", [{}])[0]
        family = name_info.get("family", "Desconocido")
        given = " ".join(name_info.get("given", ["Desconocido"]))
        return f"{given} {family}"
    return "Desconocido"

# Función para obtener todas las observaciones
def get_observations():
    response = requests.get(f"{OBSERVATION_URL}?_sort=-date", headers={"Accept": "application/fhir+json"})
    if response.status_code == 200:
        return response.json().get("entry", [])
    return []

# Función para obtener el nombre de la prueba desde LOINC
def get_loinc_name(loinc_code):
    loinc_lookup_url = f"{LOINC_URL}{loinc_code}"
    return loinc_lookup_url  # Solo retorna el enlace para búsqueda manual

# Procesar observaciones
observations = get_observations()

for entry in observations:
    observation = entry.get("resource", {})
    patient_ref = observation.get("subject", {}).get("reference", "Patient/Desconocido")
    patient_id = patient_ref.split("/")[-1]  # Extraer ID numérico del paciente
    patient_name = get_patient_name(patient_id)  # Obtener nombre del paciente

    coding = observation.get("code", {}).get("coding", [])
    for code_data in coding:
        code = code_data.get("code", "Desconocido")
        test_name = code_data.get("display", "Desconocido")

        # Si el nombre de la prueba es desconocido, buscar en LOINC
        if test_name == "Desconocido":
            loinc_link = get_loinc_name(code)
            test_name = f"Desconocido (Buscar en LOINC: {loinc_link})"

        # Extraer valor si está disponible
        value = observation.get("valueQuantity", {}).get("value")
        unit = observation.get("valueQuantity", {}).get("unit", "")

        # Mostrar información
        print(f"Paciente: {patient_id} - {patient_name}")
        print(f"  Evaluación: {test_name} ({code})")
        if value is not None:
            print(f"  Valor: {value} {unit}")
        else:
            print("  Valor: No disponible")
        print("-" * 50)
