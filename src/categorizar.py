import json

def clasificar_edad(valor):
    if valor == -1:
        return ("no registrado", -1)
    """
    if 0 <= valor < 2:
        return ("Lactante", 6)
    elif 2 <= valor < 19:
        return ("Pediátrico", 4)
    elif 19 <= valor < 35:
        return ("Adulto joven", 0)
    elif 35 <= valor < 65:
        return ("Adulto medio", 4)
    elif valor >= 65:
        return ("Adulto mayor", 6)
    """
    if 0 <= valor < 18:
        return ("Pediátrico", 4)
    elif 18 <= valor < 35:
        return ("Adulto joven", 0)
    elif 36 <= valor < 65:
        return ("Adulto medio", 4)
    elif valor >= 65:
        return ("Adulto mayor", 6)
    else:
        return ("no registrado", -1)

def clasificar_satO2(valor):
    if valor == -1:
        return ("no registrado", -1)
    if valor >= 95:
        return ("Normal", 0)
    elif 90 <= valor < 95:
        return ("Hipoxemia leve", 9)
    elif 85 <= valor < 90:
        return ("Hipoxemia moderada", 12)
    elif valor < 85:
        return ("Hipoxemia severa", 15)
    else:
        return ("no registrado", -1)

def clasificar_pulso(valor):
    if valor == -1:
        return ("no registrado", -1)
    if valor > 100:
        return ("Taquicardia", 9)
    elif 60 <= valor <= 100:
        return ("Normocardia", 0)
    elif valor < 60:
        return ("Bradicardia", 9)
    else:
        return ("no registrado", -1)

def clasificar_pas(valor):
    if valor == -1:
        return ("no registrado", -1)
    if valor >= 140:
        return ("Hipertensión", 5)
    elif 120 <= valor < 140:
        return ("Prehipertensión", 3)
    elif 90 <= valor < 120:
        return ("Normal", 2)
    elif valor < 90:
        return ("Hipotensión", 3)
    else:
        return ("no registrado", -1)

def clasificar_pad(valor):
    if valor == -1:
        return ("no registrado", -1)
    if valor >= 90:
        return ("Hipertensión", 5)
    elif 80 <= valor < 90:
        return ("Prehipertensión", 3)
    elif 60 <= valor < 80:
        return ("Normal", 2)
    elif valor < 60:
        return ("Hipotensión", 3)
    else:
        return ("no registrado", -1)

def clasificar_temperatura(valor):
    if valor == -1:
        return ("no registrado", -1)
    if valor < 36:
        return ("Hipotermia", 4)
    elif 36 <= valor <= 37:
        return ("Normotermina", 0)
    elif 37 < valor < 38:
        return ("Febrícula", 2)
    elif 38 <= valor < 39.4:
        return ("Fiebre", 4)
    elif valor >= 39.4:
        return ("Fiebre alta", 6)
    else:
        return ("no registrado", -1)

def determinar_categoria(total):
    if total >= 0 and total <= 10:
        return "C5"
    elif total >= 11 and total <= 21:
        return "C4"
    elif total >= 22 and total <= 32:
        return "C3"
    elif total >= 33 and total <= 44:
        return "C2"
    elif total > 45:
        return "C1"
    else:
        return "Sin categoría"
    
def categorizar(datos):
    
    print("------------------------------------Scoring Triaje-----------------------------------")

    # Se esperan las siguientes claves para ver si no falta ninguna:
    # Edad, SatO2, Pulso, PAS, PAD, Temperatura
    claves_esperadas = ["Edad", "SatO2", "Pulso", "PAS", "PAD", "Temperatura"]
    for clave in claves_esperadas:
        if clave not in datos:
            print(f"Falta la clave '{clave}' en el archivo JSON.")
            return
    
    # Extraer los valores
    edad_val = float(datos["Edad"].iloc[0])
    satO2_val = float(datos["SatO2"].iloc[0])
    pulso_val = float(datos["Pulso"].iloc[0])
    pas_val = float(datos["PAS"].iloc[0])
    pad_val = float(datos["PAD"].iloc[0])
    temp_val = float(datos["Temperatura"].iloc[0])

    # Clasificar cada variable
    clasif_edad, puntos_edad = clasificar_edad(edad_val)
    clasif_sat, puntos_sat = clasificar_satO2(satO2_val)
    clasif_pulso, puntos_pulso = clasificar_pulso(pulso_val)
    clasif_pas, puntos_pas = clasificar_pas(pas_val)
    clasif_pad, puntos_pad = clasificar_pad(pad_val)
    clasif_temp, puntos_temp = clasificar_temperatura(temp_val)

    print(f"\033[1mVariable\t   Valor  \t   Corte  \t\t   Score\033[0m")
    print(f"Edad\t\t-> {edad_val}  \t-> {clasif_edad}    \t-> {puntos_edad}")
    print(f"SatO2\t\t-> {satO2_val}  \t-> {clasif_sat}    \t-> {puntos_sat}")
    print(f"Pulso\t\t-> {pulso_val}  \t-> {clasif_pulso}  \t-> {puntos_pulso}")
    print(f"PAS\t\t-> {pas_val}  \t-> {clasif_pas}       \t-> {puntos_pas}")
    print(f"PAD\t\t-> {pad_val}  \t-> {clasif_pad}  \t-> {puntos_pad}")
    print(f"Temperatura\t-> {temp_val}  \t-> {clasif_temp}    \t-> {puntos_temp}")
    print("---------------------------------------------------------------")

    # Sumar puntajes solo si el valor fue registrado (no es -1)
    total = 0
    for valor, puntos in [(edad_val, puntos_edad), (satO2_val, puntos_sat), (pulso_val, puntos_pulso),
                           (pas_val, puntos_pas), (pad_val, puntos_pad), (temp_val, puntos_temp)]:
        if puntos != -1:
            total += puntos

    # Determinar la categoría
    categoria = determinar_categoria(total)

    print(f"Score total = {total}")
    print(f"Categoría = {categoria}")
    print("---------------------------------------------------------------")

    return categoria


def categorizar_archivo():
    # Se lee el archivo JSON "valores.json"
    try:
        with open("files/valores.json", "r", encoding="utf-8") as f:
            valores = json.load(f)
    except Exception as e:
        print("Error al leer el archivo 'valores.json':", e)
        return

    # Se esperan 6 valores en el siguiente orden:
    # Edad, SatO2, Pulso, PAS, PAD, Temperatura
    if len(valores) < 6:
        print("El archivo 'valores.json' debe contener 6 valores.")
        return
    
    categoria = categorizar(valores)

    # Escribir el resultado en "categoria.txt"
    try:
        with open("files/categoria.txt", "w", encoding="utf-8") as f_out:
            f_out.write(categoria)
        print("El resultado se ha guardado en 'categoria.txt'")
    except Exception as e:
        print("Error al escribir el archivo 'categoria.txt':", e)

def main():
   
    categorizar_archivo()


if __name__ == "__main__":
    main()
