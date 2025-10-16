import os
from datetime import datetime

def get_estadisticas_files(directory = "estadisticas"):
    """
    Recolecta todos los archivos que comienzan con 'estadística_' y 
    los ordena de forma descendente según la fecha contenida en su nombre.
    """
    files = []
    for filename in os.listdir(directory):
        if filename.startswith("estadística_"):
            try:
                # Se asume que el formato es: estadística_YYYYMMDDHHMM.json
                date_str = filename.split("_")[1].split('.')[0]
                file_date = datetime.strptime(date_str, "%Y%m%d%H%M")
                files.append((filename, file_date))
            except Exception as e:
                print(f"Error al procesar el archivo '{filename}': {e}")
    # Ordena de más reciente a más antiguo
    files.sort(key=lambda x: x[1], reverse=True)
    return [f[0] for f in files]

def get_graficos_files(directory = "estadisticas"):
    """
    Recolecta todos los archivos que comienzan con 'grafico_' y 
    los ordena de forma descendente según la fecha contenida en su nombre.
    """
    files = []
    for filename in os.listdir(directory):
        if filename.startswith("graficos_"):
            try:
                # Se asume que el formato es: estadística_YYYYMMDDHHMM.json
                date_str = filename.split("_")[1].split('.')[0]
                file_date = datetime.strptime(date_str, "%Y%m%d%H%M")
                files.append((filename, file_date))
            except Exception as e:
                print(f"Error al procesar el archivo '{filename}': {e}")
    # Ordena de más reciente a más antiguo
    files.sort(key=lambda x: x[1], reverse=True)
    return [f[0] for f in files]

def get_base_datos_files(directory = "databases"):
    """
    Recolecta todos los archivos que comienzan con 'base_datos_' y 
    los ordena de forma descendente según la fecha contenida en su nombre.
    """
    files = []
    for filename in os.listdir(directory):
        if filename.startswith("basedatos_"):
            try:
                # Se asume que el formato es: estadística_YYYYMMDDHHMM.json
                date_str = filename.split("_")[1].split('.')[0]
                file_date = datetime.strptime(date_str, "%Y%m%d%H%M")
                files.append((filename, file_date))
            except Exception as e:
                print(f"Error al procesar el archivo '{filename}': {e}")
    # Ordena de más reciente a más antiguo
    files.sort(key=lambda x: x[1], reverse=True)
    return [f[0] for f in files]


def get_modelos_files(directory = "modelos"):
    """
    Recolecta todos los archivos que comienzan con 'modelo_' y 
    los ordena de forma descendente según la fecha contenida en su nombre.
    """
    files = []
    for filename in os.listdir(directory):
        if filename.startswith("modelo_"):
            try:
                # Se asume que el formato es: estadística_YYYYMMDDHHMM.json
                date_str = filename.split("_")[1].split('.')[0]
                file_date = datetime.strptime(date_str, "%Y%m%d%H%M")
                files.append((filename, file_date))
            except Exception as e:
                print(f"Error al procesar el archivo '{filename}': {e}")
    # Ordena de más reciente a más antiguo
    files.sort(key=lambda x: x[1], reverse=True)
    return [f[0] for f in files]


def get_encoders_files(directory = "modelos"):
    """
    Recolecta todos los archivos que comienzan con 'encoder_' y 
    los ordena de forma descendente según la fecha contenida en su nombre.
    """
    files = []
    for filename in os.listdir(directory):
        if filename.startswith("encoder_"):
            try:
                # Se asume que el formato es: estadística_YYYYMMDDHHMM.json
                date_str = filename.split("_")[1].split('.')[0]
                file_date = datetime.strptime(date_str, "%Y%m%d%H%M")
                files.append((filename, file_date))
            except Exception as e:
                print(f"Error al procesar el archivo '{filename}': {e}")
    # Ordena de más reciente a más antiguo
    files.sort(key=lambda x: x[1], reverse=True)
    return [f[0] for f in files]


def get_reportes_files(directory = "modelos"):
    """
    Recolecta todos los archivos que comienzan con 'reporte_' y 
    los ordena de forma descendente según la fecha contenida en su nombre.
    """
    files = []
    for filename in os.listdir(directory):
        if filename.startswith("reporte_"):
            try:
                # Se asume que el formato es: estadística_YYYYMMDDHHMM.json
                date_str = filename.split("_")[1].split('.')[0]
                file_date = datetime.strptime(date_str, "%Y%m%d%H%M")
                files.append((filename, file_date))
            except Exception as e:
                print(f"Error al procesar el archivo '{filename}': {e}")
    # Ordena de más reciente a más antiguo
    files.sort(key=lambda x: x[1], reverse=True)
    return [f[0] for f in files]


def get_matrices_files(directory = "modelos"):
    """
    Recolecta todos los archivos que comienzan con 'matriz_' y 
    los ordena de forma descendente según la fecha contenida en su nombre.
    """
    files = []
    for filename in os.listdir(directory):
        if filename.startswith("matriz_"):
            try:
                # Se asume que el formato es: estadística_YYYYMMDDHHMM.json
                date_str = filename.split("_")[1].split('.')[0]
                file_date = datetime.strptime(date_str, "%Y%m%d%H%M")
                files.append((filename, file_date))
            except Exception as e:
                print(f"Error al procesar el archivo '{filename}': {e}")
    # Ordena de más reciente a más antiguo
    files.sort(key=lambda x: x[1], reverse=True)
    return [f[0] for f in files]


def get_basedatos():
    """
    Recolecta el nombre de la base de datos almacenada en el archivo 'basededatos.txt'
    """
    try:
        with open("files/basededatos.txt", "r", encoding="utf-8") as archivo:
            linea = archivo.readline().strip()  # Leer la primera línea y quitar espacios en blanco
            return linea
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return None
    
def save_basedatos(database_name):
    """
    Guarda el nombre de una base de datos en el archivo 'basededatos.txt'
    """
    try:
        with open("files/basededatos.txt", "w", encoding="utf-8") as archivo:
            archivo.write(database_name)
            return True
    except Exception as e:
        print(f"Error al escribir el archivo: {e}")
        return False
    
def get_modelo():
    """
    Recolecta el nombre del modelo almacenado en el archivo 'modelo.txt'
    """
    try:
        with open("files/modelo.txt", "r", encoding="utf-8") as archivo:
            linea = archivo.readline().strip()  # Leer la primera línea y quitar espacios en blanco
            return linea
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return None
    
def save_modelo(model_name):
    """
    Guarda el nombre de un modelo de entrenamiento en el archivo 'modelo.txt'
    """
    try:
        with open("files/modelo.txt", "w", encoding="utf-8") as archivo:
            archivo.write(model_name)
            return True
    except Exception as e:
        print(f"Error al escribir el archivo: {e}")
        return False
    
def get_encoder():
    """
    Recolecta el nombre del encoder almacenado en el archivo 'encoder.txt'
    """
    try:
        with open("files/encoder.txt", "r", encoding="utf-8") as archivo:
            linea = archivo.readline().strip()  # Leer la primera línea y quitar espacios en blanco
            return linea
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return None
    
def save_encoder(encoder_name):
    """
    Guarda el nombre de un encoder en el archivo 'encoder.txt'
    """
    try:
        with open("files/encoder.txt", "w", encoding="utf-8") as archivo:
            archivo.write(encoder_name)
            return True
    except Exception as e:
        print(f"Error al escribir el archivo: {e}")
        return False

def update_estadisticas_list(directory = "estadisticas", output_directory = "files", output_name = "estadisticas_files"):
    
    files = get_estadisticas_files(directory)
    
    try:
        output_file = output_directory + '/' + output_name + '.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            for file in files:
                f.write(file + "\n")
        # Mensaje de éxito
        print(f"\033[92mÉxito:\033[0m Archivo '\033[94m{output_file}\033[0m' actualizado correctamente")
        
    except Exception as e:
        # Mensaje de error detallado
        print(f"\n\033[91mError:\033[0m No se pudo actualizar el archivo '\033[1m{output_name}.txt\033[0m'")
        print(f"\033[94mRuta intentada:\033[0m {output_file}")
        print(f"\033[94mTipo de error:\033[0m {type(e).__name__}")
        print(f"\033[94mDetalles:\033[0m {str(e)}\n")    

def update_graficos_list(directory = "estadisticas", output_directory = "files", output_name = "graficos_files"):

    files = get_graficos_files(directory)

    try:
        output_file = output_directory + '/' + output_name + '.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            for file in files:
                f.write(file + "\n")
        # Mensaje de éxito
        print(f"\033[92mÉxito:\033[0m Archivo '\033[94m{output_file}\033[0m' actualizado correctamente")
        
    except Exception as e:
        # Mensaje de error detallado
        print(f"\n\033[91mError:\033[0m No se pudo actualizar el archivo '\033[1m{output_name}.txt\033[0m'")
        print(f"\033[94mRuta intentada:\033[0m {output_file}")
        print(f"\033[94mTipo de error:\033[0m {type(e).__name__}")
        print(f"\033[94mDetalles:\033[0m {str(e)}\n")

def update_base_datos_list(directory = "databases", output_directory = "files", output_name = "databases_files"):

    files = get_base_datos_files(directory)
    
    try:
        output_file = output_directory + '/' + output_name + '.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            for file in files:
                f.write(file + "\n")
        # Mensaje de éxito
        print(f"\033[92mÉxito:\033[0m Archivo '\033[94m{output_file}\033[0m' actualizado correctamente")
        
    except Exception as e:
        # Mensaje de error detallado
        print(f"\n\033[91mError:\033[0m No se pudo actualizar el archivo '\033[1m{output_name}.txt\033[0m'")
        print(f"\033[94mRuta intentada:\033[0m {output_file}")
        print(f"\033[94mTipo de error:\033[0m {type(e).__name__}")
        print(f"\033[94mDetalles:\033[0m {str(e)}\n")

def update_modelos_list(directory = "modelos", output_directory = "files", output_name = "modelos_files"):

    files = get_modelos_files(directory)
    
    try:
        output_file = output_directory + '/' + output_name + '.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            for file in files:
                f.write(file + "\n")
        # Mensaje de éxito
        print(f"\033[92mÉxito:\033[0m Archivo '\033[94m{output_file}\033[0m' actualizado correctamente")
        
    except Exception as e:
        # Mensaje de error detallado
        print(f"\n\033[91mError:\033[0m No se pudo actualizar el archivo '\033[1m{output_name}.txt\033[0m'")
        print(f"\033[94mRuta intentada:\033[0m {output_file}")
        print(f"\033[94mTipo de error:\033[0m {type(e).__name__}")
        print(f"\033[94mDetalles:\033[0m {str(e)}\n")


def update_encoders_list(directory = "modelos", output_directory = "files", output_name = "encoders_files"):

    files = get_encoders_files(directory)
    
    try:
        output_file = output_directory + '/' + output_name + '.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            for file in files:
                f.write(file + "\n")
        # Mensaje de éxito
        print(f"\033[92mÉxito:\033[0m Archivo '\033[94m{output_file}\033[0m' actualizado correctamente")
        
    except Exception as e:
        # Mensaje de error detallado
        print(f"\n\033[91mError:\033[0m No se pudo actualizar el archivo '\033[1m{output_name}.txt\033[0m'")
        print(f"\033[94mRuta intentada:\033[0m {output_file}")
        print(f"\033[94mTipo de error:\033[0m {type(e).__name__}")
        print(f"\033[94mDetalles:\033[0m {str(e)}\n")


def update_reportes_list(directory = "modelos", output_directory = "files", output_name = "reportes_files"):

    files = get_reportes_files(directory)
    
    try:
        output_file = output_directory + '/' + output_name + '.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            for file in files:
                f.write(file + "\n")
        # Mensaje de éxito
        print(f"\033[92mÉxito:\033[0m Archivo '\033[94m{output_file}\033[0m' actualizado correctamente")
        
    except Exception as e:
        # Mensaje de error detallado
        print(f"\n\033[91mError:\033[0m No se pudo actualizar el archivo '\033[1m{output_name}.txt\033[0m'")
        print(f"\033[94mRuta intentada:\033[0m {output_file}")
        print(f"\033[94mTipo de error:\033[0m {type(e).__name__}")
        print(f"\033[94mDetalles:\033[0m {str(e)}\n")


def update_matrices_list(directory = "modelos", output_directory = "files", output_name = "matrices_files"):

    files = get_matrices_files(directory)
    
    try:
        output_file = output_directory + '/' + output_name + '.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            for file in files:
                f.write(file + "\n")
        # Mensaje de éxito
        print(f"\033[92mÉxito:\033[0m Archivo '\033[94m{output_file}\033[0m' actualizado correctamente")
        
    except Exception as e:
        # Mensaje de error detallado
        print(f"\n\033[91mError:\033[0m No se pudo actualizar el archivo '\033[1m{output_name}.txt\033[0m'")
        print(f"\033[94mRuta intentada:\033[0m {output_file}")
        print(f"\033[94mTipo de error:\033[0m {type(e).__name__}")
        print(f"\033[94mDetalles:\033[0m {str(e)}\n")

def get_first_file(type, directory="files"):
    """
    Devuelve el nombre del archivo más reciente (primer elemento de la lista)
    """
    #print(type)
    if type == "estadistica":
        filename = "estadisticas_files"
    elif type == "grafico":
        filename = "graficos_files"
    elif type == "basedatos":
        filename = "databases_files"
    elif type == "modelo":
        filename = "modelos_files"
    elif type == "encoder":
        filename = "encoders_files"
    elif type == "reporte":
        filename = "reportes_files"
    elif type == "matriz":
        filename = "matrices_files"
    else:
        print(f"No se reconoce el tipo {type}")
        filename = None
    
    file_path = directory + '/' + filename + '.txt'
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            return first_line if first_line else None
    except Exception as e:
        print(f"Error al leer el archivo de lista '{file_path}': {e}")
        return None

def update_all():
    update_estadisticas_list()
    update_graficos_list()
    update_base_datos_list()
    update_modelos_list()
    update_encoders_list()
    update_reportes_list()
    update_matrices_list()

# Ejemplo de uso:
if __name__ == "__main__":

    update_all()
    
    # Obtiene y muestra el primer archivo
    first_file = get_first_file("estadistica")
    print(f"El primer archivo de estadística es: {first_file}")

    first_file = get_first_file("grafico")
    print(f"El primer archivo de gráficos es: {first_file}")

    first_file = get_first_file("basedatos")
    print(f"El primer archivo bases de datos es: {first_file}")

    first_file = get_first_file("modelo")
    print(f"El primer archivo modelos es: {first_file}")

    first_file = get_first_file("encoder")
    print(f"El primer archivo encoder es: {first_file}")

    first_file = get_first_file("reporte")
    print(f"El primer archivo reporte es: {first_file}")

    first_file = get_first_file("matriz")
    print(f"El primer archivo matrices es: {first_file}")