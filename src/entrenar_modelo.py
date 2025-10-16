import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import FunctionTransformer
from scipy.sparse import csr_matrix
from joblib import dump
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import funciones_distribucion as fd
import archivos as files
import os
import json
import time



# Cargar base de datos seleccionada
def load_database(opcion = 1):
    if opcion == 0:
        base_datos = files.get_first_file("basedatos")
    if opcion == 1:
        base_datos = files.get_basedatos()

    return base_datos

def guardar_modelo(encoder, model, carpeta = "modelos", guardar = 1, version = None):
    
    version = version or datetime.now().strftime("%Y%m%d%H%M")
    nombre_modelo = "modelo" + '_' + version + '.joblib'
    nombre_encoder = "encoder" + '_' + version + '.joblib'
    try:
        if guardar:
            # Crear directorio si no existe
            ruta_modelo = os.path.join(carpeta, nombre_modelo)
            ruta_encoder = os.path.join(carpeta, nombre_encoder)
            os.makedirs(carpeta, exist_ok=True)
            
            # Guardar el mejor modelo
            dump(model, ruta_modelo)
            # Guardar el LabelEncoder (para decodificar las predicciones)
            dump(encoder, ruta_encoder)
            
            # Mensaje de éxito
            print(f"\033[92mÉxito:\033[0m Modelo guardado en '\033[94m{ruta_modelo}\033[0m'")
            
            files.update_modelos_list()
            files.update_encoders_list()
            files.save_modelo(nombre_modelo)
            files.save_encoder(nombre_encoder)
            return True
        else:
            return False
    
    except Exception as e:
        # Mensaje de error detallado
        print(f"\n\033[91mError:\033[0m No se pudo guardar el archivo '\033[1m{nombre_modelo}\033[0m'")
        print(f"\033[94mRuta intentada:\033[0m {nombre_modelo}")
        print(f"\033[94mTipo de error:\033[0m {type(e).__name__}")
        print(f"\033[94mDetalles:\033[0m {str(e)}\n")
        return False

def redondear_valores(reporte):
        # Redondea todos los valores numéricos a 2 decimales (excepto "support")
        reporte_redondeado = {}
        for clave, valor in reporte.items():
            if isinstance(valor, dict):
                reporte_redondeado[clave] = {}
                for subclave, subvalor in valor.items():
                    if subclave == "support":
                        # Convierte "support" a entero (ej: 3550.0 → 3550)
                        reporte_redondeado[clave][subclave] = int(subvalor)
                    else:
                        reporte_redondeado[clave][subclave] = round(subvalor, 2)
            else:
                if clave == "accuracy":
                    reporte_redondeado[clave] = round(valor, 2)
                else:
                    reporte_redondeado[clave] = valor
        return reporte_redondeado

def guardar_reporte(y_test, y_pred, label_encoder, carpeta="modelos", guardar = 1, version=None):
    
    reporte = classification_report(
        y_test, 
        y_pred, 
        target_names=label_encoder.classes_, 
        output_dict=True
    )

    version = version or datetime.now().strftime("%Y%m%d%H%M")
    reporte = redondear_valores(reporte)
     
    try:
        if guardar:
            # Crear directorio si no existe
            nombre_reporte = "reporte" + '_' + version + '.json'
            ruta_reporte = os.path.join(carpeta, nombre_reporte)
            os.makedirs(carpeta, exist_ok=True)

            with open(ruta_reporte, 'w', encoding='utf-8') as f:
                json.dump(reporte, f, indent=4, ensure_ascii=False)
            
            # Mensaje de éxito
            print(f"\033[92mÉxito:\033[0m Reporte de entrenamiento guardado en '\033[94m{ruta_reporte}\033[0m'")
            #return {nombre_archivo + version + 'json', resultados}
            
            files.update_reportes_list()
            return True
        else:
            return False
    except Exception as e:    
        # Mensaje de error detallado
        print(f"\n\033[91mError:\033[0m No se pudo guardar el archivo '\033[1m{nombre_reporte}\033[0m'")
        print(f"\033[94mRuta intentada:\033[0m {ruta_reporte}")
        print(f"\033[94mTipo de error:\033[0m {type(e).__name__}")
        print(f"\033[94mDetalles:\033[0m {str(e)}\n")
        return False

def guardar_matriz_confusion(y_test, y_pred, label_encoder, carpeta="modelos", guardar = 1, version=None):

    try:
        version = version or datetime.now().strftime("%Y%m%d%H%M")
        nombre_matriz = "matriz_" + version + '.png'
        ruta_matriz = os.path.join(carpeta, nombre_matriz)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix(y_test, y_pred),
            annot=True,
            fmt='d',  # Formato entero
            cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_
        )   
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicciones')
        plt.ylabel('Valores Reales')
        if guardar:
            plt.savefig(ruta_matriz, bbox_inches='tight')
            plt.close()
            # Mensaje de éxito
            print(f"\033[92mÉxito:\033[0m Matriz de confusión guardada en '\033[94m{ruta_matriz}\033[0m'")
            files.update_matrices_list()
            return True
        else:
            plt.close()
            return False

    except Exception as e:
        # Mensaje de error detallado
        print(f"\n\033[91mError:\033[0m No se pudo guardar el archivo '\033[1m{nombre_matriz}\033[0m'")
        print(f"\033[94mRuta intentada:\033[0m {ruta_matriz}")
        print(f"\033[94mTipo de error:\033[0m {type(e).__name__}")
        print(f"\033[94mDetalles:\033[0m {str(e)}\n")

def print_model_stats(model, params, label_encoder, y_test, y_pred):
     # Reporte de clasificación
    print("------------------------------Resultado Entrenamiento-------------------------------")

    print("Mejores hiperparámetros:")
    for it in params.items():
        print(f"{it[0]} -> {it[1]}")

    print("----------------------------------------------------------")
    print("Características importantes")
    # Después de obtener el modelo (best_model)
    feature_importances = model.named_steps['classifier'].feature_importances_
    # Obtener los nombres de las características procesadas por el ColumnTransformer
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
    # Crear un DataFrame con las importancias
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values(by='importance', ascending=False)
    # Mostrar las 15 características más importantes
    print(importance_df.head(15))

    print("----------------------------------------------------------")
    print("Reporte de clasificación")
    reporte = classification_report(
        y_test, 
        y_pred, 
        target_names=label_encoder.classes_
    )
    print(reporte)

    print("----------------------------------------------------------")
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))

    print("----------------------------------------------------------\n")

def sparse_to_dense(x):
    return x.toarray() if isinstance(x, csr_matrix) else x 


def entrenar():
    
    print("------------------------------------Entrenamiento-----------------------------------")

    base_datos = load_database()
    df = fd.leer_csv(base_datos)

    #x = df[['Sexo', 'Edad', 'Temperatura', 'Pulso', 'PAS', 'PAD', 'SatO2']]
    #x = df[['Sexo', 'Temperatura', 'Pulso', 'PAS', 'PAD', 'SatO2', 'Triaje']]
    x = df[['Sexo', 'Edad', 'Temperatura', 'Pulso', 'PAS', 'PAD', 'SatO2', 'Triaje']]
    #x = df[['Sexo', 'Edad', 'Temperatura_cat', 'Pulso_cat', 'PAS_cat', 'PAD_cat', 'SatO2_cat', 'Edad_cat', 'Triaje']]
    #x = df.drop(columns="Destino")
    y = df['Destino']

    # Codificar las etiquetas 'y' a números
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Dividir datos en entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(
        x, y_encoded, test_size=0.3, random_state=42
    )

    # Definir columnas categóricas para OneHotEncoder
    #categorical_features = ['Sexo', 'Edad', 'Temperatura', 'Pulso', 'PAS', 'PAD', 'SatO2']
    #categorical_features = ['Sexo', 'Temperatura', 'Pulso', 'PAS', 'PAD', 'SatO2','Triaje']
    #categorical_features = ['Sexo', 'Edad', 'Temperatura', 'Pulso', 'PAS', 'PAD', 'SatO2']
    #categorical_features = ['Sexo', 'Temperatura_cat', 'Pulso_cat', 'PAS_cat', 'PAD_cat', 'SatO2_cat', 'Edad_cat', 'Triaje']
    categorical_features = ['Sexo', 'Triaje']
    numerical_features = ['Temperatura', 'Pulso', 'PAS', 'PAD', 'SatO2','Edad']
    #numerical_features = []
    
    # Preprocesamiento: OneHotEncoder para variables categóricas
    #preprocessor = ColumnTransformer(transformers=[('cat', OrdinalEncoder(handle_unknown='error'), categorical_features)],remainder='passthrough')
    preprocessor = ColumnTransformer(
        transformers=[
            #('num', 'passthrough', numerical_features),
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_features)]
    )
    
    # Calcular class_weights basado en los datos de entrenamiento originales
    class_counts = np.bincount(y_train)
    class_weights = {
        0: class_counts[1] / class_counts[0],  # Derivación
        1: 1.0,                                # Domicilio (clase mayoritaria)
        2: class_counts[1] / class_counts[2]   # Hospitalización
    }

    # Pipeline integrado
    sparse_to_dense_transformer = FunctionTransformer(sparse_to_dense)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),  # Codificación One-Hot
        ('sparse_to_dense', sparse_to_dense_transformer),
        ('undersampler', RandomUnderSampler(
            #sampling_strategy={1: class_counts[0] * 2},  # Undersample de "Domicilio"
            sampling_strategy={1: 30000},  # Undersample de "Domicilio"
            random_state=42
        )),
        ('classifier', XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            eval_metric='mlogloss'
        ))
    ])

    # Definir la cuadrícula de hiperparámetros
    param_grid = {
        'classifier__n_estimators': [200, 300],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__gamma': [0, 0.1],  # Regularización
        'classifier__subsample': [0.8, 1.0]  #% muestras por árbol
    }

    # Configurar GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),  # Validación cruzada de 5 folds
        scoring='f1_macro',  # Métrica a optimizar
        n_jobs=-1  # Paralelizar en todos los núcleos
    )

    # Mapear y_train a pesos según class_weights
    #sample_weights = np.array([class_weights[label] for label in y_train])
    # Calcular los pesos para cada muestra en el conjunto de entrenamiento
    #sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    print(f"\033[95m\033[1mIniciando entrenamiento...\033[0m")

    # Entrenar GridSearchCV (¡esto puede tomar tiempo!)
    tiempo_inicio = time.perf_counter()
    grid_search.fit(x_train, y_train)
    tiempo_fin = time.perf_counter()
    print(f"\033[96m\033[1mEntrenamiento concluido con éxito en {tiempo_fin-tiempo_inicio:.6f} segundos:\033[0m\n")

    # Obtener el mejor modelo
    best_model = grid_search.best_estimator_
    mejores_parametros = grid_search.best_params_
    
    print(mejores_parametros)
    # Predecir y evaluar con el mejor modelo
    y_pred = best_model.predict(x_test)

    # Imprimir estadísticas de entrenamiento
    print_model_stats(best_model, mejores_parametros, label_encoder, y_test, y_pred)
    
    # Guardar modelo y encoder con su respectiva versión
    nueva_version = datetime.now().strftime("%Y%m%d%H%M")
    guardar_modelo(label_encoder, best_model, version = nueva_version)
    guardar_reporte(y_test, y_pred, label_encoder, version = nueva_version)
    guardar_matriz_confusion(y_test, y_pred, label_encoder, version = nueva_version)

def entrenar2(bd = 1):
    """
    Función principal que orquesta el preprocesamiento, entrenamiento y evaluación del modelo.
    """
    print("------------------------------------Entrenamiento-----------------------------------")

    # Cargar la base de datos de entrenamiento
    # Si el parámetro es 0, se carga la última BD generada
    # Si el parámetro es 1, se carga la BD seleccionada en "basededatos.txt"
    base_datos = load_database(bd)
    df = fd.leer_csv(base_datos)
    if df is None:
        return

    # --- MEJORA 1: Selección explícita de características ---
    # En lugar de usar todas las columnas o las versiones "_cat", usamos las
    # variables numéricas originales y las categóricas más relevantes.
    # Esto evita redundancia y le da más poder al modelo para encontrar patrones.
    numerical_features = ['Temperatura', 'Pulso', 'PAS', 'PAD', 'SatO2', 'Edad']
    categorical_features = ['Sexo', 'Triaje']
    
    # Definimos X (features) y y (target)
    x = df[numerical_features + categorical_features]
    y = df['Destino']

    # Codificar la variable de salida 'y'
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # --- MEJORA 2: División estratificada ---
    # Usamos 'stratify=y_encoded' para asegurar que la proporción de clases
    # sea la misma en los conjuntos de entrenamiento y prueba. Es crucial
    # para datasets desbalanceados.
    x_train, x_test, y_train, y_test = train_test_split(
        x, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )

    # --- MEJORA 3: Preprocesador robusto ---
    # Creamos un transformador de columnas que aplica diferentes pasos a cada tipo de variable:
    # - StandardScaler: Estandariza las variables numéricas (media 0, desviación 1).
    #   Ayuda a que el modelo converja más rápido y funcione mejor.
    # - OneHotEncoder: Convierte las variables categóricas en un formato numérico
    #   que el modelo puede entender, sin crear una relación ordinal falsa.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Mantiene otras columnas si las hubiera (en este caso, ninguna)
    )

    # --- MEJORA 4: Estrategia de remuestreo combinada (SMOTE + UnderSampling) ---
    # Esta es la mejora más importante para el desbalance.
    # 1. SMOTE (Oversampling): Crea muestras sintéticas de las clases minoritarias
    #    ('Derivación', 'Hospitalización') para que tengan más representación.
    # 2. RandomUnderSampler (Undersampling): Reduce el número de muestras de la clase
    #    mayoritaria ('Domicilio') para que no domine el entrenamiento.
    # La combinación es más efectiva que solo submuestrear.
    cat_idx = list(range(len(numerical_features), len(numerical_features)+len(categorical_features)))
    #over = SMOTE(sampling_strategy={0: 25000, 2: 25000}, random_state=42) # Aumentar minorías
    over = SMOTENC(categorical_features=cat_idx, sampling_strategy={0: 25000, 2: 25000}, random_state=42) # Aumentar minorías
    under = RandomUnderSampler(sampling_strategy={1: 30000}, random_state=42) # Reducir mayoría

    # --- MEJORA 5: Uso de LightGBM y un Pipeline de Imblearn ---
    # Se reemplaza XGBoost por LightGBM (LGBMClassifier), un modelo a menudo más rápido y preciso.
    # Usamos un Pipeline de `imblearn` que asegura que el remuestreo (SMOTE, etc.)
    # se aplique CORRECTAMENTE solo a los datos de entrenamiento dentro de cada
    # pliegue de la validación cruzada, evitando la fuga de datos (data leakage).
    sparse_to_dense_transformer = FunctionTransformer(sparse_to_dense)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        #('sparse_to_dense', sparse_to_dense_transformer),
        ('oversampler', over),
        ('undersampler', under),
        ('classifier', LGBMClassifier(random_state=42, class_weight='balanced', objective='multiclass', verbose=-1, verbosity=-1))
        #('classifier', XGBClassifier(random_state=42, num_class=3, objective='multi:softmax', eval_metric='mlogloss'))
    ])

    # Definir la cuadrícula de hiperparámetros para LGBM
    param_grid = {
        'classifier__n_estimators': [200, 300],
        'classifier__learning_rate': [0.1, 0.2],
        'classifier__num_leaves': [31, 50],
        'classifier__max_depth': [-1, 7], # -1 significa sin límite
    }

    # Configurar GridSearchCV con validación cruzada estratificada
    # Se reducen los n_splits a 3 para acelerar el proceso, ya que SMOTE es computacionalmente intensivo.
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1_macro', # Métrica ideal para desbalance
        n_jobs=-1 # Usar todos los núcleos de CPU
    )

    print("\033[95m\033[1mIniciando entrenamiento con el nuevo pipeline (puede tomar tiempo)...\033[0m")
    tiempo_inicio = time.perf_counter()
    grid_search.fit(x_train, y_train)
    tiempo_fin = time.perf_counter()
    print(f"\033[96m\033[1mEntrenamiento concluido con éxito en {tiempo_fin-tiempo_inicio:.6f} segundos:\033[0m\n")

    # Obtener el mejor modelo y sus parámetros
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Predecir y evaluar con el mejor modelo
    y_pred = best_model.predict(x_test)

    # Imprimir estadísticas y guardar resul tados
    print_model_stats(best_model, best_params, label_encoder, y_test, y_pred)
    
    # Guardar modelo y encoder con su respectiva versión
    nueva_version = datetime.now().strftime("%Y%m%d%H%M")
    guardar_modelo(label_encoder, best_model, version=nueva_version)
    guardar_reporte(y_test, y_pred, label_encoder, version=nueva_version)
    guardar_matriz_confusion(y_test, y_pred, label_encoder, version=nueva_version)

#---------------------------------------Entrenamiento Triaje#---------------------------------------

def create_features(df):
    """
    Crea nuevas características de ingeniería a partir de las existentes.
    """
    # Presión de Pulso
    df['Presion_Pulso'] = df['PAS'] - df['PAD']
    
    # Presión Arterial Media (PAM)
    df['PAM'] = (df['PAD'] * 2 + df['PAS']) / 3
    
    # Índice de Choque (Shock Index) - con protección para división por cero
    df['Indice_Choque'] = df['Pulso'] / df['PAS'].replace(0, np.nan)
    df['Indice_Choque'].fillna(0, inplace=True) # Rellenar NaNs si PAS era 0

    return df

def entrenar_triaje(bd = 1):
    """
    Función principal para entrenar y evaluar un modelo LGBM para la variable 'Triaje'.
    """
    print("------------------------------------Entrenamiento Triaje-----------------------------------")

    # Cargar los datos
    base_datos = load_database(bd)
    df = fd.leer_csv(base_datos)
    if df is None:
        return

    df = create_features(df)

    # Mapeo explícito para asegurar el orden correcto
    class_mapping = {'C1': 0, 'C2': 1, 'C3': 2, 'C4': 3, 'C5': 4}
    df['Triaje_encoded'] = df['Triaje'].map(class_mapping)

    # Definir X (features) y y (target) para la predicción de 'Triaje'
    # 'Destino' ahora es una característica de entrada
    numerical_features = ['Temperatura', 'Pulso', 'PAS', 'PAD', 'SatO2', 'Edad',
                          'Presion_Pulso', 'PAM', 'Indice_Choque']
    # Se añaden las variables categóricas que representan las categorizaciones de las numéricas
    categorical_features = ['Sexo', 'Temperatura_cat', 'Pulso_cat', 'PAS_cat', 
                            'PAD_cat', 'SatO2_cat', 'Edad_cat']
    
    x = df[numerical_features + categorical_features]
    y = df['Triaje_encoded']
    
    # División estratificada para asegurar la proporción de clases en entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y
    )

    models = {}
    # El orden de las clases para entrenar la cascada (de menos a más grave)
    # Entrenaremos 4 modelos: 0 vs resto, 1 vs resto, 2 vs resto, 3 vs 4
    class_order = sorted(y_train.unique())
    
    # Hacemos una copia para ir filtrando los datos en cada paso
    X_train_step = x_train.copy()
    y_train_step = y_train.copy()

    for i in range(len(class_order) - 1):
        current_class = class_order[i]
        print(f"\n\033[95m\033[1mEntrenando modelo para la clase {current_class} vs. el resto...\033[0m")

        # Preparar etiquetas para el clasificador binario actual
        # 1 si es la clase actual, 0 si es una clase superior
        y_binary = (y_train_step == current_class).astype(int)
        
        # Si solo queda una clase, el entrenamiento para este paso no tiene sentido
        if len(np.unique(y_binary)) < 2:
            print(f"Solo queda una clase, deteniendo el entrenamiento en el paso {i}.")
            continue
            
        # Definir el preprocesador
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            remainder='passthrough'
        )

        # Crear un pipeline con preprocesamiento, remuestreo (SMOTE) y clasificador
        # Usamos SMOTE simple porque SMOTENC no es compatible directamente con el pipeline de imblearn de esta forma
        # y OneHotEncoder convierte todo a numérico antes de SMOTE.
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', LGBMClassifier(objective='binary', random_state=42, n_estimators=200, learning_rate=0.1, num_leaves=50))
        ])

        # Entrenar el modelo del paso actual
        model_pipeline.fit(X_train_step, y_binary)
        
        # Guardar el modelo entrenado
        models[current_class] = model_pipeline
        print(f"\033[96mModelo para clase {current_class} entrenado.\033[0m")

        # Filtrar los datos para el siguiente paso del entrenamiento
        # Nos quedamos solo con las instancias que no fueron clasificadas como la clase actual
        mask = y_train_step != current_class
        X_train_step = X_train_step[mask]
        y_train_step = y_train_step[mask]

    # --- Predicción y Evaluación ---
    
    print("\n\033[95m\033[1mRealizando predicciones con el modelo en cascada...\033[0m")
    y_pred = []
    
    # Copia del conjunto de test para ir prediciendo
    X_test_step = x_test.copy()
    # Guardamos los índices originales para ordenar las predicciones al final
    original_indices = x_test.index.tolist()
    predictions = {}

    for i in range(len(class_order) - 1):
        current_class = class_order[i]
        
        if X_test_step.empty:
            break
            
        model = models[current_class]
        # Predecir si las instancias restantes pertenecen a la clase actual
        binary_preds = model.predict(X_test_step)
        
        # Guardar las predicciones para las instancias clasificadas como la clase actual
        is_current_class_mask = binary_preds == 1
        indices_of_current_class = X_test_step[is_current_class_mask].index
        for idx in indices_of_current_class:
            predictions[idx] = current_class
        
        # Filtrar el conjunto de prueba para el siguiente modelo
        # Nos quedamos con los que el modelo predijo como "clase superior" (predicción = 0)
        is_not_current_class_mask = binary_preds == 0
        X_test_step = X_test_step[is_not_current_class_mask]

    # Las instancias que queden al final pertenecen a la última clase
    for idx in X_test_step.index:
        predictions[idx] = class_order[-1]
        
    # Ordenar las predicciones según el orden original del X_test
    y_pred = [predictions[i] for i in original_indices]

    # --- Mostrar Resultados ---
    
    print("\n------------------------------ Resultado del Entrenamiento en Cascada ------------------------------")
    label_encoder = LabelEncoder().fit(df['Triaje']) # Para obtener los nombres de las clases
    class_names = label_encoder.classes_
    
    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("--------------------------------------------------------------------------------------------------")

if __name__ == '__main__':
    #entrenar2()
    entrenar_triaje(0)
