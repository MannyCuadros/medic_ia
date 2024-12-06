import os
import pandas as pd
import numpy as np
from joblib import dump
import collections as cols
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping

# 1 Cargar los datos preprocesados
file_path = os.path.join(os.path.dirname(__file__), 'databases', 'Base de datos para desarrollo v2_dest(preprocesada) 2.csv')
df = pd.read_csv(file_path)

# 2 Separar las características de entrada y de salida (objetivo)
input = df.drop(columns=['__destino'])
output = df['__destino']

# 3: Escalar los datos para que tengan una media de 0 y una desviación estándar de 1
scaler = StandardScaler()
#numeric_columns = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
colums_to_scale = input[['__temperatura','__pulso','__pas','__pad','__sat02']].columns

input[colums_to_scale] = scaler.fit_transform(input[colums_to_scale])

# 4 Dividir los datos en conjuntos de entrenamiento 80% y prueba 20% 
input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.2, random_state=42)

# 5 Construir el modelo dela red neuronal (Perceptron multicapa)
def MLP_NN():
    NumNeurons = 7
    model = Sequential()
    model.add(Dense(64, input_dim=input_train.shape[1]))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.5))
    model.add(Dense(32))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Usar 'softmax' para clasificación multiclase

    #opt =  keras.optimizers.Adam(learning_rate=0.001)

    # Compilar el modelo
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# 6 Entrenar el modelo
n_epochs = 500

network = MLP_NN()
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
train = network.fit(input_train, output_train, epochs=n_epochs, batch_size=32, validation_split=0.2, callbacks=[early_stopping])


output_pred = network.predict(input_test)

# Convertir probabilidades a clases binarias
output_pred_classes = (output_pred > 0.5).astype(int)  # Umbral de 0.5

output_test_classes = output_test  

print(classification_report(output_test_classes, output_pred_classes))

# 7 Guardar el modelo y el escalador
output_file_path_networ = os.path.join(os.path.dirname(__file__), 'modelo', 'modelo_entrenado.h5')
output_file_path_scaler = os.path.join(os.path.dirname(__file__), 'modelo', 'escalador.pkl')

network.save(output_file_path_networ)
# Guardar el escalador
dump(scaler, output_file_path_scaler)