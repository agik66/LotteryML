import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from art import text2art
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

def print_intro():
    ascii_art = text2art("agik66_LotteryML")
    print("Support my work:")
    print("BTC: 13NUSU6q6mheb54FVyi3oD9CPV5tJWCfPG")
    print("ERGO: 9g77R7maUqbGVMsPAHG71pA7TgiKWkYFo71BSSk4fJrn8D4RKxr")
    print("============================================================")
    print(ascii_art)
    print("============================================================")
def load_data():
    data = np.genfromtxt('data.txt', delimiter=',', dtype=int)
    data[data == -1] = 0
    train_data = data[:int(0.8*len(data))]
    val_data = data[int(0.8*len(data)):]
    max_value = np.max(data)
    return train_data, val_data, max_value
def create_model(num_features, max_value):
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=max_value+1, output_dim=64))
    model.add(layers.LSTM(256, return_sequences=True)) 
    model.add(layers.Dropout(0.3))  
    model.add(layers.LSTM(128)) 
    model.add(layers.Dropout(0.3))  
    model.add(layers.Dense(num_features, activation='relu')) 
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mse', 'accuracy'])
    return model
def train_model(model, train_data, val_data, epochs=100):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    history = model.fit(train_data, train_data, validation_data=(val_data, val_data),
                        epochs=epochs, callbacks=[early_stopping, reduce_lr])
def predict_numbers(model, val_data, num_features):
    predictions = model.predict(val_data)
    indices = np.argsort(predictions, axis=1)[:, -num_features:]
    predicted_numbers = np.take_along_axis(val_data, indices, axis=1)
    return predicted_numbers
def print_predicted_numbers(predicted_numbers):
   print("============================================================")
   print("Predicted Numbers:")
   print(', '.join(map(str, predicted_numbers[0])))
   print("============================================================")
def main():
   print_intro()
   train_data, val_data, max_value = load_data()
   num_features = train_data.shape[1]
   model = create_model(num_features, max_value)
   train_model(model, train_data, val_data)
   predicted_numbers = predict_numbers(model, val_data, num_features)
   print_predicted_numbers(predicted_numbers)
if __name__ == "__main__":
   main()