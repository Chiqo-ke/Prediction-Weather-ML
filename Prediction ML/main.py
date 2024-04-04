import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Load historical weather data
historical_data = pd.read_csv('historical_weather_data.csv')

# Dummy data preprocessing and splitting
X = historical_data[['Start_Month_Temp', 'End_Month_Temp', 'Average_Temp', 'Rainfall']].values
y_temp = historical_data['Start_Month_Temp'].values  # Example: Using 'Start_Month_Temp' for temperature prediction
y_cond = (historical_data['Next_Month_Condition'] == 'Sunny').astype(int).values  # 1 for Sunny, 0 for Rainy

# Splitting into training and testing sets manually (80% training, 20% testing)
split_ratio = 0.8
split_index = int(len(X) * split_ratio)

X_train = X[:split_index]
y_temp_train = y_temp[:split_index]
y_cond_train = y_cond[:split_index]
X_test = X[split_index:]
y_temp_test = y_temp[split_index:]
y_cond_test = y_cond[split_index:]

# Standardizing features manually (subtract mean and divide by standard deviation)
X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)

X_train_scaled = (X_train - X_train_mean) / X_train_std
X_test_scaled = (X_test - X_train_mean) / X_train_std

# Reshape the input data for LSTM
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build the multi-output LSTM model
inputs = keras.Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
lstm_out = layers.LSTM(64)(inputs)
temp_output = layers.Dense(1, name='temp_output')(lstm_out)  # Output for temperature prediction
cond_output = layers.Dense(1, activation='sigmoid', name='cond_output')(lstm_out)  # Output for condition prediction

model = keras.Model(inputs=inputs, outputs=[temp_output, cond_output])

# Compile the model with metrics for each output
model.compile(optimizer='adam',
              loss={'temp_output': 'mean_squared_error', 'cond_output': 'binary_crossentropy'},
              metrics={'temp_output': 'mean_absolute_error', 'cond_output': 'accuracy'})

# Train the model
model.fit(X_train_reshaped, {'temp_output': y_temp_train, 'cond_output': y_cond_train},
          epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
evaluation = model.evaluate(X_test_reshaped, {'temp_output': y_temp_test, 'cond_output': y_cond_test})
print(f'Evaluation results: {evaluation}')

# Unpack the evaluation results
loss = evaluation[0]
temp_loss = evaluation[1]
cond_accuracy = evaluation[2]

print(f'Test loss: {loss}')
print(f'Test temp_loss: {temp_loss}')
print(f'Test cond_accuracy: {cond_accuracy}')

# Load new historical weather data for prediction
new_data = pd.read_csv('new_weather_data.csv')  # Replace 'new_weather_data.csv' with your new data file

# Preprocess the new data
X_new = new_data[['Start_Month_Temp', 'End_Month_Temp', 'Average_Temp', 'Rainfall']].values
X_new_scaled = (X_new - X_train_mean) / X_train_std
X_new_reshaped = X_new_scaled.reshape((X_new_scaled.shape[0], 1, X_new_scaled.shape[1]))

# Make predictions for new data
temp_pred, cond_pred = model.predict(X_new_reshaped)

# Print the predicted next month's average temperature and weather condition for new data
print(f'Predicted next month\'s average temperature: {temp_pred[0][0]}')
predicted_condition = 'Sunny' if cond_pred[0][0] > 0.5 else 'Rainy'
print(f'Predicted next month\'s weather condition: {predicted_condition}')
