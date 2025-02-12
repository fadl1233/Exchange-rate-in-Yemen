import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# تحميل البيانات
data = pd.read_excel(r'C:\Users\AL BASHA\PycharmProjects\Exchange rate in Yemen\ex - USD_YER.xlsx', sheet_name='Sheet1')

# عرض البيانات
print(data)
# Inspect the DataFrame
print(data)

# Extract the exchange rate data
usd_yer = data.iloc[0].values[1:]  # Skip the first cell

# Convert to float
usd_yer = usd_yer.astype('float32')

# Normalize the data
max_value = np.max(usd_yer)
usd_yer_normalized = usd_yer / max_value

# Prepare the dataset
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        X.append(a)
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)

look_back = 5
X, Y = create_dataset(usd_yer_normalized, look_back)

# Split the data
train_size = int(len(X) * 0.7)
test_size = len(X) - train_size
X_train, Y_train = X[:train_size], Y[:train_size]
X_test, Y_test = X[train_size:], Y[train_size:]

# Build the model
model = keras.models.Sequential([
    layers.LSTM(50, activation='relu', input_shape=(look_back, 1)),
    layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test), verbose=0)

# Evaluate the model
mse_test = model.evaluate(X_test, Y_test, verbose=0)
print(f'Mean Squared Error on test set: {mse_test}')

# Predict
Y_pred = model.predict(X_test)

# Denormalize the predictions
Y_test_actual = Y_test * max_value
Y_pred_actual = Y_pred * max_value

# Plot the results
plt.plot(Y_test_actual, label='Actual')
plt.plot(Y_pred_actual, label='Predicted')
plt.legend()
plt.show()