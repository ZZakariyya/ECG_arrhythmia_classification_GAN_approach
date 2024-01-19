from GAN_demo import Generator, Discriminator, train_gan, augment_data
from data_preparation import load_and_process_data

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# Initialize Generator and Discriminator
window_size = 300  
noise_dim = 100   
generator = Generator(window_size, noise_dim)
discriminator = Discriminator(window_size)


data_path = 'E:/PHD Projects/Project 2/mitdb'
data, labels = load_and_process_data(data_path)

# Train the GAN
epochs = 1  
batch_size = 32  
train_gan(generator, discriminator, data, epochs, batch_size, noise_dim)

# Determine the number of synthetic samples to generate for each class
num_samples = {
    0: 10,  
    1: 10,
    2: 10,
    3: 10,
    4: 10
}

augmented_data, augmented_labels = augment_data(generator, data, labels, num_samples, noise_dim)

X_train, X_test, y_train, y_test = train_test_split(augmented_data, augmented_labels, test_size=0.2, random_state=42)

# Define the model
model = Sequential()

# CNN layers
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(300, 1)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.5))

# LSTM layers
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dropout(0.5))

# Fully connected layers
model.add(Dense(100, activation='relu'))
model.add(Dense(5, activation='softmax'))  # Assuming 5 classes for your problem

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model

early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=5, verbose=1, mode='auto')

# Train the model with a specified number of epochs
model.fit(X_train, y_train, epochs=1, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)[1]
print("Accuracy:", accuracy)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Generate and print the classification report
report = classification_report(y_test, y_pred_classes, target_names=['N', 'V', 'S', 'F', 'Other'])
print(report)

model.save('E:/PHD Projects/Project 1/CNN_RNN_GAN_model.h5')
