import os
import csv
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from early_stopping import MultiMetricEarlyStopping

# Check if TensorFlow is using the GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Optional: To force TensorFlow to use a specific GPU
#tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# 1. Conv1D: Increasing Filters Across Layers:
def increasing_filters_layers(num_layers, start_filters=32, increase_factor=2):
    return [start_filters * (increase_factor ** i) for i in range(num_layers)]

# 2. Conv1D: Decreasing Filters Across Layers:
def decreasing_filters_layers(num_layers, start_filters=256, decrease_factor=2):
    return [max(start_filters // (decrease_factor ** i), 1) for i in range(num_layers)]

# 3. Conv1D: Constant Filters Across Layers:
def constant_filters_layers(num_layers, filters=64):
    return [filters] * num_layers

def calculate_start_filters(num_layers):
    """
    Calculate the start_filters based on the number of Conv1D layers.
    
    Parameters:
    num_layers (int): The number of Conv1D layers in the network.

    Returns:
    int: The calculated start_filters.
    """
    if num_layers <= 5:
        # Shallow network (1-5 layers)
        start_filters = 64
    elif 6 <= num_layers <= 10:
        # Moderate-depth network (6-10 layers)
        start_filters = 32
    elif 11 <= num_layers <= 20:
        # Deep network (11-20 layers)
        start_filters = 16
    else:
        raise ValueError("num_layers should be between 1 and 20.")

    return start_filters

def dropout_rate_generator(start_percent, end_percent):
    return (round(rate / 100.0, 2) for rate in range(int(start_percent * 100), int(end_percent * 100) + 1))

def calculate_start_units(num_layers):
    """
    Calculate the start_units for the first dense layer based on the number of dense layers.

    Parameters:
    num_layers (int): The number of dense layers in the network.

    Returns:
    int: The calculated start_units.
    """
    if num_layers <= 5:
        # Shallow network (1-5 layers)
        start_units = 256
    elif 6 <= num_layers <= 10:
        # Moderate-depth network (6-10 layers)
        start_units = 512
    elif 11 <= num_layers <= 20:
        # Deep network (11-20 layers)
        start_units = 32768
    else:
        raise ValueError("num_layers should be between 1 and 20.")

    return start_units

# 1. Dense: Exponential Decay Pattern
def exponential_decay_layers(num_layers, min_units=1):
    # Calculate the start units based on the number of layers
    start_units = calculate_start_units(num_layers)
    
    layers = []
    units = start_units
    for i in range(num_layers):
        layers.append(units)
        units = max(units // 2, min_units)  # Halve the units at each step, ensure it doesn't go below min_units
    return layers

# 2. Dense: Linear Decay Pattern
def linear_decay_layers(num_layers, start_units=2048, decrease_amount=200):
    layers = []
    units = start_units
    for i in range(num_layers):
        layers.append(units)
        units = max(units - decrease_amount, 1)  # Ensure units don't go below 1
    return layers

# 3. Dense: Plateauing Pattern
def plateauing_layers(num_layers, start_units=2048, plateau_length=2, decrease_factor=2):
    layers = []
    units = start_units
    for i in range(num_layers):
        layers.append(units)
        if (i + 1) % plateau_length == 0:  # Decrease units every 'plateau_length' layers
            units = max(units // decrease_factor, 1)  # Ensure units don't go below 1
    return layers

def calculate_percentage_difference(value1, value2):
    """
    Calculate the percentage difference between two values.

    Parameters:
    value1 (float): The first value (e.g., training accuracy or loss).
    value2 (float): The second value (e.g., validation accuracy or loss).

    Returns:
    float: The percentage difference between the two values.
    """
    return abs((value1 - value2) / value1) * 100

# Load the dataset
train_data = pd.read_csv('datasets/NSL-KDD/KDDTrain+.txt', header=None)

# Define column names
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

train_data.columns = columns

# Encode categorical features and label
categorical_features = ['protocol_type', 'service', 'flag']
train_data = pd.get_dummies(train_data, columns=categorical_features)

# Separate features and labels
X = train_data.drop(columns=['label', 'difficulty'])
y = train_data['label']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Reshape data for Conv1D: (samples, steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

log = open("cnn_model_data_generator.txt", "w")
log.write("Training shape: " + str(X_train.shape) + "\r\n")
log.write("Validation shape: " + str(X_val.shape) + "\r\n")
log.flush()
log.close()

# Define the 1D CNN model
input_shape = (X_train.shape[1], 1)

epochs = 20
conv_activation_fn = 'relu'
fullconn_activation_fn = 'relu'
out_activation_fn = 'softmax'

max_conv_layers = 5
max_dense_layers = 5
# max_kernel_size = X_train.shape[1]
max_kernel_size = 5
max_pool_size = 5

convolution_layers = [x for x in range(1, max_conv_layers + 1)]
kernel_sizes = [x for x in range(1, max_kernel_size + 1)]
pool_sizes = [x for x in range(0, max_pool_size + 1)]
# Generate dropout rates
conv_dropout_rates = dropout_rate_generator(0.1, 0.3)
flat_dropout_rates = dropout_rate_generator(0.4, 0.6)
fullconn_dropout_rates = dropout_rate_generator(0.4, 0.6)
fullconn_layers = [x for x in range(1, max_conv_layers + 1)]

for convolution_layer_size in convolution_layers:
    for kernel_size in kernel_sizes:
        for pool_size in pool_sizes:
            for fullconn_layer_size in fullconn_layers:
                model = Sequential()
                model.add(Input(shape=input_shape))
                start_filters = calculate_start_filters(convolution_layer_size)
                filters = increasing_filters_layers(convolution_layer_size, start_filters)
                for filter_size in filters:
                    model.add(Conv1D(filters=filter_size,
                                    kernel_size=kernel_size,
                                    activation=conv_activation_fn))
                    model.add(MaxPooling1D(pool_size=pool_size))

                model.add(Flatten())

                layers = exponential_decay_layers(fullconn_layer_size)
                for dense_layer in layers:
                    # Fully connected layer with x neurons
                    model.add(Dense(dense_layer, activation=fullconn_activation_fn))

                # Output layer with softmax activation for multi-class classification
                model.add(Dense(len(label_encoder.classes_), activation=out_activation_fn))

                # Compile the model
                model.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

                # Train the model
                # early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                multi_metric_early_stopping = MultiMetricEarlyStopping(val_loss_patience=3, val_accuracy_patience=3)
                start_time = time.time()
                history = model.fit(X_train, y_train,
                                    epochs=epochs,
                                    validation_data=(X_val, y_val),
                                    callbacks=[multi_metric_early_stopping])
                end_time = time.time()
                # Calculate the elapsed time
                elapsed_time = end_time - start_time
                # Print the elapsed time in seconds
                print(f"Operation took {elapsed_time:.4f} seconds")

                # Get the maximum accuracy from training history
                max_train_accuracy = max(history.history['accuracy'])
                max_val_accuracy = max(history.history['val_accuracy'])
                max_train_loss = max(history.history['loss'])
                max_val_loss = max(history.history['val_loss'])

                final_training_accuracy = history.history['accuracy'][-1]
                final_validation_accuracy = history.history['val_accuracy'][-1]
                final_training_loss = history.history['loss'][-1]
                final_validation_loss = history.history['val_loss'][-1]

                # Calculate percentage differences
                accuracy_difference = calculate_percentage_difference(final_training_accuracy, final_validation_accuracy)
                loss_difference = calculate_percentage_difference(final_training_loss, final_validation_loss)

                # Evaluate the model
                loss, accuracy = model.evaluate(X_val, y_val)
                
                log = open("cnn_model_data_generator.txt", "a")
                log.write(f"Conv layers: {0}, kernels: {1}, pools: {2}, dense layers: {3}\r\n"
                          .format(convolution_layer_size, kernel_size, pool_size, fullconn_layer_size))
                log.write(f"filters: {0}\r\n".format(str(filters)))

                # Check if early stopping was triggered
                if multi_metric_early_stopping.early_stopped:
                    log.write("The model was early stopped.\r\n")

                # Get the number of epochs that were actually executed
                num_epochs_executed = len(history.history['loss'])
                # Log the accuracy/loss and validation accuracy/loss values from the history object
                for epoch in range(num_epochs_executed):
                    accuracy = history.history['loss'][epoch]
                    val_accuracy = history.history['val_loss'][epoch]
                    train_loss = history.history['loss'][epoch]
                    val_loss = history.history['val_loss'][epoch]
                    log.write(f"Epoch {epoch+1}/{epochs}\r\n")
                    log.write(f"    Accuracy: {accuracy:.4f} - Validation Accuracy: {val_accuracy:.4f}\r\n")
                    log.write(f"    Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}\r\n")

                log.write(f'Maximum Training Accuracy: {max_train_accuracy:.4f}\r\n')
                log.write(f'Maximum Validation Accuracy: {max_val_accuracy:.4f}\r\n')
                log.write(f'Maximum Training Loss: {max_train_loss:.4f}\r\n')
                log.write(f'Maximum Validation Loss: {max_val_loss:.4f}\r\n')

                log.write(f"Final Training Accuracy: {final_training_accuracy:.4f}\r\n")
                log.write(f"Final Validation Accuracy: {final_validation_accuracy:.4f}\r\n")
                log.write(f"Final Training Loss: {final_training_loss:.4f}\r\n")
                log.write(f"Final Validation Loss: {final_validation_loss:.4f}\r\n")

                log.write(f'Validation Accuracy: {accuracy:.4f}\r\n')
                log.write(f'Validation Loss: {loss:.4f}\r\n')

                log.flush()
                log.close()

                # After training, extract the relevant data:
                experiment_data = {
                    'conv_layers': convolution_layer_size,
                    'kernel_size': kernel_size,
                    'pool_size': pool_size,
                    'filters': 0,
                    'dense_layers': fullconn_layer_size,
                    'dense_neurons': str([128, 256, 512]),  # adjust if you change this
                    'epochs': epochs,
                    'batch_size': 32,  # if varied, add as a parameter
                    'learning_rate': 0.001,  # if varied, add as a parameter
                    'optimizer': 'adam',  # as per your code
                    'train_loss': history.history['loss'][-1],
                    'val_loss': history.history['val_loss'][-1],
                    'train_accuracy': history.history['accuracy'][-1],
                    'val_accuracy': history.history['val_accuracy'][-1],
                    'epoch_train_loss': str(history.history['loss']),
                    'epoch_val_loss': str(history.history['val_loss']),
                    'epoch_train_accuracy': str(history.history['accuracy']),
                    'epoch_val_accuracy': str(history.history['val_accuracy']),
                }

                # Define CSV file and headers
                csv_file = 'model_experiments.csv'
                headers = list(experiment_data.keys())

                # Write or append to CSV file
                file_exists = os.path.isfile(csv_file)

                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=headers)
                    
                    if not file_exists:
                        writer.writeheader()  # write header only once
                    
                    writer.writerow(experiment_data)