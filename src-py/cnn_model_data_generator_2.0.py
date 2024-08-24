import os
import csv
import time
import multiprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input, Dropout
from tensorflow.keras import backend as K
from early_stopping import MultiMetricEarlyStopping

# Check if TensorFlow is using the GPU
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
    if num_layers >= 1 and num_layers <= 5:
        # Shallow network (1-5 layers)
        start_units = 16
    elif 6 <= num_layers <= 10:
        # Moderate-depth network (6-10 layers)
        start_units = 128
    elif 11 <= num_layers <= 20:
        # Deep network (11-20 layers)
        start_units = 32768
    else:
        raise ValueError("num_layers should be between 1 and 20.")

    return start_units

# 1. Dense: Exponential Decay Pattern
def exponential_decay_layers(num_layers, min_units=64):
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

def f1_score(y_true, y_pred):
    # Convert predictions to binary values
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    
    # Calculate true positives, false positives, and false negatives
    true_positives = tf.reduce_sum(tf.cast(y_true * y_pred, 'float32'))
    predicted_positives = tf.reduce_sum(tf.cast(y_pred, 'float32'))
    possible_positives = tf.reduce_sum(tf.cast(y_true, 'float32'))
    
    # Calculate precision and recall
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    
    # Calculate F1 score
    f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1

def log_to_dataset(index, conv_layers, kernel_size, pool_size, dense_layers, filters, neurons,
            training_accuracy_last, val_accuracy_last, training_loss_last, val_loss_last,
            #precision_last, val_precision_last, recall_last, val_recall_last, f1_score_last, val_f1_score_last, auc_last,
            train_accuracy_max, val_accuracy_max, train_loss_max, val_loss_max,
            #precision_max, val_precision_max, recall_max, val_recall_max, f1_score_max, val_f1_score_max, auc_max,
            eval_accuracy, eval_loss, #eval_precision, eval_recall,
            early_stopped, epochs_max, epochs_executed, cnn_config_possible, execution_time):
    # After training, extract the relevant data:
    experiment_data = {
        'conv_layers': conv_layers,
        'kernel_size': kernel_size,
        'pool_size': pool_size,
        'dense_layers': dense_layers,
        'filters_1': filters[0] if len(filters) >= 1 else 0,
        'filters_2': filters[1] if len(filters) >= 2 else 0,
        'filters_3': filters[2] if len(filters) >= 3 else 0,
        'filters_4': filters[3] if len(filters) >= 4 else 0,
        'filters_5': filters[4] if len(filters) >= 5 else 0,
        'dense_neurons_1': neurons[0] if len(neurons) >= 1 else 0,
        'dense_neurons_2': neurons[1] if len(neurons) >= 2 else 0,
        'dense_neurons_3': neurons[2] if len(neurons) >= 3 else 0,
        'dense_neurons_4': neurons[3] if len(neurons) >= 4 else 0,
        'dense_neurons_5': neurons[4] if len(neurons) >= 5 else 0,
        'train_accuracy_max': train_accuracy_max,
        'val_accuracy_max': val_accuracy_max,
        'train_loss_max': train_loss_max,
        'val_loss_max': val_loss_max,
        #'precision_max': precision_max,
        #'val_precision_max': val_precision_max,
        #'recall_max': recall_max,
        #'val_recall_max': val_recall_max,
        #'f1_score_max': f1_score_max,
        #'val_f1_score_max': val_f1_score_max,
        #'auc_max': auc_max,
        'train_accuracy_last': training_accuracy_last,
        'val_accuracy_last': val_accuracy_last,
        'train_loss_last': training_loss_last,
        'val_loss_last': val_loss_last,
        #'precision_last': precision_last,
        #'val_precision_last': val_precision_last,
        #'recall_last': recall_last,
        #'val_recall_last': val_recall_last,
        #'f1_score_last': f1_score_last,
        #'val_f1_score_last': val_f1_score_last,
        #'auc_last': auc_last,
        'eval_accuracy': eval_accuracy,
        'eval_loss': eval_loss,
        #'eval_precision': eval_precision,
        #'eval_recall': eval_recall,
        'epochs_max': epochs_max,
        'epochs_executed': epochs_executed,
        'cnn_config_possible': cnn_config_possible,
        'early_stopped': early_stopped,
        'execution_time': execution_time
    }

    # Define CSV file and headers
    csv_file = f'model_experiments{0}.csv'.format(index)
    headers = list(experiment_data.keys())

    # Write or append to dataset CSV file
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        
        if not file_exists:
            writer.writeheader()  # write header only once
        
        writer.writerow(experiment_data)

# Load the dataset
train_data = pd.read_csv('KDDTrain+.txt', header=None)

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

log = open("cnn_model_data_generator.txt", "a")
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
max_kernel_size = 5
max_pool_size = 5

conv_start = 1
kernel_start = 1
pool_start = 1
dense_start = 1

convolution_layers = [x for x in range(conv_start, max_conv_layers + 1)]
kernel_sizes = [x for x in range(kernel_start, max_kernel_size + 1)]
pool_sizes = [x for x in range(pool_start, max_pool_size + 1)]
fullconn_layers = [x for x in range(dense_start, max_dense_layers + 1)]

def run_nas(index, start_filters, start_layers, dropout_rate):
    count = 0
    done = max_conv_layers * max_dense_layers * max_kernel_size * max_pool_size
    for conv_layers in convolution_layers:
        for kernel_size in kernel_sizes:
            for pool_size in pool_sizes:
                for dense_layers in fullconn_layers:
                    count = count + 1
                    log = open(f"cnn_model_data_generator_{0}.txt".format(index + 1), "a")
                    filters = 0
                    layers = 0
                    try:
                        print("conv_layers: " + str(conv_layers) + ", kernel_size: " + str(kernel_size)
                            + ", pool_size: " + str(pool_size) + ", fullconn_layer_size: " + str(dense_layers))
                        print("Progress: [" + str(count) + "/" + str(done) + "] " + str((count / done) * 100) + "%")
                        log.write("conv_layers: " + str(conv_layers) + ", kernel_size: " + str(kernel_size)
                            + ", pool_size: " + str(pool_size) + ", fullconn_layer_size: " + str(dense_layers) + "\n")
                        log.write("[" + str(index) + "] Progress: [" + str(count) + "/" + str(done) + "] " + str((count / done) * 100) + "%\n")
                        model = Sequential()
                        model.add(Input(shape=input_shape))
                        filters = increasing_filters_layers(conv_layers, start_filters)
                        print("filters: " + str(filters))
                        log.write("filters: " + str(filters) + "\n")
                        for filter_size in filters:
                            model.add(Conv1D(filters=filter_size,
                                            kernel_size=kernel_size,
                                            activation=conv_activation_fn))
                            print("Ootput shape Conv1D:", model.output_shape)
                            model.add(MaxPooling1D(pool_size=pool_size))
                            print("Output shape MaxPooling1D:", model.output_shape)
                            model.add(Dropout(dropout_rate))

                        model.add(Flatten())
                        print("Output shape Flatten:", model.output_shape)
                        model.add(Dropout(dropout_rate))

                        layers = list(reversed(increasing_filters_layers(conv_layers, start_layers)))
                        print("dense layers: " + str(layers))
                        log.write("dense layers: " + str(layers) + "\n")
                        for dense_layer in layers:
                            # Fully connected layer with x neurons
                            model.add(Dense(dense_layer, activation=fullconn_activation_fn))
                            print("Output shape Dense:", model.output_shape)
                            model.add(Dropout(dropout_rate))

                        # Output layer with softmax activation for multi-class classification
                        model.add(Dense(len(label_encoder.classes_), activation=out_activation_fn))
                        print("Final output shape Dense:", model.output_shape)

                        # Compile the model
                        model.compile(optimizer='adam',
                                    loss='sparse_categorical_crossentropy',
                                    metrics=['accuracy'])

                        # Train the model
                        # early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                        # multi_metric_early_stopping = MultiMetricEarlyStopping(val_loss_patience=10, val_accuracy_patience=10)
                        start_time = time.time()
                        history = model.fit(X_train, y_train,
                                            epochs=epochs,
                                            validation_data=(X_val, y_val),
                                            # callbacks=[multi_metric_early_stopping]
                        )
                        end_time = time.time()
                        # Calculate the elapsed time
                        elapsed_time = end_time - start_time
                        # Print the elapsed time in seconds
                        print(f"Operation took {elapsed_time:.4f} seconds")
                        log.write(f"Operation took {elapsed_time:.4f} seconds\n")

                        # Access precision and recall from the history object
                        train_accuracy_max = max(history.history['accuracy'])
                        val_accuracy_max = max(history.history['val_accuracy'])
                        train_loss_max = max(history.history['loss'])
                        val_loss_max = max(history.history['val_loss'])
                        #precision_max = max(history.history['precision'])
                        #val_precision_max = max(history.history['val_precision'])
                        #recall_max = max(history.history['recall'])
                        #val_recall_max = max(history.history['val_recall'])
                        #f1_score_max = max(history.history['f1_score'])
                        #val_f1_score_max = max(history.history['val_f1_score'])
                        #auc_max = max(history.history['auc'])

                        # Access performance parameters from the history object
                        training_accuracy_last = history.history['accuracy'][-1]
                        validation_accuracy_last = history.history['val_accuracy'][-1]
                        training_loss_last = history.history['loss'][-1]
                        validation_loss_last = history.history['val_loss'][-1]
                        #precision_last = history.history['precision'][-1]
                        #val_precision_last = history.history['val_precision'][-1]
                        #recall_last = history.history['recall'][-1]
                        #val_recall_last = history.history['val_recall'][-1]
                        #f1_score_last = history.history['f1_score'][-1]
                        #val_f1_score_last = history.history['val_f1_score'][-1]
                        #auc_last = history.history['auc'][-1]

                        # Calculate percentage differences
                        accuracy_difference = calculate_percentage_difference(training_accuracy_last, validation_accuracy_last)
                        loss_difference = calculate_percentage_difference(training_loss_last, validation_loss_last)

                        # Evaluate the model
                        # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

                        validation_loss, validation_accuracy = model.evaluate(X_val, y_val)
                        print("history: " + str(history.history.keys()))
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                        #precision = precision_score(y_val, y_pred)
                        #print("Precision:", precision)
                        print("=========" )

                        # Get the number of epochs that were actually executed
                        epochs_executed = len(history.history['loss'])
                        # Log the accuracy/loss and validation accuracy/loss values from the history object
                        for epoch in range(epochs_executed):
                            accuracy = history.history['loss'][epoch]
                            val_accuracy = history.history['val_loss'][epoch]
                            train_loss = history.history['loss'][epoch]
                            val_loss = history.history['val_loss'][epoch]
                            log.write(f"Epoch {epoch+1}/{epochs}\r\n")
                            log.write(f"    Accuracy: {accuracy:.4f} - Validation Accuracy: {val_accuracy:.4f}\n")
                            log.write(f"    Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}\n")

                        log.flush()
                        log.close()

                        early_stopped = 0 # 1 if multi_metric_early_stopping.early_stopped else 0
                        log_to_dataset(index, conv_layers, kernel_size, pool_size, dense_layers, filters, layers,
                                    training_accuracy_last, validation_accuracy_last, training_loss_last, validation_loss_last,
                                    #precision_last, val_precision_last, recall_last, val_recall_last, f1_score_last, val_f1_score_last, auc_last,
                                    train_accuracy_max, train_loss_max, val_accuracy_max, val_loss_max,
                                    #precision_max, val_precision_max, recall_max, val_recall_max, f1_score_max, val_f1_score_max, auc_max,
                                    validation_accuracy, validation_loss, #precision, recall,
                                    epochs, epochs_executed, 1, early_stopped, int(elapsed_time))

                    except Exception as e:
                        print(f"An error occurred: {e}")
                        cnn_config_possible = 0
                        log_to_dataset(index, conv_layers, kernel_size, pool_size, start_layers, filters, layers,
                                    0, 0, 0, 0,
                                    #0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0,
                                    #0, 0, 0, 0, 0, 0, 0,
                                    0, 0, #0, 0,
                                    epochs, 0, cnn_config_possible, 0, 0)
                        try:
                            log.write(f"An error occurred: {e}\n")
                            log.flush()
                            log.close()
                        except Exception as e:
                            print(f"An error occurred: {e}")
                    
if __name__ == "__main__":
    start_filters = [1, 2, 4, 8, 16, 32, 64, 128]
    start_layers = [16, 32, 64, 128, 256, 512, 1024, 2048]
    dropout_rates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    processes = []
    run_nas(0, start_filters[0], start_layers[0], dropout_rates[0])
    #for index in range(0, 1):
    #    p = multiprocessing.Process(target=run_nas, args=(index, start_filters[index], start_layers[index], dropout_rates[index],))
    #    p.start()
    #    processes.append(p)

    #for p in processes:
    #    p.join()