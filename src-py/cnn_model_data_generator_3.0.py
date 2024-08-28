import os
import csv
import time
import threading
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, MaxPooling1D, Flatten, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, f1_score
from early_stopping import MultiMetricEarlyStopping

class CustomTrainingCallback(tf.keras.callbacks.Callback):
        def __init__(self, index):
            self.run_index = index
        
        def on_epoch_end(self, epoch, logs=None):
            print(f"[{self.run_index}] Epoch {epoch+1}: Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}, Val_Loss: {logs.get('val_loss', 'N/A'):.4f}, Val_Accuracy: {logs.get('val_accuracy', 'N/A'):.4f}")

        def on_batch_end(self, batch, logs=None):
            print(f"[{self.run_index}] Batch {batch+1}: Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")

class NAS:
    
    def __init__(self, run_index):
        self.run_index = run_index
        self.strategy = tf.distribute.MirroredStrategy()
        self.lock = threading.Lock()
        self.log = open("cnn_model_data_generator_" + str(self.run_index) + ".txt", 'w')
        # Check if TensorFlow is using the GPU
        self.logMessage(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

        # Load the dataset
        self.train_data = pd.read_csv('datasets/NSL-KDD/KDDTrain+.txt', header=None)
        # Define column names
        self.columns = [
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

        self.train_data.columns = self.columns

        # Encode categorical features and label
        self.categorical_features = ['protocol_type', 'service', 'flag']
        self.train_data = pd.get_dummies(self.train_data, columns=self.categorical_features)

        # Separate features and labels
        self.X = self.train_data.drop(columns=['label', 'difficulty'])
        self.y = self.train_data['label']

        # Split the data into training and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)

        # Encode labels
        self.label_encoder = LabelEncoder()
        self.y_train = self.label_encoder.fit_transform(self.y_train)
        self.y_val = self.label_encoder.transform(self.y_val)

        # Standardize features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)

        # Reshape data for Conv1D: (samples, steps, features)
        self.X_train = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
        self.X_val = self.X_val.reshape((self.X_val.shape[0], self.X_val.shape[1], 1))

        self.logMessage(f"Training shape: {str(self.X_train.shape)}")
        self.logMessage(f"Validation shape: {str(self.X_val.shape)}")

        # Define the 1D CNN model
        self.input_shape = (self.X_train.shape[1], 1)

        self.learning_rate = 0.001
        self.dropout_rate = 0.0
        self.epochs = 50
        self.batch_size = 1024
        self.conv_activation_fn = 'relu'
        self.fullconn_activation_fn = 'relu'
        self.out_activation_fn = 'softmax'

        self.max_conv_layers = 5
        self.max_dense_layers = 5
        self.max_kernel_size = 5
        self.max_pool_size = 5

        conv_start = 1
        kernel_start = 1
        pool_start = 1
        dense_start = 1

        self.convolution_layers = [x for x in range(conv_start, self.max_conv_layers + 1)]
        self.kernel_sizes = [x for x in range(kernel_start, self.max_kernel_size + 1)]
        self.pool_sizes = [x for x in range(pool_start, self.max_pool_size + 1)]
        self.fullconn_layers = [x for x in range(dense_start, self.max_dense_layers + 1)]
    
    def __del__(self):
        if self.log != None:
            self.log.close()
    
    def run_nas(self, start_filters, start_layers, dropout_rate):
        with self.strategy.scope():
            count = 0
            done = self.max_conv_layers * self.max_dense_layers * self.max_kernel_size * self.max_pool_size
            for conv_layers in self.convolution_layers:
                for kernel_size in self.kernel_sizes:
                    for pool_size in self.pool_sizes:
                        for dense_layers in self.fullconn_layers:
                            count = count + 1
                            history = []
                            filters = []
                            layers = []
                            try:
                                self.logMessage(f"Progress: [{str(count)}/{str(done)}] {str((count / done) * 100)}%")
                                self.logMessage(f"conv_layers: {str(conv_layers)} kernel_size: {str(kernel_size)} "
                                    + f"pool_size: {str(pool_size)} dense_layers: {str(dense_layers)}")
                                model = Sequential()
                                model.add(Input(shape=self.input_shape))
                                filters = self.increasing_filters_layers(conv_layers, start_filters)
                                self.logMessage(f"filters: {str(filters)}")
                                for filter_size in filters:
                                    # Add a convolutional layer
                                    model.add(Conv1D(filters=filter_size,
                                                    kernel_size=kernel_size,
                                                    activation=self.conv_activation_fn))
                                    self.logMessage(f"Output shape Conv1D: {str(model.output_shape)}")
                                    # Add batch normalization
                                    model.add(BatchNormalization())
                                    # Add ReLU activation
                                    model.add(ReLU())

                                    model.add(MaxPooling1D(pool_size=pool_size))
                                    self.logMessage(f"Output shape MaxPooling1D: {str(model.output_shape)}")
                                    model.add(Dropout(dropout_rate))

                                model.add(Flatten())
                                self.logMessage(f"Output shape Flatten: {str(model.output_shape)}")
                                model.add(Dropout(dropout_rate))
                                self.logMessage(f"Output shape Dropout: {str(model.output_shape)}")

                                layers = list(reversed(self.increasing_filters_layers(conv_layers, start_layers)))
                                self.logMessage(f"dense layers: {str(layers)}")
                                for dense_layer in layers:
                                    # Fully connected layer with x neurons
                                    model.add(Dense(dense_layer, activation=self.fullconn_activation_fn))
                                    self.logMessage(f"Output shape Dense: {str(model.output_shape)}")
                                    model.add(Dropout(dropout_rate))
                                    self.logMessage(f"Output shape Dropout: {str(model.output_shape)}")

                                # Output layer with softmax activation for multi-class classification
                                model.add(Dense(len(self.label_encoder.classes_), activation=self.out_activation_fn))
                                self.logMessage(f"Final output shape Dense: {str(model.output_shape)}")

                                # Train the model
                                # early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                                # multi_metric_early_stopping = MultiMetricEarlyStopping(val_loss_patience=10, val_accuracy_patience=10)
                                with self.lock:
                                    # Compile the model
                                    optimizer = Adam(learning_rate=self.learning_rate)
                                    model.compile(optimizer=optimizer,
                                                loss='sparse_categorical_crossentropy',
                                                metrics=['accuracy'])
                                
                                    start_time = time.time()
                                    history = model.fit(self.X_train, self.y_train,
                                                        epochs=self.epochs,
                                                        validation_data=(self.X_val, self.y_val),
                                                        batch_size = self.batch_size,
                                                        callbacks=[CustomTrainingCallback(self.run_index)]
                                                        # callbacks=[multi_metric_early_stopping]
                                    )
                                    end_time = time.time()
                                    # Calculate the elapsed time
                                    elapsed_time = end_time - start_time
                                    # Print the elapsed time in seconds
                                    self.logMessage(f"Operation took {elapsed_time:.4f} seconds", False)
                                    # Print the batch size used
                                    # print(f"Batch size used: {model.batch_size}")
                                    
                                    # Evaluate the model
                                    validation_loss, validation_accuracy = model.evaluate(self.X_val, self.y_val)
                                    self.logMessage(f"History: {str(history.history.keys())}", False)
                                    # Calculate precision, recall, and F1 score
                                    y_pred = model.predict(self.X_val)
                                    y_pred_classes = np.argmax(y_pred, axis=1)
                                    precision = precision_score(self.y_val, y_pred_classes, average='weighted', zero_division=0)
                                    recall = recall_score(self.y_val, y_pred_classes, average='weighted', zero_division=0)
                                    f1 = f1_score(self.y_val, y_pred_classes, average='weighted')

                                    # Log precision, recall, and F1 score
                                    self.logMessage(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}", False)

                                    # Get the number of epochs that were actually executed
                                    epochs_executed = len(history.history['loss'])

                                    early_stopped = 0 # 1 if multi_metric_early_stopping.early_stopped else 0
                                    self.log_to_dataset(history, conv_layers, kernel_size, pool_size, dense_layers, filters, layers,
                                                validation_accuracy, validation_loss, precision, recall, f1,
                                                early_stopped, self.epochs, epochs_executed, self.batch_size, 1, int(elapsed_time))

                            except Exception as e:
                                self.logMessage(f"An error occurred: {e}")
                                self.log_to_dataset(history, conv_layers, kernel_size, pool_size, dense_layers, filters, layers,
                                            0, 0, 0, 0, 0,
                                            0, self.epochs, 0, 0, 0, 0)
                                self.logMessage(f"An error occurred: {e}")
                            

    def log_to_dataset(self, history, conv_layers, kernel_size, pool_size, dense_layers, filters, neurons,
                eval_accuracy, eval_loss, precision, recall, f1,
                early_stopped, epochs_max, epochs_executed, batch_size, cnn_config_possible, execution_time, err=""):
        '''After training, extract the relevant data'''
        
        # FEATURES
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
            'epochs_max': epochs_max,
            'epochs_executed': epochs_executed,
            'batch_size': batch_size,
            'learning_rate': self.learning_rate,
            'dropout_rate': self.dropout_rate,
            'early_stopped': early_stopped,
            'cnn_config_possible': cnn_config_possible,
            'execution_time': execution_time,
        }

        # TARGETS
        if hasattr(history, 'history'):
            experiment_data['train_accuracy_max'] = max(history.history['accuracy']) if len(history.history['accuracy']) >= 1 else 0
            experiment_data['val_accuracy_max'] = max(history.history['val_accuracy']) if len(history.history['val_accuracy']) >= 1 else 0
            experiment_data['train_loss_max'] = max(history.history['loss']) if len(history.history['loss']) >= 1 else 0
            experiment_data['val_loss_max'] = max(history.history['val_loss']) if len(history.history['val_loss']) >= 1 else 0
            experiment_data['train_accuracy_last'] = history.history['accuracy'][-1] if len(history.history['accuracy']) >= 1 else 0
            experiment_data['val_accuracy_last'] = history.history['val_accuracy'][-1] if len(history.history['val_accuracy']) >= 1 else 0
            experiment_data['train_loss_last'] = history.history['loss'][-1] if len(history.history['loss']) >= 1 else 0
            experiment_data['val_loss_last'] = history.history['val_loss'][-1] if len(history.history['val_loss']) >= 1 else 0
            experiment_data['eval_accuracy'] = eval_accuracy
            experiment_data['eval_loss'] = eval_loss
            experiment_data['precision'] = precision
            experiment_data['recall'] = recall
            experiment_data['f1'] = f1
        else:
            experiment_data['train_accuracy_max'] = 0
            experiment_data['val_accuracy_max'] = 0
            experiment_data['train_loss_max'] = 0
            experiment_data['val_loss_max'] = 0
            experiment_data['train_accuracy_last'] = 0
            experiment_data['val_accuracy_last'] = 0
            experiment_data['train_loss_last'] = 0
            experiment_data['val_loss_last'] = 0
            experiment_data['eval_accuracy'] = 0
            experiment_data['eval_loss'] = 0
            experiment_data['precision'] = 0
            experiment_data['recall'] = 0
            experiment_data['f1'] = 0

        if  not hasattr(history, 'history'):
            for index in range(self.epochs):
                experiment_data[f'hist_accuracy_{index}'] = 0
                experiment_data[f'hist_val_accuracy_{index}'] = 0
                experiment_data[f'hist_loss_{index}'] = 0
                experiment_data[f'hist_val_loss_{index}'] = 0
        else:
            for index, accuracy in enumerate(history.history['accuracy']):
                experiment_data[f'hist_accuracy_{index}'] = accuracy
            for index, val_accuracy in enumerate(history.history['val_accuracy']):
                experiment_data[f'hist_val_accuracy_{index}'] = val_accuracy
            for index, loss in enumerate(history.history['loss']):
                experiment_data[f'hist_loss_{index}'] = loss
            for index, val_loss in enumerate(history.history['val_loss']):
                experiment_data[f'hist_val_loss_{index}'] = val_loss
        
        if err != "":
            experiment_data['err'] = err

        # Define CSV file and headers
        csv_file = 'meta_data_' + str(self.run_index) + '.csv'
        headers = list(experiment_data.keys())

        # Write or append to dataset CSV file
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            
            if not file_exists:
                writer.writeheader()  # write header only once
            
            writer.writerow(experiment_data)
    
    def increasing_filters_layers(self, num_layers, start_filters=32, increase_factor=2):
        '''1. Conv1D: Increasing Filters Across Layers'''
        return [start_filters * (increase_factor ** i) for i in range(num_layers)]

    def decreasing_filters_layers(self, num_layers, start_filters=256, decrease_factor=2):
        '''2. Conv1D: Decreasing Filters Across Layers'''
        return [max(start_filters // (decrease_factor ** i), 1) for i in range(num_layers)]

    def constant_filters_layers(self, num_layers, filters=64):
        '''3. Conv1D: Constant Filters Across Layers'''
        return [filters] * num_layers

    def dropout_rate_generator(self, start_percent, end_percent):
        return (round(rate / 100.0, 2) for rate in range(int(start_percent * 100), int(end_percent * 100) + 1))

    def exponential_decay_layers(self, num_layers, min_units=64):
        '''1. Dense: Exponential Decay Pattern'''
        # Calculate the start units based on the number of layers
        start_units = self.calculate_start_units(num_layers)
        
        layers = []
        units = start_units
        for i in range(num_layers):
            layers.append(units)
            units = max(units // 2, min_units)  # Halve the units at each step, ensure it doesn't go below min_units
        return layers

    def linear_decay_layers(self, num_layers, start_units=2048, decrease_amount=200):
        '''2. Dense: Linear Decay Pattern'''
        layers = []
        units = start_units
        for i in range(num_layers):
            layers.append(units)
            units = max(units - decrease_amount, 1)  # Ensure units don't go below 1
        return layers

    def plateauing_layers(self, num_layers, start_units=2048, plateau_length=2, decrease_factor=2):
        '''3. Dense: Plateauing Pattern'''
        layers = []
        units = start_units
        for i in range(num_layers):
            layers.append(units)
            if (i + 1) % plateau_length == 0:  # Decrease units every 'plateau_length' layers
                units = max(units // decrease_factor, 1)  # Ensure units don't go below 1
        return layers
    
    def logMessage(self, message, lock = True):
        if lock:
            with self.lock:
                print(f"[{str(self.run_index)}] {message}")
                self.log.write(f"[{str(self.run_index)}] {message}\n")
                self.log.flush()
        else:
            print(f"[{str(self.run_index)}] {message}")
            self.log.write(f"[{str(self.run_index)}] {message}\n")
            self.log.flush()
                    
if __name__ == "__main__":
    # Model Architectures
    start_filters = [1, 2, 4, 8, 16, 32, 64, 128]
    start_layers = [16, 32, 64, 128, 256, 512, 1024, 2048]
    # Hyperparameters
    batch_sizes = [32, 64, 128, 512, 1024]
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    learning_rates = [0.001, 0.01, 0.0001]  # NAS Stage I

    #nas = NAS(0)
    #nas.run_nas(start_filters[0], start_layers[0], dropout_rates[0])

    threads = []
    for index in range(0, 2):  # Adjust the range as needed
        nas = NAS(index)
        t = threading.Thread(target=nas.run_nas, args=(start_filters[index], start_layers[0], dropout_rates[0]))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
