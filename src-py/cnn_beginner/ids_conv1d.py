import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input

# Load the dataset
train_data = pd.read_csv('../../train-test-data/NSL-KDD/KDDTrain+.txt', header=None)
test_data = pd.read_csv('../../train-test-data/NSL-KDD/KDDTest+.txt', header=None)

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
test_data.columns = columns

# Encode categorical features and label
categorical_features = ['protocol_type', 'service', 'flag']
train_data = pd.get_dummies(train_data, columns=categorical_features)
test_data = pd.get_dummies(test_data, columns=categorical_features)

# Align columns in train and test sets
train_data, test_data = train_data.align(test_data, join='inner', axis=1)

# Separate features and labels
X_train = train_data.drop(columns=['label', 'difficulty'])
y_train = train_data['label']
X_test = test_data.drop(columns=['label', 'difficulty'])
y_test = test_data['label']

# Combine labels from both training and testing sets for fitting the label encoder
all_labels = pd.concat([y_train, y_test])

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape data for Conv1D: (samples, steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define the model
# e.g., (41, 1) for 41 features
input_shape = (X_train.shape[1], 1)
model = Sequential()
model.add(Input(shape=input_shape))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
# For multi-class classification
model.add(Dense(len(label_encoder.classes_), activation='softmax'))
# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test Loss: {loss:.4f}')

# Print the model summary
model.summary()
