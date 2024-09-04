import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Load and preprocess the images
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = image / 255.0  # Normalize the image
    return image

# Define a deeper CNN model for the Siamese Network
def create_deep_cnn_model(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    return Model(input, x)

# Function to generate Grad-CAM
def generate_gradcam(model, img_tensor, layer_name):
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    guided_grads = tf.cast(conv_outputs > 0, "float32") * tf.cast(grads > 0, "float32") * grads

    conv_outputs = conv_outputs[0]
    guided_grads = guided_grads[0]

    weights = tf.reduce_mean(guided_grads, axis=(0, 1))

    cam = np.zeros(conv_outputs.shape[0:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    # Resize CAM to match the input image's resolution
    cam = cv2.resize(cam.numpy(), (img_tensor.shape[2], img_tensor.shape[1]))
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    return cam

# Load images
image1_path = 'screenshot1.png'
image2_path = 'screenshot3.png'

image1 = load_and_preprocess_image(image1_path)
image2 = load_and_preprocess_image(image2_path)

# Expand dimensions to create batch size of 1
image1 = np.expand_dims(image1, axis=0)
image2 = np.expand_dims(image2, axis=0)

# Assuming your CNN model creation is already defined
input_shape = image1.shape[1:]
model_cnn = create_deep_cnn_model(input_shape)

# Generate Grad-CAM for each image
gradcam_image1 = generate_gradcam(model_cnn, image1, 'conv2d_4')  # Last convolutional layer
gradcam_image2 = generate_gradcam(model_cnn, image2, 'conv2d_4')  # Last convolutional layer

# Calculate the difference between the Grad-CAMs
difference_cam = np.abs(gradcam_image1 - gradcam_image2)

# Normalize the difference
difference_cam = (difference_cam - np.min(difference_cam)) / (np.max(difference_cam) - np.min(difference_cam))

# Overlay the difference map on one of the original images
overlay = cv2.applyColorMap(np.uint8(255 * difference_cam), cv2.COLORMAP_JET)
overlayed_image = cv2.addWeighted((image1[0] * 255).astype(np.uint8), 0.6, overlay, 0.4, 0)

# Plot the results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor((image1[0] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.title('Original Image 1')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor((image2[0] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.title('Original Image 2')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB))
plt.title('Difference Highlighted')

# Save the combined output with full resolution
plt.savefig('gradcam_difference_full_res_deep2.png')
plt.close()
