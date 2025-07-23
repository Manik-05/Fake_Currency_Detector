# Import necessary libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np

# Import deep learning framework components
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os # To check for model file existence

# Initialize the Flask application
app = Flask(__name__)
# Enable CORS for all origins, allowing the frontend to communicate with this backend
CORS(app)

# --- CNN Model Loading ---
# Define the path where the trained model is expected to be saved.
# This assumes 'cnn_model.h5' will be in the same directory as this 'app.py' file.
MODEL_PATH = 'cnn_model.h5'

model = None # Initialize model as None

try:
    # Check if the model file exists before attempting to load
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print(f"CNN model loaded successfully from {MODEL_PATH}")
    else:
        print(f"Warning: Model file '{MODEL_PATH}' not found. Using a dummy model for simulation.")
        print("Please ensure you have trained your model using 'cnn-training-script.py' and placed 'cnn_model.h5' in the same directory as 'app.py'.")
        # Create a dummy model for demonstration if the real one isn't found.
        # This dummy model will always predict 'Authentic' or 'Fake' randomly,
        # similar to the previous version, but within the TensorFlow context.
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid') # Binary classification
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # Simulate random weights for the dummy model
        model.set_weights([np.random.rand(model.weights[0].shape[0], model.weights[0].shape[1]), np.random.rand(model.weights[1].shape[0])])

except Exception as e:
    print(f"Error loading model: {e}")
    print("A dummy model will be used for predictions. Please check your model file and path.")
    # Fallback to a dummy model if loading fails for any reason
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.set_weights([np.random.rand(model.weights[0].shape[0], model.weights[0].shape[1]), np.random.rand(model.weights[1].shape[0])])


def preprocess_image_for_cnn(image_data_base64):
    """
    Decodes the base64 image data and preprocesses it for the CNN model.
    This function now performs actual image processing.
    """
    try:
        # 1. Decode base64 to bytes
        image_bytes = base64.b64decode(image_data_base64)

        # 2. Convert bytes to an image using PIL (Pillow)
        image = Image.open(io.BytesIO(image_bytes))

        # Ensure image is in RGB format (some images might be RGBA or grayscale)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 3. Resize the image to the input size expected by your CNN model (e.g., 224x224)
        # Adjust target_size if your model expects a different input shape
        target_size = (224, 224)
        image = image.resize(target_size)

        # 4. Convert image to a NumPy array and normalize pixel values (e.g., scale to 0-1)
        image_array = np.array(image) / 255.0

        # 5. Reshape the image to match the model's input shape (e.g., (1, 224, 224, 3))
        # The '1' is for the batch dimension, as models expect a batch of images.
        preprocessed_image = np.expand_dims(image_array, axis=0)

        print("Image data received and preprocessed for CNN.")
        return preprocessed_image
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

def predict_currency(preprocessed_image):
    """
    Uses the loaded CNN model to make a prediction.
    This function now performs actual model inference.
    """
    if preprocessed_image is None:
        print("No preprocessed image provided for prediction.")
        return "Error: No image"
    
    if model is None:
        print("Model not loaded. Cannot make a prediction.")
        return "Error: Model not loaded"

    try:
        # Use the loaded CNN model to make a prediction
        # The output depends on your model's last layer (e.g., sigmoid for binary, softmax for multi-class)
        predictions = model.predict(preprocessed_image)

        # Interpret the model's output
        # For a binary classification model with sigmoid activation (output between 0 and 1):
        # If prediction > 0.5, classify as 'Authentic', otherwise 'Fake'.
        # Adjust this threshold and logic based on your specific model's output.
        if predictions[0][0] > 0.5: # Assuming a single output neuron for binary classification
            return "Authentic"
        else:
            return "Fake"

    except Exception as e:
        print(f"Error during CNN prediction: {e}")
        return "Error: Prediction Failed"

# Define the API endpoint for currency detection
@app.route('/detect_currency', methods=['POST'])
def detect_currency():
    """
    Handles the POST request for currency detection.
    Expects a JSON body with an 'image' field containing base64 encoded image data.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    image_data_base64 = data.get('image')

    if not image_data_base64:
        return jsonify({"error": "No image data provided"}), 400

    # Perform actual image preprocessing
    preprocessed_image = preprocess_image_for_cnn(image_data_base64)
    if preprocessed_image is None:
        return jsonify({"error": "Failed to preprocess image"}), 500

    # Perform actual CNN prediction
    prediction = predict_currency(preprocessed_image)

    # Return the prediction as a JSON response
    return jsonify({"prediction": prediction})

# Run the Flask app
if __name__ == '__main__':
    # The debug mode is useful for development, but should be False in production
    app.run(debug=True, port=5000)
