import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename
import lime
import lime.lime_image
import shap
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# Set Matplotlib backend
plt.switch_backend('Agg')

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the trained model
model = tf.keras.models.load_model('fake_face_detection_model.h5')

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess image for model input
def preprocess_image(file_path, target_size=(128, 128)):
    try:
        img = tf.keras.preprocessing.image.load_img(file_path, target_size=target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize
        return img_array
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        return None

# Home route to render the upload form
@app.route('/')
def index():
    return render_template('index.html')

# LIME explanation function
def generate_lime_explanation(image, model, save_path):
    explainer = lime.lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image[0].astype('double'), model.predict, top_labels=1, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False)
    img_boundry = mark_boundaries(temp, mask)

    plt.figure(figsize=(10, 10))
    plt.imshow(img_boundry)
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

# Advanced SHAP explanation function using DeepExplainer with proper scaling
def generate_shap_explanation(image, model, save_path):
    # Using SHAP's DeepExplainer for deep learning models
    explainer = shap.DeepExplainer(model, image)

    # Compute SHAP values
    shap_values = explainer.shap_values(image)

    # Rescale SHAP values to fit within the range [-1, 1]
    shap_values_rescaled = np.sum(shap_values[0], axis=-1)
    shap_values_rescaled = shap_values_rescaled / np.max(np.abs(shap_values_rescaled))  # Normalize between -1 and 1

    # Plot the original image with SHAP values as an overlay (heatmap)
    plt.figure(figsize=(10, 10))

    # Display original image
    plt.subplot(1, 2, 1)
    plt.imshow(image[0])
    plt.title("Original Image")
    plt.axis('off')

    # Overlay SHAP values
    plt.subplot(1, 2, 2)
    plt.imshow(image[0])
    plt.imshow(shap_values_rescaled, cmap='coolwarm', alpha=0.6)  # Set alpha for transparency
    plt.colorbar(label='SHAP Value')  # Add colorbar to represent SHAP value scale
    plt.title("SHAP Explanation")
    plt.axis('off')

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # Save the SHAP explanation image
    plt.close()

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the uploaded image
            img_array = preprocess_image(file_path)
            if img_array is None:
                return jsonify({'error': 'Image preprocessing failed'}), 500

            # Make a prediction
            prediction = model.predict(img_array)
            pred_class = "fake" if prediction[0][0] > 0.4 else "real"
            prob = float(prediction[0][0])

            # Generate LIME and SHAP explanations
            lime_path = os.path.join(app.config['UPLOAD_FOLDER'], 'lime_explanation.png')
            shap_path = os.path.join(app.config['UPLOAD_FOLDER'], 'shap_explanation.png')

            generate_lime_explanation(img_array, model, lime_path)
            generate_shap_explanation(img_array, model, shap_path)

            # Return result
            return jsonify({
                'prediction': pred_class,
                'probability': prob,
                'lime_path': lime_path,
                'shap_path': shap_path
            })

        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({'error': 'Prediction failed'}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
