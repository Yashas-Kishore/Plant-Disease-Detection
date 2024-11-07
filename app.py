import os
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2

app = Flask(__name__)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
IMAGE_SIZE = 64

UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def read_and_resize_image(filepath, image_size):
    img = cv2.imread(filepath)
    resized_img = cv2.resize(img, (image_size, image_size))
    resized_img = resized_img.astype('float32') / 255.0
    return resized_img

def predict_disease(image_path, model):
    input_image = read_and_resize_image(image_path, IMAGE_SIZE)
    input_image = np.expand_dims(input_image, axis=0)
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions)
    disease_class = ['Apple_scab','Apple_black_rot','Apple_cedar_apple_rust','Apple_healthy','Background_without_leaves','Blueberry_healthy','Cherry_powdery_mildew','Cherry_healthy','Corn_gray_leaf_spot','Corn_common_rust','Corn_northern_leaf_blight','Corn_healthy','Grape_black_rot','Grape_black_measles','Grape_leaf_blight','Grape_healthy','Orange_haunglongbing','Peach_bacterial_spot','Peach_healthy','Pepper_bacterial_spot','Pepper_healthy','Potato_early_blight','Potato_healthy','Potato_late_blight','Raspberry_healthy','Soybean_healthy','Squash_powdery_mildew','Strawberry_healthy','Strawberry_leaf_scorch','Tomato_bacterial_spot','Tomato_early_blight','Tomato_healthy','Tomato_late_blight','Tomato_leaf_mold','Tomato_septoria_leaf_spot','Tomato_spider_mites_two-spotted_spider_mite','Tomato_target_spot','Tomato_mosaic_virus','Tomato_yellow_leaf_curl_virus']
    predicted_disease = disease_class[predicted_class_index]
    return predicted_disease

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'imageUpload' not in request.files:
        return "No file uploaded", 400

    file = request.files['imageUpload']
    if file.filename == '':
        return "No selected file", 400

    # Save the uploaded file to the uploads directory
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(upload_path)

    # Load the model
    model = load_model('./my_model.keras')

    # Get the predicted disease
    predicted_disease = predict_disease(upload_path, model)

    # Render the result template with prediction and image filename
    return render_template('result.html', predicted_disease=predicted_disease, image_filename=file.filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
