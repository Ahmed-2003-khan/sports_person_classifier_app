import streamlit as st
from PIL import Image
import pickle
import numpy as np
import cv2
import pywt
import json
import cvlib as cv
from cvlib.object_detection import draw_bbox
import os

# Function to apply wavelet transform
def w2d(img, mode='haar', level=1):
    imArray = img
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    return imArray_H

# Function to get cropped image if a face is detected using cvlib
def get_cropped_image_if_face_detected(image_path):
    img = cv2.imread(image_path)
    faces, confidences = cv.detect_face(img)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_img = img[y:h, x:w]
        return face_img
    return None

# Load your model and label dictionary
model_path = "model_compressed.pklz"
labels_path = "class_dictionary.json"

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(labels_path, 'r') as labels_file:
    labels = json.load(labels_file)

# Define the Streamlit app
st.set_page_config(page_title="Cricket Player Face Classification", layout="wide", page_icon="🏏")

st.title("Cricket Player Face Classification")
st.markdown("Upload a photo to classify the cricket player from the following list:")

# Show player images
player_images = {
    "Imran Khan": "imran_khan.jpg",
    "Kapil Dev": "kapil_dev.jpg",
    "Shoaib Akhtar": "shoaib_akhtar.jpeg",
    "Virat Kohli": "virat_kohli.jpeg",
    "MS Dhoni": "ms_dhoni.jpeg",
    "Wasim Akram": "wasim_akram.jpeg"
}

# Resize and display player images
resize_dimensions = (200, 200)  # Set the desired size here

cols = st.columns(3)
for i, (player_name, image_path) in enumerate(player_images.items()):
    with cols[i % 3]:
        if os.path.exists(image_path):
            img = Image.open(image_path)
            img = img.resize(resize_dimensions)
            st.image(img, caption=player_name, use_column_width=True)
        else:
            st.warning(f"Image for {player_name} not found at {image_path}.")

st.sidebar.header("Instructions")
st.sidebar.write("1. Upload a clear photo of the face.")
st.sidebar.write("2. The app will detect the face and classify it.")
st.sidebar.write("3. If successful, the predicted class will be shown.")

# File uploader for image
uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner('Processing image...'):
        # Save the uploaded file to disk temporarily
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Get cropped image
        cropped_image = get_cropped_image_if_face_detected("temp_image.jpg")

        if cropped_image is not None:
            # Preprocess the image
            scalled_raw_image = cv2.resize(cropped_image, (32, 32))
            img_har = w2d(cropped_image, 'db1', 5)
            scalled_img_har = cv2.resize(img_har, (32, 32))
            combined_img = np.hstack((scalled_raw_image.flatten(), scalled_img_har.flatten()))
            combined_img = combined_img.reshape(1, -1).astype(float)

            # Predict the class
            prediction = model.predict(combined_img)
            predicted_class = labels[str(prediction[0])]

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded Image")
                st.image(uploaded_file, caption='Cropped Face Detected', use_column_width=True)
            
            with col2:
                st.subheader("Prediction")
                st.write(f"**Predicted Class:** {predicted_class}")
                
        else:
            st.error("No face detected. Please upload a clearer photo.")
