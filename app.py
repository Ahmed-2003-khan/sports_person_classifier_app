import streamlit as st
from PIL import Image
import pickle
import numpy as np
import cv2
import pywt
import json
import face_recognition

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

# Function to get cropped image if a face is detected
def get_cropped_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    face_locations = face_recognition.face_locations(img)
    if face_locations:
        top, right, bottom, left = face_locations[0]
        roi_color = img[top:bottom, left:right]
        return roi_color
    return None

# Load your model and label dictionary
model_path = "model.pkl"
labels_path = "class_dictionary.json"

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(labels_path, 'r') as labels_file:
    labels = json.load(labels_file)

# Define the Streamlit app
st.set_page_config(page_title="Cricket Player Face Classification", layout="wide", page_icon="🏏")

st.title("Cricket Player Face Classification")
st.markdown("Upload a photo to classify the cricket player. Ensure the face is clear for best results.")

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
        cropped_image = get_cropped_image("temp_image.jpg")

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
                st.subheader("Uploaded pic")
                st.image(uploaded_file, caption='Cropped Face detected', use_column_width=True)
            
            with col2:
                st.subheader("Prediction")
                st.write(f"**Predicted Class:** {predicted_class}")
                
        else:
            st.error("No face detected. Please upload a clearer photo.")
