import streamlit as st
import cv2
import numpy as np
from predict import predict_language

st.title("📝 Handwritten Language Detection System")

option = st.radio("Choose Input Method", ["Upload Image", "Use Camera"])

if option == "Upload Image":
    file = st.file_uploader("Upload handwritten image")
    if file:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        st.image(image, channels="BGR")

        lang, conf = predict_language(image)
        st.success(f"Predicted Language: {lang}")
        st.write(f"Confidence: {conf:.2f}")

if option == "Use Camera":
    camera = st.camera_input("Take Photo")
    if camera:
        image = cv2.imdecode(np.frombuffer(camera.read(), np.uint8), 1)
        st.image(image, channels="BGR")

        lang, conf = predict_language(image)
        st.success(f"Predicted Language: {lang}")
        st.write(f"Confidence: {conf:.2f}")