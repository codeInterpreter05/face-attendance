import streamlit as st
import cv2
import numpy as np
import requests

# Define the FastAPI endpoints
UPLOAD_FACE_URL = "http://localhost:8000/upload_face/"
RECOGNIZE_FACE_URL = "http://localhost:8000/recognize/"

st.title("Face Recognition with FastAPI and Streamlit")

# Upload face image and name
st.header("Upload Face Image")
name = st.text_input("Enter your name")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file and name:
    # Convert the uploaded file to a format suitable for FastAPI
    files = {"file": (uploaded_file.name, uploaded_file, "image/jpeg")}
    data = {"name": name}
    response = requests.post(UPLOAD_FACE_URL, data=data, files=files)
    if response.status_code == 200:
        st.success(f"Face encoding for {name} added successfully.")
    else:
        st.error(f"Failed to upload face: {response.json()}")

# Recognize face using webcam
st.header("Recognize Face with Webcam")
run = st.checkbox("Run Webcam")

if run:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    recognized_message = st.empty()
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from webcam.")
            break

        # Convert the frame to bytes for FastAPI
        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(RECOGNIZE_FACE_URL, files={"file": img_encoded.tobytes()})

        if response.status_code == 200:
            result = response.json()
            if "message" in result and "Attendance marked for" in result["message"]:
                recognized_message.success(result["message"])

        stframe.image(frame, channels="BGR")

    cap.release()
    stframe.empty()
    st.write("Video capture stopped.")
