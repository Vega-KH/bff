import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from modules.face_extractor import create_face_helper, extract_faces
from modules.devices import get_optimal_device

st.title("BFF - Better Face Fixer")

# Create a cache directory if it doesn't exist
if not os.path.exists('.cache'):
    os.makedirs('.cache')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    pil_image = Image.open(uploaded_file)
    np_image = np.array(pil_image)
    
    st.image(np_image, caption='Uploaded Image.', use_container_width=True)

    device = get_optimal_device()
    face_helper = create_face_helper(device)
    
    cropped_faces = extract_faces(np_image, face_helper)

    st.write(f"Found {len(cropped_faces)} faces:")

    for i, face in enumerate(cropped_faces):
        face_path = f".cache/face_{i}.png"
        # Convert BGR to RGB before saving
        cv2.imwrite(face_path, cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        st.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), caption=f'Face {i+1}', use_container_width=None)