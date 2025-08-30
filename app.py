import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from modules.face_extractor import create_face_helper, extract_faces
from modules.devices import get_optimal_device
from modules.gfpgan_model import setup_model as setup_gfpgan, gfpgan_face_restorer
from modules.codeformer_model import setup_model as setup_codeformer, codeformer_face_restorer
from modules import shared

st.title("BFF - Better Face Fixer")

# Create a cache directory if it doesn't exist
if not os.path.exists('.cache'):
    os.makedirs('.cache')

# Create a models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

setup_gfpgan('models')
setup_codeformer('models')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    pil_image = Image.open(uploaded_file)
    np_image = np.array(pil_image)
    
    st.image(np_image, caption='Uploaded Image.', use_container_width=True)

    device = get_optimal_device()
    face_helper = create_face_helper(device)
    
    cropped_faces = extract_faces(np_image, face_helper)

    if st.button('Restore Faces'):
        st.write("Restoring faces...")
        gfpgan_faces = []
        codeformer_faces = []

        if gfpgan_face_restorer is not None and codeformer_face_restorer is not None:
            for face in cropped_faces:
                gfpgan_faces.append(gfpgan_face_restorer.restore(face))
                codeformer_faces.append(codeformer_face_restorer.restore(face))
        else:
            st.error("Models not loaded.")

        if gfpgan_faces and codeformer_faces:
            st.write("Restored faces:")
            for i, (gfpgan, codeformer) in enumerate(zip(gfpgan_faces, codeformer_faces)):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(gfpgan, cv2.COLOR_BGR2RGB), caption=f'GFPGAN {i+1}', use_container_width=True)
                with col2:
                    st.image(cv2.cvtColor(codeformer, cv2.COLOR_BGR2RGB), caption=f'CodeFormer {i+1}', use_container_width=True)
    else:
        st.write(f"Found {len(cropped_faces)} faces:")
        for i, face in enumerate(cropped_faces):
            st.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), caption=f'Face {i+1}', use_container_width=None)