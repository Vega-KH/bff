import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from modules.face_extractor import create_face_helper, extract_faces
from modules.devices import get_optimal_device
from modules.gfpgan_model import setup_model as setup_gfpgan, gfpgan_face_restorer
from modules.codeformer_model import setup_model as setup_codeformer, codeformer_face_restorer
from modules import shared, images

st.title("BFF - Better Face Fixer")

# Create a cache directory if it doesn't exist
if not os.path.exists('.cache'):
    os.makedirs('.cache')

# Create a models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

setup_gfpgan('models')
setup_codeformer('models')

# Initialize session state
if 'gfpgan_faces' not in st.session_state:
    st.session_state.gfpgan_faces = []
# Initialize session state
if 'gfpgan_faces' not in st.session_state:
    st.session_state.gfpgan_faces = []
if 'codeformer_faces' not in st.session_state:
    st.session_state.codeformer_faces = []
if 'cropped_faces' not in st.session_state:
    st.session_state.cropped_faces = []
if 'codeformer_weights' not in st.session_state:
    st.session_state.codeformer_weights = []

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    images.clear_cache()
    pil_image = Image.open(uploaded_file)
    np_image = np.array(pil_image)
    
    st.image(np_image, caption='Uploaded Image.', width='stretch')

    device = get_optimal_device()
    face_helper = create_face_helper(device)
    
    st.session_state.cropped_faces = extract_faces(np_image, face_helper)

    if st.button('Restore Faces'):
        with st.spinner("Restoring faces..."):
            if gfpgan_face_restorer is not None and codeformer_face_restorer is not None:
                st.session_state.gfpgan_faces = [gfpgan_face_restorer.restore(face) for face in st.session_state.cropped_faces]
                st.session_state.codeformer_weights = [0.5] * len(st.session_state.cropped_faces)
                st.session_state.codeformer_faces = [codeformer_face_restorer.restore(face, w=st.session_state.codeformer_weights[i]) for i, face in enumerate(st.session_state.cropped_faces)]
            else:
                st.error("Models not loaded.")

if st.session_state.gfpgan_faces and st.session_state.codeformer_faces:
    st.write("Restored faces:")
    
    for i, (gfpgan, codeformer) in enumerate(zip(st.session_state.gfpgan_faces, st.session_state.codeformer_faces)):
        images.save_image_to_cache(gfpgan, f"face{i+1}_gfpgan.png")
        images.save_image_to_cache(codeformer, f"face{i+1}_codeformer.png")
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(gfpgan, cv2.COLOR_BGR2RGB), caption=f'GFPGAN {i+1}', width='stretch')
        with col2:
            st.image(cv2.cvtColor(codeformer, cv2.COLOR_BGR2RGB), caption=f'CodeFormer {i+1}', width='stretch')

            weight_options = {'low': 0.25, 'medium': 0.5, 'high': 0.75, 'extreme': 1.0}
            current_weight = st.session_state.codeformer_weights[i]
            current_weight_label = [label for label, weight in weight_options.items() if weight == current_weight][0]

            selected_weight_label = st.select_slider(
                f'CodeFormer Weight Face {i+1}',
                options=list(weight_options.keys()),
                value=current_weight_label,
                key=f"weight_slider_{i}"
            )
            new_weight = weight_options[selected_weight_label]

            if new_weight != current_weight:
                st.session_state.codeformer_weights[i] = new_weight
                with st.spinner(f"Updating CodeFormer for face {i+1}..."):
                    if codeformer_face_restorer is not None:
                        st.session_state.codeformer_faces[i] = codeformer_face_restorer.restore(st.session_state.cropped_faces[i], w=new_weight)
                st.rerun()

else:
    if 'cropped_faces' in st.session_state and len(st.session_state.cropped_faces) > 0:
        st.write(f"Found {len(st.session_state.cropped_faces)} faces:")
        for i, face in enumerate(st.session_state.cropped_faces):
            st.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), caption=f'Face {i+1}', width='content')