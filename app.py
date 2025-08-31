import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from modules.face_extractor import create_face_helper, extract_faces, paste_faces
from modules.devices import get_optimal_device
from modules.gfpgan_model import setup_model as setup_gfpgan, gfpgan_face_restorer
from modules.codeformer_model import setup_model as setup_codeformer, codeformer_face_restorer
from modules import shared, images
from modules.merger import merge_images_hybrid, precompute_error_maps
from modules.esrgan_model import setup_model as setup_esrgan, esrgan_upscaler
from modules.upscaler import UpscalerNone

st.set_page_config(layout="wide")
st.title("BFF - Better Face Fixer")

# --- Image Display Area ---
if st.session_state.get('final_image') is not None:
    st.image(st.session_state.final_image, caption='Final Result', use_column_width=True)
elif st.session_state.get('original_image') is not None:
    st.image(st.session_state.original_image, caption='Uploaded Image', use_column_width=True)

# Create a cache directory if it doesn't exist
if not os.path.exists('.cache'):
    os.makedirs('.cache')

# Create a models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('results'):
    os.makedirs('results')

# --- Model Setup ---
# This should only run once
if 'models_loaded' not in st.session_state:
    shared.face_restorers.clear()
    shared.sd_upscalers.clear()
    setup_gfpgan('models')
    setup_codeformer('models')
    setup_esrgan('models/ESRGAN')
    shared.sd_upscalers.insert(0, UpscalerNone().scalers[0])
    st.session_state.models_loaded = True

# Initialize session state
if 'gfpgan_faces' not in st.session_state:
    st.session_state.gfpgan_faces = []
if 'codeformer_faces' not in st.session_state:
    st.session_state.codeformer_faces = []
if 'cropped_faces' not in st.session_state:
    st.session_state.cropped_faces = []

if 'merged_faces' not in st.session_state:
    st.session_state.merged_faces = []
if 'codeformer_weights' not in st.session_state:
    st.session_state.codeformer_weights = []
if 'alpha' not in st.session_state:
    st.session_state.alpha = 0.5
if 'delta_threshold' not in st.session_state:
    st.session_state.delta_threshold = 0.35
if 'tie_epsilon' not in st.session_state:
    st.session_state.tie_epsilon = 1e-3
if 'merge_stats' not in st.session_state:
    st.session_state.merge_stats = []
if 'error_maps_cache' not in st.session_state:
    st.session_state.error_maps_cache = []
if 'face_helper' not in st.session_state:
    st.session_state.face_helper = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'final_image' not in st.session_state:
    st.session_state.final_image = None
if 'selected_upscaler' not in st.session_state:
    st.session_state.selected_upscaler = "None"
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None


def run_merge(face_index: int):
    """
    Merges the specified face using cached error maps and updates session state.
    """
    if not all(k in st.session_state for k in ['cropped_faces', 'gfpgan_faces', 'codeformer_faces', 'error_maps_cache']):
        return
    if len(st.session_state.error_maps_cache) <= face_index:
        return # Maps not computed yet

    original_face = st.session_state.cropped_faces[face_index].astype(np.float32) / 255.0
    gfpgan_face = st.session_state.gfpgan_faces[face_index].astype(np.float32) / 255.0
    codeformer_face = st.session_state.codeformer_faces[face_index].astype(np.float32) / 255.0
    
    error_maps = st.session_state.error_maps_cache[face_index]

    merged_face_float, stats = merge_images_hybrid(
        original_face,
        gfpgan_face,
        codeformer_face,
        error_maps,
        alpha=st.session_state.alpha,
        delta_threshold=st.session_state.delta_threshold,
        tie_epsilon=st.session_state.tie_epsilon,
        return_stats=True
    )
    
    # Convert back to uint8 for display
    merged_face_uint8 = (merged_face_float * 255).clip(0, 255).astype(np.uint8)
    
    # Update main image
    if 'merged_faces' not in st.session_state or len(st.session_state.merged_faces) <= face_index:
        st.session_state.merged_faces.append(merged_face_uint8)
    else:
        st.session_state.merged_faces[face_index] = merged_face_uint8
        
    # Update stats
    if 'merge_stats' not in st.session_state or len(st.session_state.merge_stats) <= face_index:
        st.session_state.merge_stats.append(stats)
    else:
        st.session_state.merge_stats[face_index] = stats

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Check if a new file has been uploaded
    if st.session_state.uploaded_file_name != uploaded_file.name:
        images.clear_cache()
        
        # Reset all state for the new image
        pil_image = Image.open(uploaded_file)
        np_image = np.array(pil_image)
        st.session_state.original_image = np_image
        st.session_state.final_image = None
        st.session_state.gfpgan_faces = []
        st.session_state.codeformer_faces = []
        st.session_state.cropped_faces = []
        st.session_state.merged_faces = []
        st.session_state.codeformer_weights = []
        st.session_state.merge_stats = []
        st.session_state.error_maps_cache = []
        
        # Process the new image
        device = get_optimal_device()
        st.session_state.face_helper = create_face_helper(device)
        st.session_state.cropped_faces = extract_faces(np_image, st.session_state.face_helper)

        # Store the new file name *after* processing
        st.session_state.uploaded_file_name = uploaded_file.name
        st.rerun()

    if st.button('Restore Faces'):
        with st.spinner("Restoring faces..."):
            if gfpgan_face_restorer is not None and codeformer_face_restorer is not None:
                st.session_state.gfpgan_faces = [gfpgan_face_restorer.restore(face) for face in st.session_state.cropped_faces]
                st.session_state.codeformer_weights = [0.5] * len(st.session_state.cropped_faces)
                st.session_state.codeformer_faces = [codeformer_face_restorer.restore(face, w=st.session_state.codeformer_weights[i]) for i, face in enumerate(st.session_state.cropped_faces)]
                
                # Pre-compute error maps
                st.session_state.error_maps_cache = []
                with st.spinner("Pre-computing error maps..."):
                    for i in range(len(st.session_state.cropped_faces)):
                        original_face = st.session_state.cropped_faces[i].astype(np.float32) / 255.0
                        gfpgan_face = st.session_state.gfpgan_faces[i].astype(np.float32) / 255.0
                        codeformer_face = st.session_state.codeformer_faces[i].astype(np.float32) / 255.0
                        maps = precompute_error_maps(original_face, gfpgan_face, codeformer_face)
                        st.session_state.error_maps_cache.append(maps)

                # Run the initial merge for all faces
                st.session_state.merged_faces = []
                st.session_state.merge_stats = []
                for i in range(len(st.session_state.cropped_faces)):
                    run_merge(i)
            else:
                st.error("Models not loaded.")

if st.session_state.gfpgan_faces and st.session_state.codeformer_faces:
    st.write("Restored faces:")
    
    for i, (gfpgan, codeformer) in enumerate(zip(st.session_state.gfpgan_faces, st.session_state.codeformer_faces)):    
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(cv2.cvtColor(gfpgan, cv2.COLOR_BGR2RGB), caption=f'GFPGAN Result {i+1}')
        with col2:
            st.image(cv2.cvtColor(codeformer, cv2.COLOR_BGR2RGB), caption=f'CodeFormer Result {i+1}')

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
                        
                        # Re-compute maps for the updated face
                        with st.spinner(f"Re-computing maps for face {i+1}..."):
                            original_face = st.session_state.cropped_faces[i].astype(np.float32) / 255.0
                            gfpgan_face = st.session_state.gfpgan_faces[i].astype(np.float32) / 255.0
                            codeformer_face = st.session_state.codeformer_faces[i].astype(np.float32) / 255.0
                            maps = precompute_error_maps(original_face, gfpgan_face, codeformer_face)
                            st.session_state.error_maps_cache[i] = maps

                        run_merge(i) # Re-run merge after CodeFormer update
                st.rerun()
        
        with col3:
            if st.session_state.merged_faces and len(st.session_state.merged_faces) > i:
                st.image(cv2.cvtColor(st.session_state.merged_faces[i], cv2.COLOR_BGR2RGB), caption=f'Merged Result {i+1}')

    st.sidebar.title("Merge Settings")
    new_alpha = st.sidebar.slider(
        "Structure vs. Color (Alpha)", 0.0, 1.0, st.session_state.alpha, 0.1,
        help="Lower values prioritize structural integrity (SSIM), higher values prioritize color accuracy (MSE)."
    )
    new_delta_threshold = st.sidebar.slider(
        "Fidelity Threshold", 0.0, 1.0, st.session_state.delta_threshold, 0.05,
        help="How much difference from the original structure is allowed. Lower is stricter, and will look more like the original image."
    )
    new_tie_epsilon = st.sidebar.slider(
        "Blending", min_value=0.0, max_value=0.01, value=st.session_state.tie_epsilon, step=1e-3,
        help="Setting a higher value leads to more blending, and produces a smoother result. May reduce artifacts.",
        format="%.4f"
    )

    if (new_alpha != st.session_state.alpha or
        new_delta_threshold != st.session_state.delta_threshold or
        new_tie_epsilon != st.session_state.tie_epsilon):
        st.session_state.alpha = new_alpha
        st.session_state.delta_threshold = new_delta_threshold
        st.session_state.tie_epsilon = new_tie_epsilon
        with st.spinner("Re-merging faces with new settings..."):
            for i in range(len(st.session_state.cropped_faces)):
                run_merge(i)
        st.rerun()

    if st.sidebar.button("Paste Faces to Original"):
        if st.session_state.face_helper is not None and st.session_state.merged_faces:
            with st.spinner("Pasting faces..."):
                final_image = paste_faces(st.session_state.face_helper, st.session_state.merged_faces)
                st.session_state.final_image = final_image
                st.rerun()
        else:
            st.error("Could not paste faces. Please restore faces first.")

    st.sidebar.markdown("---")
    st.sidebar.title("Upscaling")

    if shared.sd_upscalers:
        upscaler_names = [upscaler.name for upscaler in shared.sd_upscalers]
        st.sidebar.selectbox("Select Upscaler", upscaler_names, key='selected_upscaler')
        
        upscale_button_disabled = st.session_state.get('final_image') is None
        if st.sidebar.button("Upscale Final Image", disabled=upscale_button_disabled):
            selected_upscaler_name = st.session_state.selected_upscaler
            if st.session_state.final_image is not None:
                selected_upscaler = next((u for u in shared.sd_upscalers if u.name == selected_upscaler_name), None)
                if selected_upscaler and esrgan_upscaler and selected_upscaler.name != "None":
                    with st.spinner(f"Upscaling with {selected_upscaler_name}..."):
                        pil_image = Image.fromarray(st.session_state.final_image)
                        upscaled_image = esrgan_upscaler.upscale(pil_image, selected_upscaler.scale, selected_upscaler.data_path)
                        
                        # Save the image instead of displaying it
                        date_str = datetime.now().strftime('%Y-%m-%d')
                        
                        # Find the next available sequence number for the filename
                        i = 1
                        while True:
                            filename = f"{date_str}-{i}.jpg"
                            save_path = os.path.join('results', filename)
                            if not os.path.exists(save_path):
                                break
                            i += 1
                        
                        # Convert to RGB if it has an alpha channel, as JPEG doesn't support it
                        if upscaled_image.mode == 'RGBA':
                            upscaled_image = upscaled_image.convert('RGB')

                        upscaled_image.save(save_path, 'jpeg')
                        st.success(f"Image saved to {save_path}")

                elif selected_upscaler and selected_upscaler.name == "None":
                    st.info("Please select an upscaler.")
                else:
                    st.error("Selected upscaler not found.")

    if st.session_state.merge_stats:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Merge Statistics")
        for i, stats in enumerate(st.session_state.merge_stats):
            st.sidebar.markdown(f"**Face {i+1}**")
            st.sidebar.json(stats)

elif 'cropped_faces' in st.session_state and len(st.session_state.cropped_faces) > 0:
    st.write(f"Found {len(st.session_state.cropped_faces)} faces:")
    for i, face in enumerate(st.session_state.cropped_faces):
        st.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), caption=f'Face {i+1}')
