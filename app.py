import streamlit as st
import os
import cv2
import numpy as np
import io
from PIL import Image
from datetime import datetime
from streamlit_image_comparison import image_comparison

from modules.face_extractor import create_face_helper, extract_faces, paste_faces
from modules.devices import get_optimal_device
from modules.gfpgan_model import setup_model as setup_gfpgan, gfpgan_face_restorer
from modules.codeformer_model import setup_model as setup_codeformer, codeformer_face_restorer
from modules import shared, images
from modules.merger import merge_images_hybrid, precompute_error_maps
from modules.esrgan_model import setup_model as setup_esrgan, esrgan_upscaler
from modules.upscaler import UpscalerNone

# --- Page Config ---
st.set_page_config(layout="wide")

# --- Model Setup ---
# This should only run once, and we'll use a function to keep it clean.
@st.cache_resource
def load_models():
    """
    Loads all the required models into memory, and caches them.
    """
    # Create directories if they don't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('.cache'):
        os.makedirs('.cache')

    shared.face_restorers.clear()
    shared.sd_upscalers.clear()
    setup_gfpgan('models')
    setup_codeformer('models')
    setup_esrgan('models/ESRGAN')
    shared.sd_upscalers.insert(0, UpscalerNone().scalers[0])
    
    # Also create the face helper here, as it's model-dependent
    device = get_optimal_device()
    face_helper = create_face_helper(device)
    return face_helper

# Load models and get the face_helper
face_helper = load_models()


# --- Main App ---
# --- Session State ---
def initialize_session_state():
    """
    Initializes the session state with default values.
    """
    if 'app_state' not in st.session_state:
        st.session_state.app_state = {
            'uploaded_file_name': None,
            'original_image': None,
            'image_with_boxes': None,
            'final_image': None,
            'detected_faces': [],
            'active_face_index': None,
        }

def reset_session_state():
    """
    Resets the session state for a new image upload.
    """
    st.session_state.app_state = {
        'uploaded_file_name': None,
        'original_image': None,
        'image_with_boxes': None,
        'final_image': None,
        'detected_faces': [],
        'active_face_index': None,
    }

# --- Image Processing ---
def draw_faces_on_image(image: np.ndarray, faces: list, active_face_index: int | None) -> np.ndarray:
    """
    Draws bounding boxes on the image for each detected face.
    Highlights the active face.
    """
    image_with_boxes = image.copy()
    for i, face_data in enumerate(faces):
        box = face_data['box']
        color = (255, 255, 0) if i == active_face_index else (0, 255, 0) # Highlight active face in yellow
        thickness = 4 if i == active_face_index else 2
        
        # Draw a rectangle around the face
        cv2.rectangle(image_with_boxes, (box[0], box[1]), (box[2], box[3]), color, thickness)
        # Add a label for the face number
        cv2.putText(image_with_boxes, f"Face {i+1}", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image_with_boxes

def process_active_face(face_index: int):
    """
    Runs restoration, error map computation, and initial merge for the selected face.
    """
    app_state = st.session_state.app_state
    face_data = app_state['detected_faces'][face_index]

    # Only run if not already processed
    if face_data['restored_gfpgan'] is not None:
        return

    with st.spinner(f"Restoring Face {face_index + 1}..."):
        original_face_bgr = face_data['cropped']
        
        if gfpgan_face_restorer is None or codeformer_face_restorer is None:
            st.error("Models not loaded correctly. Please restart the application.")
            return

        # Restore with GFPGAN
        gfpgan_restored = gfpgan_face_restorer.restore(original_face_bgr)
        face_data['restored_gfpgan'] = gfpgan_restored

        # Restore with CodeFormer (weight is hardcoded to 0.5)
        codeformer_restored = codeformer_face_restorer.restore(original_face_bgr, w=0.5)
        face_data['restored_codeformer'] = codeformer_restored

    with st.spinner(f"Analyzing Face {face_index + 1}..."):
        # Pre-compute error maps
        original_float = original_face_bgr.astype(np.float32) / 255.0
        gfpgan_float = gfpgan_restored.astype(np.float32) / 255.0
        codeformer_float = codeformer_restored.astype(np.float32) / 255.0
        face_data['error_maps'] = precompute_error_maps(original_float, gfpgan_float, codeformer_float)

        # Run initial merge
        merged_face_float = merge_images_hybrid(
            original_float,
            gfpgan_float,
            codeformer_float,
            face_data['error_maps'],
            alpha=face_data['merge_settings']['alpha'],
            delta_threshold=face_data['merge_settings']['delta_threshold'],
            tie_epsilon=face_data['merge_settings']['tie_epsilon'],
            return_stats=False
        )
        face_data['merged'] = (merged_face_float * 255).clip(0, 255).astype(np.uint8)


def display_sidebar_controls(face_data: dict):
    """
    Displays the merge control sliders in the sidebar and handles updates.
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("Merge Settings")

    settings = face_data['merge_settings']
    
    new_alpha = st.sidebar.slider(
        "Structure vs. Color (Alpha)", 0.0, 1.0, settings['alpha'], 0.05,
        help="Lower values prioritize structural integrity (SSIM), higher values prioritize color accuracy (MSE)."
    )
    new_delta_threshold = st.sidebar.slider(
        "Fidelity Threshold", 0.0, 1.0, settings['delta_threshold'], 0.05,
        help="How much difference from the original structure is allowed. Lower is stricter."
    )
    new_tie_epsilon = st.sidebar.slider(
        "Blending", min_value=0.0, max_value=0.01, value=settings['tie_epsilon'], step=1e-3,
        help="Higher values lead to more blending for smoother results.",
        format="%.4f"
    )

    if (new_alpha != settings['alpha'] or
        new_delta_threshold != settings['delta_threshold'] or
        new_tie_epsilon != settings['tie_epsilon']):
        
        settings['alpha'] = new_alpha
        settings['delta_threshold'] = new_delta_threshold
        settings['tie_epsilon'] = new_tie_epsilon

        # Re-run the merge with new settings
        with st.spinner("Updating merge..."):
            original_float = face_data['cropped'].astype(np.float32) / 255.0
            gfpgan_float = face_data['restored_gfpgan'].astype(np.float32) / 255.0
            codeformer_float = face_data['restored_codeformer'].astype(np.float32) / 255.0
            
            merged_face_float = merge_images_hybrid(
                original_float,
                gfpgan_float,
                codeformer_float,
                face_data['error_maps'],
                alpha=new_alpha,
                delta_threshold=new_delta_threshold,
                tie_epsilon=new_tie_epsilon,
                return_stats=False
            )
            face_data['merged'] = (merged_face_float * 255).clip(0, 255).astype(np.uint8)
        # The widget interaction causes a natural rerun, so this is not needed.
        # st.rerun()


# --- Main App ---
def main():
    """
    The main function for the Streamlit app.
    """
    initialize_session_state()
    app_state = st.session_state.app_state

    # --- Sidebar ---
    with st.sidebar:
        #st.sidebar.image(load_image("logo.png"), width='stretch')
        #Image.open(Path(get_project_root()) / f"references/{image_name}")
        #Image.open('assets/logo.jpg')
        st.sidebar.image("assets/logo.jpg", width='stretch')
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        # Display merge controls if a face is active
        if app_state.get('active_face_index') is not None:
            display_sidebar_controls(app_state['detected_faces'][app_state['active_face_index']])
        
        # --- Finalize and Upscale ---
        if any(f['merged'] is not None for f in app_state.get('detected_faces', [])):
            st.sidebar.markdown("---")
            st.sidebar.subheader("Finalize")
            if st.sidebar.button("Finalize & Paste"):
                with st.spinner("Pasting faces into final image..."):
                    # Ensure all faces are processed before pasting
                    for i in range(len(app_state['detected_faces'])):
                        if app_state['detected_faces'][i]['merged'] is None:
                            process_active_face(i)
                    
                    restored_faces = [f['merged'] for f in app_state['detected_faces']]
                    final_image = paste_faces(face_helper, restored_faces)
                    app_state['final_image'] = final_image
                st.rerun()

            # --- Upscaling ---
            st.sidebar.markdown("---")
            st.sidebar.title("Upscaling")

            if shared.sd_upscalers:
                upscaler_names = [upscaler.name for upscaler in shared.sd_upscalers]
                st.sidebar.selectbox("Select Upscaler", upscaler_names, key='selected_upscaler')
                
                upscale_button_disabled = app_state.get('final_image') is None
                if st.sidebar.button("Upscale Final Image", disabled=upscale_button_disabled):
                    selected_upscaler_name = st.session_state.selected_upscaler
                    if app_state['final_image'] is not None:
                        # Ensure final image has same size as original
                        h, w, _ = app_state['original_image'].shape
                        final_image_resized = cv2.resize(app_state['final_image'], (w, h))

                        selected_upscaler = next((u for u in shared.sd_upscalers if u.name == selected_upscaler_name), None)
                        if selected_upscaler and esrgan_upscaler and selected_upscaler.name != "None":
                            with st.spinner(f"Upscaling with {selected_upscaler_name}..."):
                                # Convert BGR numpy array to RGB PIL Image for the upscaler
                                pil_image = Image.fromarray(cv2.cvtColor(final_image_resized, cv2.COLOR_BGR2RGB))
                                upscaled_image = esrgan_upscaler.upscale(pil_image, selected_upscaler.scale, selected_upscaler.data_path)
                                
                                date_str = datetime.now().strftime('%Y-%m-%d')
                                i = 1
                                while True:
                                    filename = f"{date_str}-{i}.jpg"
                                    save_path = os.path.join('results', filename)
                                    if not os.path.exists(save_path):
                                        break
                                    i += 1
                                
                                if upscaled_image.mode == 'RGBA':
                                    upscaled_image = upscaled_image.convert('RGB')

                                upscaled_image.save(save_path, 'jpeg')
                                st.success(f"Image saved to {save_path}")

                        elif selected_upscaler and selected_upscaler.name == "None":
                            st.info("Please select an upscaler.")
                        else:
                            st.error("Selected upscaler not found.")


    # --- Main Content ---
    if uploaded_file is None:
        st.markdown("<h3 style='text-align: center;'>Upload an image to get started</h3>", unsafe_allow_html=True)
        return

    # --- Image Upload and State Reset Logic ---
    # This is the most important check. If a new file is uploaded, reset everything.
    if app_state.get('uploaded_file_name') != uploaded_file.name:
        reset_session_state()
        app_state = st.session_state.app_state # Re-fetch the state dictionary
        app_state['uploaded_file_name'] = uploaded_file.name
        
        # Read the image
        pil_image = Image.open(uploaded_file)
        np_image = np.array(pil_image)
        if np_image.shape[2] == 4: # Handle RGBA images from PNGs
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGBA2BGR)
        else:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        app_state['original_image'] = np_image
        
        # Detect faces
        with st.spinner("Detecting faces..."):
            cropped_faces = extract_faces(np_image, face_helper)
            
            face_locations = face_helper.det_faces
            
            for i, (loc, crop) in enumerate(zip(face_locations, cropped_faces)):
                app_state['detected_faces'].append({
                    'box': [int(x) for x in loc[:4]],
                    'cropped': crop,
                    'restored_gfpgan': None,
                    'restored_codeformer': None,
                    'merged': None,
                    'error_maps': None,
                    'merge_settings': {
                        'alpha': 0.5,
                        'delta_threshold': 0.35,
                        'tie_epsilon': 1e-3,
                    }
                })
        
        if app_state['detected_faces']:
            app_state['image_with_boxes'] = draw_faces_on_image(np_image, app_state['detected_faces'], app_state['active_face_index'])
        else:
            app_state['image_with_boxes'] = np_image
        
        # Force a rerun after processing a new image to ensure the UI updates correctly
        st.rerun()

    # --- Display Logic ---
    # Now that the state is correct, decide what to show the user.
    
    # 1. Show the final comparison slider if it exists
    if app_state.get('final_image') is not None:
        st.header("Final Result")
        image_comparison(
            img1=cv2.cvtColor(app_state['original_image'], cv2.COLOR_BGR2RGB),
            img2=cv2.cvtColor(app_state['final_image'], cv2.COLOR_BGR2RGB),
            label1="Original",
            label2="Restored",
        )
        
        # Download button for the final image
        h, w, _ = app_state['original_image'].shape
        final_image_resized = cv2.resize(app_state['final_image'], (w, h))
        
        final_img_pil = Image.fromarray(cv2.cvtColor(final_image_resized, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        final_img_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(
            label="Download Restored Image",
            data=byte_im,
            file_name=f"BFF_restored_{app_state['uploaded_file_name']}",
            mime="image/png"
        )
        return # Stop further rendering

    # 2. Show the main editing view if an image has been processed
    elif app_state.get('image_with_boxes') is not None:
        st.image(cv2.cvtColor(app_state['image_with_boxes'], cv2.COLOR_BGR2RGB), caption='Uploaded Image with Detected Faces', width='stretch')
        if not app_state['detected_faces']:
            st.warning("No faces were detected in the uploaded image.")
        else:
            st.info(f"Detected {len(app_state['detected_faces'])} faces. Select a face to begin editing.")

        # Face Selection Buttons
        if app_state['detected_faces']:
            num_faces = len(app_state['detected_faces'])
            cols = st.columns(num_faces)
            for i in range(num_faces):
                with cols[i]:
                    if st.button(f"Edit Face {i+1}"):
                        app_state['active_face_index'] = i
                        process_active_face(i)
                        st.rerun()

        # Comparison View for Active Face
        active_face_idx = app_state.get('active_face_index')
        if active_face_idx is not None:
            st.markdown("---")
            st.subheader(f"Editing Face {active_face_idx + 1}")
            
            active_face_data = app_state['detected_faces'][active_face_idx]
            
            if active_face_data['merged'] is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(active_face_data['restored_gfpgan'], caption='GFPGAN Result', width='stretch')
                with col2:
                    st.image(active_face_data['restored_codeformer'], caption='CodeFormer Result', width='stretch')
                with col3:
                    st.image(active_face_data['merged'], caption='Merged Result', width='stretch')
            else:
                st.warning("This face has not been processed yet. Click the button again.")


if __name__ == '__main__':
    main()