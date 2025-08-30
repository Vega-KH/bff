import logging
import numpy as np
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from facexlib.detection import retinaface

from modules.images import bgr_image_to_rgb_tensor, rgb_tensor_to_bgr_image

logger = logging.getLogger(__name__)


def create_face_helper(device) -> FaceRestoreHelper:
    # if hasattr(retinaface, 'device'):
    #     retinaface.device = device
    return FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        use_parse=True,
        device=device,
    )

def extract_faces(np_image: np.ndarray, face_helper: FaceRestoreHelper):
    """
    Find faces in the image using face_helper and return the cropped faces.
    """
    np_image = np_image[:, :, ::-1]
    
    try:
        logger.debug("Detecting faces...")
        face_helper.clean_all()
        face_helper.read_image(np_image)
        face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        face_helper.align_warp_face()
        logger.debug("Found %d faces", len(face_helper.cropped_faces))
        return face_helper.cropped_faces
    finally:
        face_helper.clean_all()