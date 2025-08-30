from __future__ import annotations

import logging
import os
from functools import cached_property
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
import gc
from torchvision.transforms.functional import normalize

from modules import devices, errors, face_restoration
from modules.images import bgr_image_to_rgb_tensor, rgb_tensor_to_bgr_image
from modules.face_extractor import create_face_helper

if TYPE_CHECKING:
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper

logger = logging.getLogger(__name__)


class CommonFaceRestoration(face_restoration.FaceRestoration):
    net: torch.nn.Module | None
    model_url: str
    model_download_name: str

    def __init__(self, model_path: str):
        super().__init__()
        self.net = None
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)

    @cached_property
    def face_helper(self) -> FaceRestoreHelper:
        return create_face_helper(self.get_device())

    def send_model_to(self, device):
        if self.net:
            logger.debug("Sending %s to %s", self.net, device)
            self.net.to(device)
        if self.face_helper:
            logger.debug("Sending face helper to %s", device)
            self.face_helper.face_det.to(device)
            self.face_helper.face_parse.to(device)

    def get_device(self):
        raise NotImplementedError("get_device must be implemented by subclasses")

    def load_net(self) -> torch.nn.Module:
        raise NotImplementedError("load_net must be implemented by subclasses")

    def restore(self, np_image: np.ndarray, w: float | None = None) -> np.ndarray:
        raise NotImplementedError("restore must be implemented by subclasses")


    def restore_with_helper(
        self,
        np_image: np.ndarray,
        restore_face: Callable[[torch.Tensor], torch.Tensor],
    ) -> np.ndarray:
        original_resolution = np_image.shape[0:2]

        try:
            if self.net is None:
                self.net = self.load_net()
        except Exception:
            logger.warning("Unable to load face-restoration model", exc_info=True)
            return np_image

        try:
            self.send_model_to(self.get_device())

            face_helper = self.face_helper
            face_helper.clean_all()
            face_helper.read_image(np_image)
            face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
            face_helper.align_warp_face()

            for cropped_face in face_helper.cropped_faces:
                cropped_face_t = bgr_image_to_rgb_tensor(cropped_face / 255.0)
                normalize(cropped_face_t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
                cropped_face_t = cropped_face_t.unsqueeze(0).to(self.get_device())

                try:
                    with torch.no_grad():
                        cropped_face_t = restore_face(cropped_face_t)
                except Exception:
                    errors.report('Failed face-restoration inference', exc_info=True)

                restored_face = rgb_tensor_to_bgr_image(cropped_face_t, min_max=(-1, 1))
                restored_face = (restored_face * 255.0).astype('uint8')
                face_helper.add_restored_face(restored_face)

            face_helper.get_inverse_affine(None)
            img = face_helper.paste_faces_to_input_image()

            if original_resolution != img.shape[0:2]:
                import cv2
                img = cv2.resize(
                    img,
                    (0, 0),
                    fx=original_resolution[1] / img.shape[1],
                    fy=original_resolution[0] / img.shape[0],
                    interpolation=cv2.INTER_LINEAR,
                )

            return img
        finally:
            self.send_model_to(devices.cpu)
            gc.collect()      
            torch.cuda.empty_cache()
            # devices.torch_gc() THIS LINE WAS CAUSING AN ERROR
