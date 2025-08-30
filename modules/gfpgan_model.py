from __future__ import annotations

import logging
import os

import torch

from modules import (
    devices,
    errors,
    face_restoration,
    face_restoration_utils,
    modelloader,
    shared,
)

logger = logging.getLogger(__name__)
model_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
model_download_name = "GFPGANv1.4.pth"
gfpgan_face_restorer: face_restoration.FaceRestoration | None = None


class FaceRestorerGFPGAN(face_restoration_utils.CommonFaceRestoration):
    def name(self):
        return "GFPGAN"

    def get_device(self):
        return devices.get_optimal_device()

    def load_net(self) -> torch.nn.Module:
        for model_path in modelloader.load_models(
            model_path=self.model_path,
            model_url=model_url,
            download_name=model_download_name,
            ext_filter=['.pth'],
            name_filter='GFPGAN'
        ):
            return modelloader.load_spandrel_model(
                model_path,
                device=self.get_device(),
                expected_architecture='GFPGAN',
            ).model
        raise ValueError("No GFPGAN model found")

    def restore(self, np_image, w: float | None = None):
        def restore_face(cropped_face_t):
            assert self.net is not None
            output = self.net(cropped_face_t, return_rgb=False)
            return output[0]

        return self.restore_with_helper(np_image, restore_face)


def setup_model(model_path: str) -> None:
    global gfpgan_face_restorer
    try:
        gfpgan_face_restorer = FaceRestorerGFPGAN(model_path=model_path)
        shared.face_restorers.append(gfpgan_face_restorer)
    except Exception:
        errors.report("Error setting up GFPGAN", exc_info=True)