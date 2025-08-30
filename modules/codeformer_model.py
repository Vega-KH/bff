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
model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
model_download_name = 'codeformer-v0.1.0.pth'
codeformer_face_restorer: face_restoration.FaceRestoration | None = None


class FaceRestorerCodeFormer(face_restoration_utils.CommonFaceRestoration):
    def name(self):
        return "CodeFormer"

    def load_net(self) -> torch.nn.Module:
        for model_path in modelloader.load_models(
            model_path=self.model_path,
            model_url=model_url,
            download_name=model_download_name,
            ext_filter=['.pth'],
            name_filter='codeformer'
        ):
            return modelloader.load_spandrel_model(
                model_path,
                device=self.get_device(),
                expected_architecture='CodeFormer',
            ).model
        raise ValueError("No codeformer model found")

    def get_device(self):
        return devices.device_codeformer

    def restore(self, np_image, w: float | None = None):
        if w is None:
            w = 0.5

        def restore_face(cropped_face_t):
            assert self.net is not None
            return self.net(cropped_face_t, weight=w, adain=True)[0]

        return self.restore_with_helper(np_image, restore_face)


def setup_model(model_path: str) -> None:
    global codeformer_face_restorer
    try:
        codeformer_face_restorer = FaceRestorerCodeFormer(model_path)
        shared.face_restorers.append(codeformer_face_restorer)
    except Exception:
        errors.report("Error setting up CodeFormer", exc_info=True)